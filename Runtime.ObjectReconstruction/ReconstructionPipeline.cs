#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Orchestrates the full reconstruction pipeline: rembg -> preprocess -> forward -> mesh extraction.
    /// All inference is async with multi-frame splitting to avoid main thread stalls.
    /// </summary>
    internal sealed class ReconstructionPipeline : IDisposable
    {
        private readonly int _frameBudgetMs;
        private readonly int _gridResolution;
        private readonly float _densityThreshold;
        private readonly ComputeShader _triplaneShader;
        private readonly ComputeShader _surfaceNetsShader;

        private RembgModel _rembg;
        private ReconstructionModel _reconstruction;
        private DecoderModel _decoder;
        private bool _modelsLoaded;

        internal ReconstructionPipeline(
            int frameBudgetMs,
            int gridResolution,
            float densityThreshold,
            ComputeShader triplaneShader,
            ComputeShader surfaceNetsShader)
        {
            _frameBudgetMs = frameBudgetMs;
            _gridResolution = gridResolution;
            _densityThreshold = densityThreshold;
            _triplaneShader = triplaneShader;
            _surfaceNetsShader = surfaceNetsShader;
        }

        internal async Task LoadModelsAsync(CancellationToken ct)
        {
            if (_modelsLoaded) return;

            _rembg ??= new RembgModel();
            _reconstruction ??= new ReconstructionModel();
            _decoder ??= new DecoderModel();

            await _rembg.LoadAsync(ct);
            await _reconstruction.LoadAsync(ct);
            await _decoder.LoadAsync(ct);

            _modelsLoaded = true;
        }

        internal async Task<Tensor<float>> PreprocessAsync(Texture2D image, CancellationToken ct)
        {
            var readable = MakeReadable(image);
            try
            {
                var mask = await _rembg.InferAsync(readable, ct);
                await AsyncHelper.YieldFrame();
                var result = await ImagePreprocessor.ApplyMaskAndCompositeAsync(readable, mask, 0.85f);
                mask.Dispose();
                return result;
            }
            finally
            {
                if (readable != image)
                    SafeDestroy(readable);
            }
        }

        /// <summary>
        /// Returns a readable copy of the texture if it isn't already readable.
        /// Uses GPU blit + ReadPixels to avoid requiring Read/Write on the import settings.
        /// </summary>
        private static Texture2D MakeReadable(Texture2D src)
        {
            if (src.isReadable) return src;

            var rt = RenderTexture.GetTemporary(src.width, src.height, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(src, rt);
            RenderTexture.active = rt;

            var copy = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false);
            copy.ReadPixels(new Rect(0, 0, src.width, src.height), 0, 0);
            copy.Apply();

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
            return copy;
        }

        internal async Task<Tensor<float>> RunForwardAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            return await _reconstruction.InferAsync(preprocessed, ct);
        }

        internal async Task<Mesh> ExtractMeshAsync(Tensor<float> sceneCodes, CancellationToken ct)
        {
            int res = _gridResolution;
            int totalPoints = res * res * res;
            int featureDim = 120; // 3 planes * 40 channels

            var sampler = new TriplaneGridSampler(_triplaneShader, res);
            var featuresTensor = sampler.SampleFeatures(sceneCodes);
            await AsyncHelper.YieldFrame();
            var featuresData = featuresTensor.DownloadToArray();
            featuresTensor.Dispose();
            sampler.Dispose();
            await AsyncHelper.YieldFrame();
            ct.ThrowIfCancellationRequested();

            int chunkSize = 65536;
            int numChunks = (totalPoints + chunkSize - 1) / chunkSize;
            var density = new float[totalPoints];
            var colors = new Color[totalPoints];
            var budget = new AsyncHelper.FrameBudget();

            for (int c = 0; c < numChunks; c++)
            {
                ct.ThrowIfCancellationRequested();
                int start = c * chunkSize;
                int count = Mathf.Min(chunkSize, totalPoints - start);

                var chunkFeatures = ExtractChunk(featuresData, start, count, featureDim);
                var chunkResult = await _decoder.InferAsync(chunkFeatures, ct);
                chunkFeatures.Dispose();

                var resultData = chunkResult.DownloadToArray();
                chunkResult.Dispose();
                await Task.Run(() => WriteDecoderResults(resultData, density, colors, start, count));

                await budget.YieldIfNeeded();
            }

            var surfaceNets = new DensitySurfaceNets(_surfaceNetsShader, res, _densityThreshold);
            var mesh = await surfaceNets.ExtractAsync(density, colors);
            surfaceNets.Dispose();

            return mesh;
        }

        private static Tensor<float> ExtractChunk(float[] featuresData, int start, int count, int dim)
        {
            var chunkData = new float[count * dim];
            Array.Copy(featuresData, start * dim, chunkData, 0, count * dim);
            var chunk = new Tensor<float>(new TensorShape(count, dim));
            chunk.Upload(chunkData);
            return chunk;
        }

        private static void WriteDecoderResults(
            float[] data, float[] density, Color[] colors, int start, int count)
        {
            for (int i = 0; i < count; i++)
            {
                int idx = start + i;
                float rawDensity = data[i * 4];
                density[idx] = Mathf.Exp(Mathf.Clamp(rawDensity - 1f, -10f, 10f));
                colors[idx] = new Color(
                    Sigmoid(data[i * 4 + 1]),
                    Sigmoid(data[i * 4 + 2]),
                    Sigmoid(data[i * 4 + 3]));
            }
        }

        private static float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

        private static void SafeDestroy(UnityEngine.Object obj)
        {
            if (obj == null) return;
#if UNITY_EDITOR
            if (!Application.isPlaying)
                UnityEngine.Object.DestroyImmediate(obj);
            else
#endif
                UnityEngine.Object.Destroy(obj);
        }

        public void Dispose()
        {
            _rembg?.Dispose();
            _reconstruction?.Dispose();
            _decoder?.Dispose();
            _modelsLoaded = false;
        }
    }
}
#endif
