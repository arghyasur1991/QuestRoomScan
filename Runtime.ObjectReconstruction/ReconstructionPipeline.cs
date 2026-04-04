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
            var sampler = new TriplaneGridSampler(_triplaneShader, res);
            sampler.CacheSceneCodes(sceneCodes);
            await AsyncHelper.YieldFrame();

            int totalPoints = sampler.TotalGridPoints;
            int featureDim = sampler.FeatureDim;
            int chunkSize = 131072;
            var budget = new AsyncHelper.FrameBudget();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            bool gpu = sampler.UseGPU;

            Logger.Info($"[Pipeline] Triplane sampling: {(gpu ? "GPU" : "CPU")}, " +
                        $"res={res}, {totalPoints} points, chunks={chunkSize}");

            // --- Pass 1: sample grid chunks → decode → keep only density ---
            var density = new float[totalPoints];
            int numChunks = (totalPoints + chunkSize - 1) / chunkSize;

            for (int c = 0; c < numChunks; c++)
            {
                ct.ThrowIfCancellationRequested();
                int start = c * chunkSize;
                int count = Mathf.Min(chunkSize, totalPoints - start);

                float[] chunkData;
                if (gpu)
                    chunkData = sampler.SampleGridChunkGPU(start, count);
                else
                    chunkData = await Task.Run(() => sampler.SampleGridChunk(start, count));

                var chunkTensor = UploadChunk(chunkData, count, featureDim);
                var chunkResult = await _decoder.InferAsync(chunkTensor, ct);
                chunkTensor.Dispose();

                var resultData = chunkResult.DownloadToArray();
                chunkResult.Dispose();
                WriteDensityOnly(resultData, density, start, count);
                await budget.YieldIfNeeded();
            }
            Logger.Info($"[Pipeline] Pass 1 density: {sw.ElapsedMilliseconds}ms ({numChunks} chunks)");

            // --- Surface nets: density → mesh geometry ---
            sw.Restart();
            var surfaceNets = new DensitySurfaceNets(_surfaceNetsShader, res, _densityThreshold);
            var mesh = await surfaceNets.ExtractAsync(density);
            surfaceNets.Dispose();
            Logger.Info($"[Pipeline] Surface nets: {sw.ElapsedMilliseconds}ms");
            await AsyncHelper.YieldFrame();
            ct.ThrowIfCancellationRequested();

            // --- Pass 2: query decoder at mesh vertex positions for accurate surface colors ---
            sw.Restart();
            var meshVerts = mesh.vertices;
            int numVerts = meshVerts.Length;
            Logger.Info($"[Pipeline] Pass 2: {numVerts} vertex color queries ({(gpu ? "GPU" : "CPU")})");

            var vertColors = new Color[numVerts];
            int vertChunks = (numVerts + chunkSize - 1) / chunkSize;
            for (int c = 0; c < vertChunks; c++)
            {
                ct.ThrowIfCancellationRequested();
                int start = c * chunkSize;
                int count = Mathf.Min(chunkSize, numVerts - start);

                var vertsSlice = new Vector3[count];
                Array.Copy(meshVerts, start, vertsSlice, 0, count);

                float[] chunkData;
                if (gpu)
                    chunkData = sampler.SampleFeaturesAtPositionsGPU(vertsSlice);
                else
                    chunkData = await Task.Run(() => sampler.SampleFeaturesAtPositions(vertsSlice));

                var chunkTensor = UploadChunk(chunkData, count, featureDim);
                var chunkResult = await _decoder.InferAsync(chunkTensor, ct);
                chunkTensor.Dispose();

                var resultData = chunkResult.DownloadToArray();
                chunkResult.Dispose();
                WriteColorsOnly(resultData, vertColors, start, count);
                await budget.YieldIfNeeded();
            }
            Logger.Info($"[Pipeline] Pass 2 colors: {sw.ElapsedMilliseconds}ms ({vertChunks} chunks)");

            sampler.Dispose();
            mesh.SetColors(vertColors);
            return mesh;
        }

        private static Tensor<float> UploadChunk(float[] data, int count, int dim)
        {
            var tensor = new Tensor<float>(new TensorShape(count, dim));
            tensor.Upload(data);
            return tensor;
        }

        private static void WriteDensityOnly(float[] data, float[] density, int start, int count)
        {
            for (int i = 0; i < count; i++)
            {
                float rawDensity = data[i * 4];
                density[start + i] = Mathf.Exp(Mathf.Clamp(rawDensity - 1f, -10f, 10f));
            }
        }

        private static void WriteColorsOnly(float[] data, Color[] colors, int start, int count)
        {
            for (int i = 0; i < count; i++)
            {
                colors[start + i] = new Color(
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
