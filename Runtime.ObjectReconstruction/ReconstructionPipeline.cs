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
        private readonly int _forwardLayersPerFrame;
        private readonly int _decoderLayersPerFrame;
        private readonly int _gridSampleChunksPerFrame;
        private readonly int _gridResolution;
        private readonly float _densityThreshold;
        private readonly ComputeShader _triplaneShader;
        private readonly ComputeShader _surfaceNetsShader;

        private RembgModel _rembg;
        private ReconstructionModel _reconstruction;
        private DecoderModel _decoder;
        private bool _modelsLoaded;

        internal ReconstructionPipeline(
            int forwardLayersPerFrame,
            int decoderLayersPerFrame,
            int gridSampleChunksPerFrame,
            int gridResolution,
            float densityThreshold,
            ComputeShader triplaneShader,
            ComputeShader surfaceNetsShader)
        {
            _forwardLayersPerFrame = forwardLayersPerFrame;
            _decoderLayersPerFrame = decoderLayersPerFrame;
            _gridSampleChunksPerFrame = gridSampleChunksPerFrame;
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
            var mask = await _rembg.InferAsync(image, ct);
            var result = ImagePreprocessor.ApplyMaskAndComposite(image, mask, 0.85f);
            mask.Dispose();
            return result;
        }

        internal async Task<Tensor<float>> RunForwardAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            return await _reconstruction.InferAsync(preprocessed, _forwardLayersPerFrame, ct);
        }

        internal async Task<Mesh> ExtractMeshAsync(Tensor<float> sceneCodes, CancellationToken ct)
        {
            int res = _gridResolution;
            int totalPoints = res * res * res;
            int featureDim = 120; // 3 planes * 40 channels

            var sampler = new TriplaneGridSampler(_triplaneShader, res);
            var features = sampler.SampleFeatures(sceneCodes);
            await Task.Yield();
            ct.ThrowIfCancellationRequested();

            int chunkSize = 65536;
            int numChunks = (totalPoints + chunkSize - 1) / chunkSize;
            var density = new float[totalPoints];
            var colors = new Color[totalPoints];

            for (int c = 0; c < numChunks; c++)
            {
                ct.ThrowIfCancellationRequested();
                int start = c * chunkSize;
                int count = Mathf.Min(chunkSize, totalPoints - start);

                var chunkFeatures = ExtractChunk(features, start, count, featureDim);
                var chunkResult = await _decoder.InferAsync(chunkFeatures, _decoderLayersPerFrame, ct);
                chunkFeatures.Dispose();

                WriteDecoderResults(chunkResult, density, colors, start, count);
                chunkResult.Dispose();

                if ((c + 1) % _gridSampleChunksPerFrame == 0)
                    await Task.Yield();
            }

            features.Dispose();
            sampler.Dispose();

            var surfaceNets = new DensitySurfaceNets(_surfaceNetsShader, res, _densityThreshold);
            var mesh = surfaceNets.Extract(density, colors);
            surfaceNets.Dispose();

            return mesh;
        }

        private static Tensor<float> ExtractChunk(Tensor<float> features, int start, int count, int dim)
        {
            var chunk = new Tensor<float>(new TensorShape(count, dim));
            for (int i = 0; i < count; i++)
                for (int d = 0; d < dim; d++)
                    chunk[i * dim + d] = features[(start + i) * dim + d];
            return chunk;
        }

        private static void WriteDecoderResults(
            Tensor<float> result, float[] density, Color[] colors, int start, int count)
        {
            var data = result.DownloadToArray();
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
