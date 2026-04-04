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
    /// Supports two modes:
    /// <list type="bullet">
    /// <item><b>Preload</b> (editor): All models loaded once via <see cref="LoadModelsAsync"/>,
    ///   kept alive across runs. Zero per-run load/dispose overhead.</item>
    /// <item><b>On-demand</b> (Quest): Each model loaded, executed, and disposed per stage
    ///   to minimize peak GPU memory.</item>
    /// </list>
    /// </summary>
    internal sealed class ReconstructionPipeline : IDisposable
    {
        private readonly int _gridResolution;
        private readonly float _densityThreshold;
        private readonly ComputeShader _triplaneShader;
        private readonly ComputeShader _surfaceNetsShader;
        private readonly ComputeShader _marchingCubesShader;
        private readonly ComputeShader _postprocessShader;
        private readonly MeshAlgorithm _meshAlgorithm;
        private readonly bool _preloadModels;

        private TriplaneGridSampler _sampler;
        private IMeshExtractor _extractor;

        private RembgModel _rembg;
        private ReconstructionModel _reconstruction;
        private DecoderModel _decoder;

        private int _postKernelDensity;
        private int _postKernelColors;

        internal ReconstructionPipeline(
            int gridResolution,
            float densityThreshold,
            ComputeShader triplaneShader,
            ComputeShader surfaceNetsShader,
            ComputeShader marchingCubesShader,
            ComputeShader postprocessShader,
            MeshAlgorithm meshAlgorithm = MeshAlgorithm.MarchingCubes,
            bool preloadModels = false)
        {
            _gridResolution = gridResolution;
            _densityThreshold = densityThreshold;
            _triplaneShader = triplaneShader;
            _surfaceNetsShader = surfaceNetsShader;
            _marchingCubesShader = marchingCubesShader;
            _postprocessShader = postprocessShader;
            _meshAlgorithm = meshAlgorithm;
            _preloadModels = preloadModels;

            _postKernelDensity = _postprocessShader.FindKernel("ExtractDensity");
            _postKernelColors = _postprocessShader.FindKernel("ExtractColors");
        }

        /// <summary>
        /// In preload mode, loads all models and keeps workers alive.
        /// In on-demand mode, this is a no-op.
        /// </summary>
        internal async Task LoadModelsAsync(CancellationToken ct)
        {
            if (!_preloadModels) return;

            _rembg = new RembgModel();
            await _rembg.LoadAsync(ct);

            _reconstruction = new ReconstructionModel();
            await _reconstruction.PreloadAsync(ct);

            _decoder = new DecoderModel();
            await _decoder.LoadAsync(ct);
        }

        internal async Task<Tensor<float>> PreprocessAsync(Texture2D image, CancellationToken ct)
        {
            var readable = MakeReadable(image);
            try
            {
                bool hasAlpha = readable.format == TextureFormat.RGBA32 ||
                                readable.format == TextureFormat.ARGB32 ||
                                readable.format == TextureFormat.RGBA64 ||
                                readable.format == TextureFormat.RGBAFloat ||
                                readable.format == TextureFormat.RGBAHalf ||
                                readable.format == TextureFormat.BGRA32;

                if (hasAlpha && HasMeaningfulAlpha(readable))
                    return await ImagePreprocessor.CompositeFromRGBAAsync(readable, 0.85f);

                if (_rembg != null)
                {
                    var mask = await _rembg.InferAsync(readable, ct);
                    var result = await ImagePreprocessor.ApplyMaskAndCompositeAsync(readable, mask, 0.85f);
                    mask.Dispose();
                    return result;
                }

                using var rembg = new RembgModel();
                await rembg.LoadAsync(ct);
                var maskLocal = await rembg.InferAsync(readable, ct);
                rembg.Dispose();
                await AsyncHelper.YieldFrame();
                var resultLocal = await ImagePreprocessor.ApplyMaskAndCompositeAsync(readable, maskLocal, 0.85f);
                maskLocal.Dispose();
                return resultLocal;
            }
            finally
            {
                if (readable != image)
                    SafeDestroy(readable);
            }
        }

        private static bool HasMeaningfulAlpha(Texture2D tex)
        {
            var pixels = tex.GetRawTextureData<Color32>();
            int step = Mathf.Max(1, pixels.Length / 200);
            for (int i = 0; i < pixels.Length; i += step)
                if (pixels[i].a < 250) return true;
            return false;
        }

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

        internal async Task RunForwardAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            if (_reconstruction != null)
            {
                var sceneCodes = await _reconstruction.RunAsync(preprocessed, ct);
                EnsureSampler();
                _sampler.CacheSceneCodesGPU(sceneCodes);
                sceneCodes.Dispose();
                return;
            }

            using var reconstruction = new ReconstructionModel();
            var sceneCodesLocal = await reconstruction.RunAsync(preprocessed, ct);
            EnsureSampler();
            _sampler.CacheSceneCodesGPU(sceneCodesLocal);
            sceneCodesLocal.Dispose();
            await AsyncHelper.YieldFrame();
        }

        internal async Task<Mesh> ExtractMeshAsync(CancellationToken ct)
        {
            int totalPoints = _sampler.TotalGridPoints;
            int featureDim = _sampler.FeatureDim;
            const int maxBufferBytes = 128 * 1024 * 1024;
            int chunkSize = maxBufferBytes / (featureDim * sizeof(float));

            var decoder = _decoder;
            bool ownsDecoder = decoder == null;
            if (ownsDecoder)
            {
                decoder = new DecoderModel();
                await decoder.LoadAsync(ct);
            }

            var densityBuf = new ComputeBuffer(totalPoints, sizeof(float));
            int numChunks = (totalPoints + chunkSize - 1) / chunkSize;

            try
            {
                for (int c = 0; c < numChunks; c++)
                {
                    ct.ThrowIfCancellationRequested();
                    int start = c * chunkSize;
                    int count = Mathf.Min(chunkSize, totalPoints - start);

                    using var chunkTensor = new Tensor<float>(new TensorShape(count, featureDim));
                    var pinned = ComputeTensorData.Pin(chunkTensor);

                    _sampler.SampleGridChunkGPU(start, count, pinned.buffer);
                    await decoder.RunAsync(chunkTensor, ct);

                    var decoderOutBuf = decoder.PeekOutputBuffer();
                    DispatchDensityExtraction(decoderOutBuf, densityBuf, start, count);
                    await AsyncHelper.YieldFrame();
                }

                EnsureExtractor();
                var mesh = await _extractor.ExtractAsync(densityBuf);
                await AsyncHelper.YieldFrame();
                ct.ThrowIfCancellationRequested();

                var meshVerts = mesh.vertices;
                int numVerts = meshVerts.Length;

                if (numVerts == 0)
                    return mesh;

                var posData = new float[numVerts * 3];
                for (int i = 0; i < numVerts; i++)
                {
                    posData[i * 3 + 0] = meshVerts[i].x;
                    posData[i * 3 + 1] = meshVerts[i].y;
                    posData[i * 3 + 2] = meshVerts[i].z;
                }
                var allPosBuf = new ComputeBuffer(numVerts * 3, sizeof(float));
                allPosBuf.SetData(posData);

                var colorBuf = new ComputeBuffer(numVerts * 3, sizeof(float));

                try
                {
                    int vertChunks = (numVerts + chunkSize - 1) / chunkSize;
                    for (int c = 0; c < vertChunks; c++)
                    {
                        ct.ThrowIfCancellationRequested();
                        int start = c * chunkSize;
                        int count = Mathf.Min(chunkSize, numVerts - start);

                        using var chunkTensor = new Tensor<float>(new TensorShape(count, featureDim));
                        var pinned = ComputeTensorData.Pin(chunkTensor);

                        _sampler.SampleAtPositionsGPU(allPosBuf, start, count, pinned.buffer);
                        await decoder.RunAsync(chunkTensor, ct);

                        var decoderOutBuf = decoder.PeekOutputBuffer();
                        DispatchColorExtraction(decoderOutBuf, colorBuf, start, count);
                        await AsyncHelper.YieldFrame();
                    }

                    var colorData = await AsyncHelper.ReadbackAsync<float>(colorBuf, numVerts * 3);
                    var vertColors = new Color[numVerts];
                    for (int i = 0; i < numVerts; i++)
                    {
                        vertColors[i] = new Color(
                            colorData[i * 3 + 0],
                            colorData[i * 3 + 1],
                            colorData[i * 3 + 2]);
                    }
                    mesh.SetColors(vertColors);
                }
                finally
                {
                    allPosBuf.Release();
                    colorBuf.Release();
                }

                return mesh;
            }
            finally
            {
                densityBuf.Release();
                if (ownsDecoder) decoder.Dispose();
            }
        }

        private void DispatchDensityExtraction(
            ComputeBuffer decoderOutput, ComputeBuffer densityVolume, int offset, int count)
        {
            _postprocessShader.SetInt("_Offset", offset);
            _postprocessShader.SetInt("_Count", count);
            _postprocessShader.SetBuffer(_postKernelDensity, "_DecoderOutput", decoderOutput);
            _postprocessShader.SetBuffer(_postKernelDensity, "_DensityVolume", densityVolume);
            _postprocessShader.Dispatch(_postKernelDensity, (count + 255) / 256, 1, 1);
        }

        private void DispatchColorExtraction(
            ComputeBuffer decoderOutput, ComputeBuffer colorOutput, int offset, int count)
        {
            _postprocessShader.SetInt("_Offset", offset);
            _postprocessShader.SetInt("_Count", count);
            _postprocessShader.SetBuffer(_postKernelColors, "_DecoderOutput", decoderOutput);
            _postprocessShader.SetBuffer(_postKernelColors, "_ColorOutput", colorOutput);
            _postprocessShader.Dispatch(_postKernelColors, (count + 255) / 256, 1, 1);
        }

        private void EnsureSampler()
        {
            _sampler ??= new TriplaneGridSampler(_triplaneShader, _gridResolution);
        }

        private void EnsureExtractor()
        {
            if (_extractor != null) return;
            _extractor = _meshAlgorithm switch
            {
                MeshAlgorithm.MarchingCubes =>
                    new MarchingCubes(_marchingCubesShader, _gridResolution, _densityThreshold),
                MeshAlgorithm.SurfaceNets =>
                    new DensitySurfaceNets(_surfaceNetsShader, _gridResolution, _densityThreshold),
                _ => new MarchingCubes(_marchingCubesShader, _gridResolution, _densityThreshold)
            };
        }

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
            _sampler?.Dispose();
            _extractor?.Dispose();
            _rembg = null;
            _reconstruction = null;
            _decoder = null;
            _sampler = null;
            _extractor = null;
        }
    }
}
#endif
