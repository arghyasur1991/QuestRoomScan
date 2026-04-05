#if HAS_ONNXRUNTIME
using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Orchestrates the full reconstruction pipeline: rembg → preprocess → forward → mesh extraction.
    /// All neural network inference runs on background threads via Task.Run (ORT).
    /// Main thread only handles pixel ops, GPU compute dispatch, and async readback.
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
        private readonly ExecutionProvider _executionProvider;
        private readonly bool _mobileOptimized;
        private readonly int _densitySmoothPasses;

        private TriplaneGridSampler _sampler;
        private IMeshExtractor _extractor;

        private OrtRembgModel _rembg;
        private OrtReconstructionModel _reconstruction;
        private OrtDecoderModel _decoder;

        private int _postKernelDensity;
        private int _postKernelSmooth;
        private int _postKernelColors;

        // Reusable buffers for decoder hot path
        private ComputeBuffer _featureBuffer;
        private float[] _readbackBuffer;
        private ComputeBuffer _uploadBuffer;

        internal ReconstructionPipeline(
            int gridResolution,
            float densityThreshold,
            ComputeShader triplaneShader,
            ComputeShader surfaceNetsShader,
            ComputeShader marchingCubesShader,
            ComputeShader postprocessShader,
            MeshAlgorithm meshAlgorithm = MeshAlgorithm.MarchingCubes,
            bool preloadModels = false,
            ExecutionProvider executionProvider = ExecutionProvider.CPU,
            bool mobileOptimized = false,
            int densitySmoothPasses = 1)
        {
            _gridResolution = gridResolution;
            _densityThreshold = densityThreshold;
            _triplaneShader = triplaneShader;
            _surfaceNetsShader = surfaceNetsShader;
            _marchingCubesShader = marchingCubesShader;
            _postprocessShader = postprocessShader;
            _meshAlgorithm = meshAlgorithm;
            _preloadModels = preloadModels;
            _executionProvider = executionProvider;
            _mobileOptimized = mobileOptimized;
            _densitySmoothPasses = densitySmoothPasses;

            _postKernelDensity = _postprocessShader.FindKernel("ExtractDensity");
            _postKernelSmooth = _postprocessShader.FindKernel("SmoothDensity");
            _postKernelColors = _postprocessShader.FindKernel("ExtractColors");
        }

        /// <summary>
        /// In preload mode, loads all models and keeps sessions alive.
        /// In on-demand mode, this is a no-op.
        /// </summary>
        internal async Task LoadModelsAsync(CancellationToken ct)
        {
            if (!_preloadModels) return;

            _rembg = new OrtRembgModel();
            await _rembg.LoadAsync(_executionProvider, _mobileOptimized, ct);

            _reconstruction = new OrtReconstructionModel(_executionProvider, _mobileOptimized);
            await _reconstruction.PreloadAsync(ct);

            _decoder = new OrtDecoderModel();
            await _decoder.LoadAsync(_executionProvider, _mobileOptimized, ct);
        }

        internal async Task<float[]> PreprocessAsync(Texture2D image, CancellationToken ct)
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
                    return await ImagePreprocessor.ApplyMaskAndCompositeAsync(
                        readable, mask, 320, 320, 0.85f);
                }

                using var rembg = new OrtRembgModel();
                await rembg.LoadAsync(_executionProvider, _mobileOptimized, ct);
                var maskLocal = await rembg.InferAsync(readable, ct);
                await AsyncHelper.YieldFrame();
                return await ImagePreprocessor.ApplyMaskAndCompositeAsync(
                    readable, maskLocal, 320, 320, 0.85f);
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

        internal async Task RunForwardAsync(float[] preprocessed, CancellationToken ct)
        {
            float[] sceneCodes;

            if (_reconstruction != null)
            {
                sceneCodes = await _reconstruction.RunAsync(preprocessed, ct);
            }
            else
            {
                using var reconstruction = new OrtReconstructionModel(_executionProvider, _mobileOptimized);
                sceneCodes = await reconstruction.RunAsync(preprocessed, ct);
                await AsyncHelper.YieldFrame();
            }

            EnsureSampler();
            // TripoSR scene codes shape: [1, numPlanes, channels, planeH, planeW]
            // For the split model: typically [1, 3, 40, 64, 64]
            _sampler.CacheSceneCodesGPU(sceneCodes, 3, 40, 64, 64);
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
                decoder = new OrtDecoderModel();
                await decoder.LoadAsync(_executionProvider, _mobileOptimized, ct);
            }

            EnsureFeatureBuffer(chunkSize, featureDim);
            var densityBuf = new ComputeBuffer(totalPoints, sizeof(float));
            int numChunks = (totalPoints + chunkSize - 1) / chunkSize;

            try
            {
                for (int c = 0; c < numChunks; c++)
                {
                    ct.ThrowIfCancellationRequested();
                    int start = c * chunkSize;
                    int count = Mathf.Min(chunkSize, totalPoints - start);

                    _sampler.SampleGridChunkGPU(start, count, _featureBuffer);

                    await AsyncHelper.ReadbackAsync(_featureBuffer, ref _readbackBuffer, count * featureDim);

                    float[] decoderOut = await decoder.RunChunkAsync(_readbackBuffer, count);

                    EnsureUploadBuffer(count * 4);
                    _uploadBuffer.SetData(decoderOut, 0, 0, count * 4);

                    DispatchDensityExtraction(_uploadBuffer, densityBuf, start, count);
                    await AsyncHelper.YieldFrame();
                }

                if (_densitySmoothPasses > 0)
                {
                    SmoothDensityVolume(densityBuf, totalPoints);
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

                        EnsureFeatureBuffer(count, featureDim);
                        _sampler.SampleAtPositionsGPU(allPosBuf, start, count, _featureBuffer);

                        await AsyncHelper.ReadbackAsync(_featureBuffer, ref _readbackBuffer, count * featureDim);

                        float[] decoderOut = await decoder.RunChunkAsync(_readbackBuffer, count);

                        EnsureUploadBuffer(count * 4);
                        _uploadBuffer.SetData(decoderOut, 0, 0, count * 4);

                        DispatchColorExtraction(_uploadBuffer, colorBuf, start, count);
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

        private void EnsureFeatureBuffer(int count, int featureDim)
        {
            int needed = count * featureDim;
            if (_featureBuffer != null && _featureBuffer.count >= needed) return;
            _featureBuffer?.Release();
            _featureBuffer = new ComputeBuffer(needed, sizeof(float));
        }

        private void EnsureUploadBuffer(int minCount)
        {
            if (_uploadBuffer != null && _uploadBuffer.count >= minCount) return;
            _uploadBuffer?.Release();
            _uploadBuffer = new ComputeBuffer(minCount, sizeof(float));
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

        /// <summary>
        /// 3x3x3 Gaussian-weighted smoothing on the density volume to filter
        /// INT8 quantization noise amplified by exp(). Ping-pongs between two buffers.
        /// </summary>
        private void SmoothDensityVolume(ComputeBuffer densityBuf, int totalPoints)
        {
            var tempBuf = new ComputeBuffer(totalPoints, sizeof(float));
            int groups = (totalPoints + 255) / 256;
            _postprocessShader.SetInt("_SmoothRes", _gridResolution);

            try
            {
                var src = densityBuf;
                var dst = tempBuf;

                for (int pass = 0; pass < _densitySmoothPasses; pass++)
                {
                    _postprocessShader.SetBuffer(_postKernelSmooth, "_DensityInput", src);
                    _postprocessShader.SetBuffer(_postKernelSmooth, "_DensityVolume", dst);
                    _postprocessShader.Dispatch(_postKernelSmooth, groups, 1, 1);

                    (src, dst) = (dst, src);
                }

                // After all passes, result is in 'src'. If it's tempBuf, copy back.
                if (src != densityBuf)
                    Graphics.CopyBuffer(src, densityBuf);
            }
            finally
            {
                tempBuf.Release();
            }
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
            _featureBuffer?.Release();
            _uploadBuffer?.Release();
            _rembg = null;
            _reconstruction = null;
            _decoder = null;
            _sampler = null;
            _extractor = null;
            _featureBuffer = null;
            _uploadBuffer = null;
        }
    }
}
#endif
