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
        private readonly MeshExtractionBackend _meshBackend;

        private TriplaneGridSampler _sampler;
        private IMeshExtractor _extractor;

        private OrtRembgModel _rembg;
        private OrtReconstructionModel _reconstruction;
        private OrtDecoderModel _decoder;

        private int _postKernelExtractRaw;
        private int _postKernelSmoothRaw;
        private int _postKernelCopyBuf;
        private int _postKernelActivate;
        private int _postKernelDensity;
        private int _postKernelColors;

        // Reusable buffers for decoder hot path (GPU path)
        private ComputeBuffer _featureBuffer;
        private float[] _readbackBuffer;
        private ComputeBuffer _uploadBuffer;

        // Cached scene codes for CPU mesh extraction path
        private float[] _sceneCodes;
        private int _sceneNumPlanes, _sceneChannels, _scenePlaneH, _scenePlaneW;

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
            int densitySmoothPasses = 1,
            MeshExtractionBackend meshBackend = MeshExtractionBackend.GPU)
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
            _meshBackend = meshBackend;

            _postKernelExtractRaw = _postprocessShader.FindKernel("ExtractRawDensity");
            _postKernelSmoothRaw = _postprocessShader.FindKernel("SmoothRaw");
            _postKernelCopyBuf = _postprocessShader.FindKernel("CopyBuffer");
            _postKernelActivate = _postprocessShader.FindKernel("ActivateDensity");
            _postKernelDensity = _postprocessShader.FindKernel("ExtractDensity");
            _postKernelColors = _postprocessShader.FindKernel("ExtractColors");
        }

        /// <summary>
        /// In preload mode, loads all models and keeps sessions alive.
        /// In on-demand mode, this is a no-op.
        /// </summary>
        internal async Task LoadModelsAsync(CancellationToken ct)
        {
            if (!_preloadModels) return;
            if (_rembg != null && _reconstruction != null && _decoder != null) return;

            _rembg ??= new OrtRembgModel();
            if (!_rembg.IsLoaded)
                await _rembg.LoadAsync(_executionProvider, _mobileOptimized, ct);

            _reconstruction ??= new OrtReconstructionModel(_executionProvider, _mobileOptimized);
            if (!_reconstruction.IsLoaded)
                await _reconstruction.PreloadAsync(ct);

            _decoder ??= new OrtDecoderModel();
            if (!_decoder.IsLoaded)
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

            _sceneNumPlanes = 3;
            _sceneChannels = 40;
            _scenePlaneH = 64;
            _scenePlaneW = 64;

            if (_meshBackend == MeshExtractionBackend.CPU)
            {
                _sceneCodes = sceneCodes;
            }
            else
            {
                EnsureSampler();
                _sampler.CacheSceneCodesGPU(sceneCodes, _sceneNumPlanes, _sceneChannels, _scenePlaneH, _scenePlaneW);
            }
        }

        internal async Task<Mesh> ExtractMeshAsync(CancellationToken ct)
        {
            if (_meshBackend == MeshExtractionBackend.CPU)
                return await ExtractMeshCpuAsync(ct);

            return await ExtractMeshGpuAsync(ct);
        }

        private async Task<Mesh> ExtractMeshCpuAsync(CancellationToken ct)
        {
            var decoder = _decoder;
            bool ownsDecoder = decoder == null;
            if (ownsDecoder)
            {
                decoder = new OrtDecoderModel();
                await decoder.LoadAsync(_executionProvider, _mobileOptimized, ct);
            }

            try
            {
                var cpuExtractor = new CpuMeshExtractor(_gridResolution, _densityThreshold);
                return await cpuExtractor.ExtractAsync(
                    _sceneCodes, _sceneNumPlanes, _sceneChannels, _scenePlaneH, _scenePlaneW,
                    decoder, ct);
            }
            finally
            {
                if (ownsDecoder) decoder.Dispose();
            }
        }

        private async Task<Mesh> ExtractMeshGpuAsync(CancellationToken ct)
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
            ComputeBuffer rawDensityBuf = _densitySmoothPasses > 0
                ? new ComputeBuffer(totalPoints, sizeof(float))
                : null;
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

                    if (_densitySmoothPasses > 0)
                        DispatchExtractRaw(_uploadBuffer, rawDensityBuf, start, count);
                    else
                        DispatchDensityExtraction(_uploadBuffer, densityBuf, start, count);
                    await AsyncHelper.YieldFrame();
                }

                if (_densitySmoothPasses > 0)
                {
                    SmoothAndActivate(rawDensityBuf, densityBuf, totalPoints);
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
                rawDensityBuf?.Release();
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

        private void DispatchExtractRaw(
            ComputeBuffer decoderOutput, ComputeBuffer rawBuf, int offset, int count)
        {
            _postprocessShader.SetInt("_Offset", offset);
            _postprocessShader.SetInt("_Count", count);
            _postprocessShader.SetBuffer(_postKernelExtractRaw, "_DecoderOutput", decoderOutput);
            _postprocessShader.SetBuffer(_postKernelExtractRaw, "_RawOutput", rawBuf);
            _postprocessShader.Dispatch(_postKernelExtractRaw, (count + 255) / 256, 1, 1);
        }

        /// <summary>
        /// Smooth raw (pre-exp) density values, then apply exp(raw-1) activation.
        /// Smoothing in log-space filters additive INT8 quantization noise uniformly
        /// without smudging edges or colors in the final density field.
        /// </summary>
        private void SmoothAndActivate(ComputeBuffer rawBuf, ComputeBuffer densityBuf, int totalPoints)
        {
            int groups = (totalPoints + 255) / 256;
            _postprocessShader.SetInt("_SmoothRes", _gridResolution);

            var tempBuf = new ComputeBuffer(totalPoints, sizeof(float));
            try
            {
                var src = rawBuf;
                var dst = tempBuf;

                for (int pass = 0; pass < _densitySmoothPasses; pass++)
                {
                    _postprocessShader.SetBuffer(_postKernelSmoothRaw, "_RawInput", src);
                    _postprocessShader.SetBuffer(_postKernelSmoothRaw, "_RawOutput", dst);
                    _postprocessShader.Dispatch(_postKernelSmoothRaw, groups, 1, 1);

                    (src, dst) = (dst, src);
                }

                // Result is in 'src'. If it's tempBuf (odd passes), copy to rawBuf first.
                if (src != rawBuf)
                {
                    _postprocessShader.SetBuffer(_postKernelCopyBuf, "_RawInput", src);
                    _postprocessShader.SetBuffer(_postKernelCopyBuf, "_RawOutput", rawBuf);
                    _postprocessShader.Dispatch(_postKernelCopyBuf, groups, 1, 1);
                    src = rawBuf;
                }

                // Apply exp(smoothed_raw - 1) → final density
                _postprocessShader.SetBuffer(_postKernelActivate, "_RawInput", src);
                _postprocessShader.SetBuffer(_postKernelActivate, "_DensityVolume", densityBuf);
                _postprocessShader.Dispatch(_postKernelActivate, groups, 1, 1);
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
