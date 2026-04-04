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
    /// All inference is async with multi-frame splitting. Post-forward mesh extraction uses a fully
    /// GPU-resident data flow: triplane sampling, decoder inference, density/color extraction all stay
    /// on GPU via ComputeTensorData.Pin, with a single async readback at the end.
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

        private TriplaneGridSampler _sampler;
        private IMeshExtractor _extractor;

        private int _postKernelDensity;
        private int _postKernelColors;

        internal ReconstructionPipeline(
            int gridResolution,
            float densityThreshold,
            ComputeShader triplaneShader,
            ComputeShader surfaceNetsShader,
            ComputeShader marchingCubesShader,
            ComputeShader postprocessShader,
            MeshAlgorithm meshAlgorithm = MeshAlgorithm.MarchingCubes)
        {
            _gridResolution = gridResolution;
            _densityThreshold = densityThreshold;
            _triplaneShader = triplaneShader;
            _surfaceNetsShader = surfaceNetsShader;
            _marchingCubesShader = marchingCubesShader;
            _postprocessShader = postprocessShader;
            _meshAlgorithm = meshAlgorithm;

            _postKernelDensity = _postprocessShader.FindKernel("ExtractDensity");
            _postKernelColors = _postprocessShader.FindKernel("ExtractColors");
        }

        /// <summary>No-op retained for API compatibility. Models are now loaded on-demand per stage.</summary>
        internal Task LoadModelsAsync(CancellationToken ct) => Task.CompletedTask;

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

                using var rembg = new RembgModel();
                await rembg.LoadAsync(ct);

                var mask = await rembg.InferAsync(readable, ct);
                rembg.Dispose();
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

        /// <summary>
        /// Run the split TripoSR model (part 1 then part 2). Each half is loaded, executed,
        /// and disposed independently. Scene codes are copied to the sampler's own buffer.
        /// </summary>
        internal async Task RunForwardAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            using var reconstruction = new ReconstructionModel();
            var sceneCodes = await reconstruction.RunAsync(preprocessed, ct);

            EnsureSampler();
            _sampler.CacheSceneCodesGPU(sceneCodes);
            sceneCodes.Dispose();
            await AsyncHelper.YieldFrame();
        }

        /// <summary>
        /// GPU-resident mesh extraction pipeline. Scene codes are already cached in the sampler.
        /// Decoder is loaded on-demand and disposed when extraction is complete.
        /// </summary>
        internal async Task<Mesh> ExtractMeshAsync(CancellationToken ct)
        {
            int totalPoints = _sampler.TotalGridPoints;
            int featureDim = _sampler.FeatureDim;
            const int maxBufferBytes = 128 * 1024 * 1024; // Quest 3 per-buffer limit
            int chunkSize = maxBufferBytes / (featureDim * sizeof(float));

            using var decoder = new DecoderModel();
            await decoder.LoadAsync(ct);

            // --- Pass 1: GPU triplane → decoder → density extraction (zero CPU transfers) ---
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
                // --- Mesh extraction: density buffer already on GPU ---
                EnsureExtractor();
                var mesh = await _extractor.ExtractAsync(densityBuf);
                await AsyncHelper.YieldFrame();
                ct.ThrowIfCancellationRequested();

                // --- Pass 2: vertex color extraction (GPU-resident) ---
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
            _sampler?.Dispose();
            _extractor?.Dispose();
            _sampler = null;
            _extractor = null;
        }
    }
}
#endif
