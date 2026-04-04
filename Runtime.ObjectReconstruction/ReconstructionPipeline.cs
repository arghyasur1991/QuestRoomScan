#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

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
        private readonly int _frameBudgetMs;
        private readonly int _gridResolution;
        private readonly float _densityThreshold;
        private readonly ComputeShader _triplaneShader;
        private readonly ComputeShader _surfaceNetsShader;
        private readonly ComputeShader _marchingCubesShader;
        private readonly ComputeShader _postprocessShader;
        private readonly MeshAlgorithm _meshAlgorithm;

        private RembgModel _rembg;
        private ReconstructionModel _reconstruction;
        private DecoderModel _decoder;
        private bool _modelsLoaded;

        private TriplaneGridSampler _sampler;
        private IMeshExtractor _extractor;

        private int _postKernelDensity;
        private int _postKernelColors;

        internal ReconstructionPipeline(
            int frameBudgetMs,
            int gridResolution,
            float densityThreshold,
            ComputeShader triplaneShader,
            ComputeShader surfaceNetsShader,
            ComputeShader marchingCubesShader,
            ComputeShader postprocessShader,
            MeshAlgorithm meshAlgorithm = MeshAlgorithm.MarchingCubes)
        {
            _frameBudgetMs = frameBudgetMs;
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
                bool hasAlpha = readable.format == TextureFormat.RGBA32 ||
                                readable.format == TextureFormat.ARGB32 ||
                                readable.format == TextureFormat.RGBA64 ||
                                readable.format == TextureFormat.RGBAFloat ||
                                readable.format == TextureFormat.RGBAHalf ||
                                readable.format == TextureFormat.BGRA32;

                if (hasAlpha && HasMeaningfulAlpha(readable))
                {
                    Logger.Info("[Pipeline] RGBA image detected — using built-in alpha (skipping rembg)");
                    return await ImagePreprocessor.CompositeFromRGBAAsync(readable, 0.85f);
                }

                Logger.Info("[Pipeline] Running rembg for background removal");
                var mask = await _rembg.InferAsync(readable, ct);
                await AsyncHelper.YieldFrame();

                if (ImagePreprocessor.DebugOutputDir != null)
                    ImagePreprocessor.SaveMaskDebugImage(mask,
                        System.IO.Path.Combine(ImagePreprocessor.DebugOutputDir, "unity_rembg_mask.png"));

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
        /// Run the TripoSR forward pass. Scene codes stay on GPU in the worker's output.
        /// Use ExtractMeshAsync() afterwards to get the mesh.
        /// </summary>
        internal async Task RunForwardAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            await _reconstruction.RunAsync(preprocessed, ct);
        }

        /// <summary>
        /// Dumps the scene codes tensor (after RunForwardAsync) to a raw binary file for comparison.
        /// </summary>
        internal void DumpSceneCodes(string path)
        {
            var sceneCodes = _reconstruction.PeekOutput();
            var data = sceneCodes.ReadbackAndClone();
            var floats = data.DownloadToArray();
            data.Dispose();

            var bytes = new byte[floats.Length * sizeof(float)];
            System.Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            System.IO.File.WriteAllBytes(path, bytes);

            var shape = sceneCodes.shape;
            var metaPath = path + ".meta.txt";
            System.IO.File.WriteAllText(metaPath,
                $"dtype=float32\nshape={shape[0]},{shape[1]},{shape[2]},{shape[3]},{shape[4]}\n");

            float min = float.MaxValue, max = float.MinValue, sum = 0;
            for (int i = 0; i < floats.Length; i++)
            {
                if (floats[i] < min) min = floats[i];
                if (floats[i] > max) max = floats[i];
                sum += floats[i];
            }
            Logger.Info($"[Pipeline] Scene codes dumped: {path}");
            Logger.Info($"[Pipeline] Shape: ({shape[0]},{shape[1]},{shape[2]},{shape[3]},{shape[4]}), " +
                        $"range: [{min:F4}, {max:F4}], mean: {sum / floats.Length:F4}");
        }

        internal string DebugDumpDir { get; set; }

        /// <summary>
        /// GPU-resident mesh extraction pipeline. Scene codes come from the reconstruction
        /// worker's last output (no readback). All triplane sampling, decoder inference, and
        /// density/color extraction happen on GPU with zero CPU round-trips per chunk.
        /// </summary>
        internal async Task<Mesh> ExtractMeshAsync(CancellationToken ct)
        {
            int res = _gridResolution;
            var sceneCodes = _reconstruction.PeekOutput();

            EnsureSampler();
            _sampler.CacheSceneCodesGPU(sceneCodes);
            await AsyncHelper.YieldFrame();

            int totalPoints = _sampler.TotalGridPoints;
            int featureDim = _sampler.FeatureDim;
            int chunkSize = 524288;
            var budget = new AsyncHelper.FrameBudget();
            var sw = System.Diagnostics.Stopwatch.StartNew();

            Logger.Info($"[Pipeline] GPU-resident extraction, res={res}, " +
                        $"{totalPoints} points, chunks of {chunkSize}");

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
                    await _decoder.RunAsync(chunkTensor, ct);

                    var decoderOutBuf = _decoder.PeekOutputBuffer();
                    DispatchDensityExtraction(decoderOutBuf, densityBuf, start, count);
                    await budget.YieldIfNeeded();
                }
                Logger.Info($"[Pipeline] Pass 1 density (GPU-resident): {sw.ElapsedMilliseconds}ms ({numChunks} chunks)");

                // Dump density field for debugging
                if (!string.IsNullOrEmpty(DebugDumpDir))
                {
                    try
                    {
                        var densityData = await AsyncHelper.ReadbackAsync<float>(densityBuf, totalPoints);
                        float dMin = float.MaxValue, dMax = float.MinValue, dSum = 0;
                        int aboveThresh = 0;
                        for (int i = 0; i < densityData.Length; i++)
                        {
                            if (densityData[i] < dMin) dMin = densityData[i];
                            if (densityData[i] > dMax) dMax = densityData[i];
                            dSum += densityData[i];
                            if (densityData[i] > _densityThreshold) aboveThresh++;
                        }
                        Logger.Info($"[Pipeline] Density field: range=[{dMin:F4}, {dMax:F4}], " +
                                    $"mean={dSum / densityData.Length:F4}, " +
                                    $"above {_densityThreshold}: {aboveThresh}/{totalPoints} ({100f * aboveThresh / totalPoints:F1}%)");

                        string dPath = System.IO.Path.Combine(DebugDumpDir, "sentis_density.bin");
                        var dBytes = new byte[densityData.Length * sizeof(float)];
                        System.Buffer.BlockCopy(densityData, 0, dBytes, 0, dBytes.Length);
                        System.IO.File.WriteAllBytes(dPath, dBytes);
                        System.IO.File.WriteAllText(dPath + ".meta.txt",
                            $"dtype=float32\nshape={totalPoints}\nresolution={res}\nnote=x varies fastest: ix + iy*res + iz*res*res\n");
                        Logger.Info($"[Pipeline] Density dumped: {dPath}");
                    }
                    catch (System.Exception e)
                    {
                        Logger.Info($"[Pipeline] Density dump failed: {e.Message}");
                    }
                }

                // --- Mesh extraction: density buffer already on GPU ---
                sw.Restart();
                EnsureExtractor();
                var mesh = await _extractor.ExtractAsync(densityBuf);
                Logger.Info($"[Pipeline] {_meshAlgorithm}: {sw.ElapsedMilliseconds}ms");
                await AsyncHelper.YieldFrame();
                ct.ThrowIfCancellationRequested();

                // --- Pass 2: vertex color extraction (GPU-resident) ---
                sw.Restart();
                var meshVerts = mesh.vertices;
                int numVerts = meshVerts.Length;
                Logger.Info($"[Pipeline] Pass 2: {numVerts} vertex color queries");

                if (numVerts == 0)
                {
                    Logger.Info("[Pipeline] No vertices — skipping color pass");
                    return mesh;
                }

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
                        await _decoder.RunAsync(chunkTensor, ct);

                        var decoderOutBuf = _decoder.PeekOutputBuffer();
                        DispatchColorExtraction(decoderOutBuf, colorBuf, start, count);
                        await budget.YieldIfNeeded();
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

                    Logger.Info($"[Pipeline] Pass 2 colors (GPU-resident): {sw.ElapsedMilliseconds}ms ({(numVerts + chunkSize - 1) / chunkSize} chunks)");
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
            _rembg?.Dispose();
            _reconstruction?.Dispose();
            _decoder?.Dispose();
            _sampler?.Dispose();
            _extractor?.Dispose();
            _sampler = null;
            _extractor = null;
            _modelsLoaded = false;
        }
    }
}
#endif
