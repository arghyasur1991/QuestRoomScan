#if HAS_ONNXRUNTIME
using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Single-image 3D object reconstruction via TripoSR + ONNX Runtime.
    /// Implements <see cref="IRoomScanModule"/> for automatic discovery by <see cref="RoomScanner"/>.
    /// All neural network inference runs on background threads — no main-thread blocking.
    /// </summary>
    [DisallowMultipleComponent]
    public class ObjectReconstructionModule : MonoBehaviour, IRoomScanModule, IObjectReconstructionProvider
    {
        [Header("Shaders")]
        [SerializeField] internal ComputeShader triplaneGridSampleShader;
        [SerializeField] internal ComputeShader densitySurfaceNetsShader;
        [SerializeField] internal ComputeShader densityMarchingCubesShader;
        [SerializeField] internal ComputeShader decoderPostprocessShader;
        [SerializeField] internal Shader vertexColorShader;
        [SerializeField] internal Shader projectedTextureShader;

        [Header("Test Images")]
        [SerializeField] private Texture2D[] testImages = Array.Empty<Texture2D>();

        [Header("Inference")]
        [SerializeField, Tooltip("CPU: works everywhere. NNAPI: Android GPU/NPU. " +
            "XNNPACK: optimized CPU. CoreML: macOS/iOS acceleration.")]
        private ExecutionProvider executionProvider = ExecutionProvider.CPU;

        [SerializeField, Tooltip("Enable mobile-optimized session options (disable memory patterns, " +
            "single thread). Recommended for Quest 3.")]
        private bool mobileOptimized = true;

        [Header("Mesh Extraction")]
        [SerializeField] private int gridResolution = 256;
        [SerializeField] private float densityThreshold = 25f;
        [SerializeField] private MeshAlgorithm meshAlgorithm = MeshAlgorithm.MarchingCubes;

        [SerializeField, Tooltip("3x3x3 Gaussian smoothing passes on raw density (pre-exp) volume. " +
            "Workaround for ORT < 1.24 INT8 quantization noise. " +
            "0 = disabled (default), 1+ = smoothing passes. Not needed with ORT >= 1.24.")]
        private int densitySmoothPasses = 0;

        [SerializeField, Tooltip("Laplacian smoothing iterations on the mesh after extraction. " +
            "Reduces high-frequency noise from marching cubes. 0 = disabled, 1-3 recommended.")]
        private int laplacianSmoothIterations = 0;

        [SerializeField, Range(0.1f, 0.9f), Tooltip("Laplacian smoothing strength per iteration. " +
            "Higher values smooth more aggressively. 0.5 is standard.")]
        private float laplacianSmoothLambda = 0.5f;

        [SerializeField, Tooltip("GPU: uses compute shaders for triplane sampling, postprocessing, " +
            "and marching cubes (fast, but uses GPU and causes FPS drops). " +
            "CPU: pure background-thread pipeline matching Python exactly (no FPS impact, " +
            "useful for quality parity testing and when GPU is constrained).")]
        private MeshExtractionBackend meshExtractionBackend = MeshExtractionBackend.GPU;

        public string ModuleName => "Object Reconstruction";
        public bool IsRunning => _running;
        public string Status => _status;
        public Texture2D[] TestImages => testImages;

        public event Action<string> StatusChanged;

        private ReconstructionPipeline _pipeline;
        private bool _running;
        private string _status = "Idle";
        private CancellationTokenSource _cts;

        public void OnModuleInitialize(RoomScanner scanner)
        {
            Logger.Info("[ObjectReconstruction] Module initialized");
        }

        public async Task LoadModelsAsync(CancellationToken ct = default)
        {
            EnsurePipeline();
            ReportStatus("Loading models...");
            await _pipeline.LoadModelsAsync(ct);
            ReportStatus("Models loaded");
        }

        public async Task<Mesh> ReconstructAsync(Texture2D image, CancellationToken ct = default)
        {
            if (_running)
            {
                Logger.Warning("[ObjectReconstruction] Reconstruction already in progress");
                return null;
            }

            _running = true;
            _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);

            float[] preprocessed = null;
            try
            {
                EnsurePipeline();
                _pipeline.ReleaseTransientData();

                Logger.Info($"[ObjectReconstruction] Starting: EP={executionProvider} " +
                    $"Mobile={mobileOptimized} Grid={gridResolution} Backend={meshExtractionBackend}");
                var sw = System.Diagnostics.Stopwatch.StartNew();

                ReportStatus("Loading models...");
                await _pipeline.LoadModelsAsync(_cts.Token);
                float loadMs = sw.ElapsedMilliseconds;
                sw.Restart();

                ReportStatus("Removing background...");
                preprocessed = await _pipeline.PreprocessAsync(image, _cts.Token);
                float rembgMs = sw.ElapsedMilliseconds;
                sw.Restart();

                ReportStatus("Running reconstruction...");
                await _pipeline.RunForwardAsync(preprocessed, _cts.Token);
                preprocessed = null;
                float forwardMs = sw.ElapsedMilliseconds;
                sw.Restart();

                ReportStatus("Extracting mesh...");
                var mesh = await _pipeline.ExtractMeshAsync(_cts.Token);
                float meshMs = sw.ElapsedMilliseconds;
                sw.Restart();

                // UV projection: compute per-vertex UVs from canonical camera view
                if (mesh != null && mesh.vertexCount > 0)
                {
                    ReportStatus("Projecting texture UVs...");
                    await TextureProjection.ApplyProjectionAsync(mesh);
                }
                float uvMs = sw.ElapsedMilliseconds;
                sw.Stop();

                float totalMs = loadMs + rembgMs + forwardMs + meshMs + uvMs;
                string timing = $"load={loadMs / 1000f:F1}s rembg={rembgMs / 1000f:F1}s " +
                    $"fwd={forwardMs / 1000f:F1}s mesh={meshMs / 1000f:F1}s | total={totalMs / 1000f:F1}s";
                var fwd = _pipeline.LastForwardTiming;
                string fwdDetail = $"fwd: {fwd}";
                Logger.Info($"[ObjectReconstruction] Done: EP={executionProvider} {timing}");
                Logger.Info($"[ObjectReconstruction] {fwdDetail}");
                ReportStatus($"Done ({totalMs / 1000f:F1}s)\n{timing}\n{fwdDetail}");

                return mesh;
            }
            catch (OperationCanceledException)
            {
                ReportStatus("Cancelled");
                return null;
            }
            catch (Exception e)
            {
                Logger.Error($"[ObjectReconstruction] {e.Message}");
                ReportStatus($"Error: {e.Message}");
                return null;
            }
            finally
            {
                _running = false;
                _cts?.Dispose();
                _cts = null;
            }
        }

        public async Task<Texture2D> TestRembgAsync(Texture2D image, CancellationToken ct = default)
        {
            if (_running)
            {
                Logger.Warning("[ObjectReconstruction] Pipeline already in progress");
                return null;
            }

            _running = true;
            float loadMs = 0, inferMs = 0;
            try
            {
                EnsurePipeline();

                ReportStatus($"Loading rembg (EP={executionProvider})...");
                var mask = await _pipeline.TestRembgAsync(image, ct,
                    lm => { loadMs = lm; ReportStatus($"Rembg loaded ({lm:F0}ms), inferring..."); },
                    im => { inferMs = im; });

                float totalMs = loadMs + inferMs;
                string result = $"EP={executionProvider} load={loadMs:F0}ms infer={inferMs:F0}ms total={totalMs:F0}ms";
                Logger.Info($"[ObjectReconstruction] Rembg test: {result}");

                var tex = MaskToTexture(mask, 320, 320);
                ReportStatus(result);
                return tex;
            }
            catch (Exception e)
            {
                string phase = loadMs == 0 ? "LOAD FAILED" : "INFER FAILED";
                string msg = $"{phase} (EP={executionProvider}): {e.Message}";
                Logger.Error($"[ObjectReconstruction] Rembg test {msg}");
                if (e.InnerException != null)
                    Logger.Error($"[ObjectReconstruction] Inner: {e.InnerException.Message}");
                ReportStatus(msg);
                return null;
            }
            finally
            {
                _running = false;
            }
        }

        private static Texture2D MaskToTexture(float[] mask, int w, int h)
        {
            if (mask == null || mask.Length != w * h) return null;
            var tex = new Texture2D(w, h, TextureFormat.R8, false);
            var pixels = tex.GetRawTextureData<byte>();
            for (int i = 0; i < mask.Length; i++)
                pixels[i] = (byte)(Mathf.Clamp01(mask[i]) * 255f);
            tex.Apply();
            return tex;
        }

        public void Cancel()
        {
            _cts?.Cancel();
        }

        internal void ResetPipeline()
        {
            _pipeline?.Dispose();
            _pipeline = null;
        }

        private void EnsurePipeline()
        {
            _pipeline ??= new ReconstructionPipeline(
                gridResolution,
                densityThreshold,
                triplaneGridSampleShader,
                densitySurfaceNetsShader,
                densityMarchingCubesShader,
                decoderPostprocessShader,
                meshAlgorithm,
                preloadModels: !mobileOptimized,
                executionProvider: executionProvider,
                mobileOptimized: mobileOptimized,
                densitySmoothPasses: densitySmoothPasses,
                meshBackend: meshExtractionBackend,
                laplacianSmoothIterations: laplacianSmoothIterations,
                laplacianSmoothLambda: laplacianSmoothLambda);
        }

        /// <summary>
        /// The preprocessed source image from the last reconstruction, for texture projection.
        /// </summary>
        public Texture2D LastPreprocessedImage => _pipeline?.LastPreprocessedImage;

        public Material CreateMaterial()
        {
            return CreateMaterial(null);
        }

        /// <summary>
        /// Creates a material for the reconstructed mesh. If a projected texture is provided,
        /// uses the ProjectedTexture shader that blends texture with vertex colors.
        /// Otherwise uses the vertex color shader.
        /// </summary>
        public Material CreateMaterial(Texture2D projectedTexture)
        {
            if (projectedTexture != null && projectedTextureShader != null)
            {
                var mat = new Material(projectedTextureShader);
                mat.SetTexture("_MainTex", projectedTexture);
                return mat;
            }

            if (vertexColorShader != null)
                return new Material(vertexColorShader);

            var fallback = Shader.Find("Universal Render Pipeline/Lit")
                           ?? Shader.Find("Standard");
            if (fallback == null)
                throw new InvalidOperationException(
                    "No vertex color shader assigned and no fallback shader found.");
            return new Material(fallback);
        }

        private void ReportStatus(string status)
        {
            _status = status;
            StatusChanged?.Invoke(status);
        }

        private void OnDestroy()
        {
            _cts?.Cancel();
            _pipeline?.Dispose();
        }
    }
}
#endif
