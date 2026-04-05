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
                sw.Stop();

                float totalMs = loadMs + rembgMs + forwardMs + meshMs;
                Logger.Info($"[ObjectReconstruction] Timing: load={loadMs:F0}ms rembg={rembgMs:F0}ms " +
                    $"forward={forwardMs:F0}ms mesh={meshMs:F0}ms total={totalMs:F0}ms");
                ReportStatus($"Done! ({totalMs / 1000f:F1}s)");

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
                executionProvider: executionProvider,
                mobileOptimized: mobileOptimized,
                densitySmoothPasses: densitySmoothPasses);
        }

        public Material CreateMaterial()
        {
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
