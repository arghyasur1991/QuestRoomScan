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

        [SerializeField, Tooltip("3x3x3 Gaussian smoothing passes on the density volume before " +
            "marching cubes. Filters INT8 quantization noise amplified by exp(). " +
            "0 = disabled, 1 = recommended for INT8, 2+ = more smoothing.")]
        private int densitySmoothPasses = 1;

        public string ModuleName => "Object Reconstruction";
        public bool IsRunning => _running;
        public string Status => _status;
        public Texture2D[] TestImages => testImages;

        public event Action<string> StatusChanged;

        private RoomScanner _scanner;
        private ReconstructionPipeline _pipeline;
        private bool _running;
        private string _status = "Idle";
        private CancellationTokenSource _cts;
        private GameObject _spawnedMesh;

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _scanner = scanner;
            Logger.Info("[ObjectReconstruction] Module initialized");
        }

        public async Task LoadModelsAsync(CancellationToken ct = default)
        {
            EnsurePipeline();
            ReportStatus("Loading models...");
            await _pipeline.LoadModelsAsync(ct);
            ReportStatus("Models loaded");
        }

        public async Task ReconstructAsync(Texture2D image, CancellationToken ct = default)
        {
            if (_running)
            {
                Logger.Warning("[ObjectReconstruction] Reconstruction already in progress");
                return;
            }

            _running = true;
            _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);

            float[] preprocessed = null;
            try
            {
                EnsurePipeline();
                ClearMesh();

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

                SpawnMesh(mesh);

                float totalMs = loadMs + rembgMs + forwardMs + meshMs;
                Logger.Info($"[ObjectReconstruction] Timing: load={loadMs:F0}ms rembg={rembgMs:F0}ms " +
                    $"forward={forwardMs:F0}ms mesh={meshMs:F0}ms total={totalMs:F0}ms");
                ReportStatus($"Done! ({totalMs / 1000f:F1}s)");
            }
            catch (OperationCanceledException)
            {
                ReportStatus("Cancelled");
            }
            catch (Exception e)
            {
                Logger.Error($"[ObjectReconstruction] {e.Message}");
                ReportStatus($"Error: {e.Message}");
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

        public void ClearMesh()
        {
            if (_spawnedMesh != null)
            {
                Destroy(_spawnedMesh);
                _spawnedMesh = null;
            }
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

        private void SpawnMesh(Mesh mesh)
        {
            ClearMesh();

            _spawnedMesh = new GameObject("ReconstructedObject");
            var mf = _spawnedMesh.AddComponent<MeshFilter>();
            var mr = _spawnedMesh.AddComponent<MeshRenderer>();

            mf.sharedMesh = mesh;
            mr.sharedMaterial = CreateVertexColorMaterial();

            var center = _scanner != null
                ? _scanner.transform.position + Vector3.forward * 1.5f
                : Vector3.forward * 1.5f;
            _spawnedMesh.transform.position = center;
            _spawnedMesh.transform.localScale = Vector3.one * 0.5f;

            Logger.Info($"[ObjectReconstruction] Mesh spawned: {mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris");
        }

        private Material CreateVertexColorMaterial()
        {
            if (vertexColorShader != null)
                return new Material(vertexColorShader);

            Logger.Warning("[ObjectReconstruction] vertexColorShader not assigned — " +
                "Shader.Find fallback may fail in stripped builds. Wire it via the Setup Wizard.");
            var fallback = Shader.Find("Universal Render Pipeline/Lit")
                           ?? Shader.Find("Standard");
            if (fallback == null)
                throw new InvalidOperationException(
                    "No vertex color shader assigned and no fallback shader found. " +
                    "Assign vertexColorShader in the Inspector or run the Setup Wizard.");
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
            ClearMesh();
        }
    }
}
#endif
