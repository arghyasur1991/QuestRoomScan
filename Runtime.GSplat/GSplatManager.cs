using System.Threading.Tasks;
using GaussianSplatting.Runtime;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Loads server-trained Gaussian splats from PLY data and renders them
    /// via the Unity Gaussian Splatting (UGS) package's <see cref="GaussianSplatRenderer"/>
    /// on a dedicated child GameObject whose transform can be set for room-anchor relocation.
    /// </summary>
    [RequireComponent(typeof(KeyframeCollector))]
    public class GSplatManager : MonoBehaviour, IRoomScanModule, IGSplatProvider
    {
        public string ModuleName => "Gaussian Splat";

        private RoomScanner _scanner;
        private GSplatServerClient _serverClient;
        GaussianSplatRenderer _ugsRenderer;
        Transform _splatHolder;

        public bool HasServerTrainedSplats =>
            _ugsRenderer != null && _ugsRenderer.isRuntimeLoaded && _ugsRenderer.splatCount > 0;

        /// <summary>
        /// The child transform hosting the GaussianSplatRenderer.
        /// Set its position/rotation to apply room-anchor relocation.
        /// </summary>
        public Transform SplatHolder => _splatHolder;

        /// <summary>
        /// Toggle splat visibility without releasing GPU resources.
        /// </summary>
        public bool RenderVisible
        {
            get => _ugsRenderer != null && _ugsRenderer.renderVisible;
            set { if (_ugsRenderer != null) _ugsRenderer.renderVisible = value; }
        }

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _scanner = scanner;
            _serverClient = GetComponent<GSplatServerClient>();
        }

        void Awake()
        {
            EnsureRendererOnChild();
        }

        void EnsureRendererOnChild()
        {
            if (_ugsRenderer != null) return;

            // Migrate: if GaussianSplatRenderer is on this GO (legacy), move it to a child.
            var existingOnSelf = GetComponent<GaussianSplatRenderer>();

            var child = transform.Find("SplatRenderer");
            if (child == null)
            {
                var go = new GameObject("SplatRenderer");
                go.transform.SetParent(transform, false);
                child = go.transform;
            }
            _splatHolder = child;

            _ugsRenderer = child.GetComponent<GaussianSplatRenderer>();
            if (_ugsRenderer == null)
            {
                if (existingOnSelf != null)
                {
                    // Can't move components at runtime; create a new one and copy serialized
                    // shader/compute refs. The setup wizard will re-wire them on next run.
                    _ugsRenderer = child.gameObject.AddComponent<GaussianSplatRenderer>();
                    CopyRendererSettings(existingOnSelf, _ugsRenderer);
                    Destroy(existingOnSelf);
                }
                else
                {
                    _ugsRenderer = child.gameObject.AddComponent<GaussianSplatRenderer>();
                }
            }
        }

        static void CopyRendererSettings(GaussianSplatRenderer src, GaussianSplatRenderer dst)
        {
            // Copy serialized shader/compute references via reflection-free public fields.
            // GaussianSplatRenderer exposes these as serialized fields set by the wizard.
            // If the fields aren't accessible, the wizard will re-wire on next "Fix All".
            try
            {
                var srcJson = JsonUtility.ToJson(src);
                JsonUtility.FromJsonOverwrite(srcJson, dst);
            }
            catch
            {
                Logger.Warning("Could not copy renderer settings; run Setup Wizard to re-wire shaders");
            }
        }

        /// <summary>
        /// Parses a 3DGS-format PLY, converts from COLMAP to Unity coordinates,
        /// and uploads to GPU buffers for UGS rendering.
        /// </summary>
        public void LoadTrainedPly(byte[] plyData)
        {
            EnsureRendererOnChild();

            if (_ugsRenderer == null)
            {
                Logger.Error("No GaussianSplatRenderer component found");
                return;
            }

            GaussianSplatPlyLoader.LoadFromPlyBytes(_ugsRenderer, plyData, colmapToUnity: true);
            Logger.Info($"Loaded trained splat via UGS ({_ugsRenderer.splatCount} Gaussians)");
        }

        /// <summary>Release all GPU resources for the loaded splat.</summary>
        public void ClearSplat()
        {
            if (_ugsRenderer != null)
                _ugsRenderer.ClearRuntimeSplatData();
        }

        public void ApplySplatRelocation(Matrix4x4 relocation)
        {
            if (_splatHolder != null)
            {
                _splatHolder.SetPositionAndRotation(
                    new Vector3(relocation.m03, relocation.m13, relocation.m23),
                    relocation.rotation);
            }
        }

        /// <summary>Reset the splat holder to local identity (no relocation offset).</summary>
        public void ResetSplatTransform()
        {
            if (_splatHolder != null)
            {
                _splatHolder.localPosition = Vector3.zero;
                _splatHolder.localRotation = Quaternion.identity;
                _splatHolder.localScale = Vector3.one;
            }
        }

        void OnDestroy()
        {
            ClearSplat();
        }

        public async Task<byte[]> RunServerTrainingAsync(string keyframeDir, UnityEngine.Matrix4x4 keyframeRelocation)
        {
            if (_serverClient == null) return null;

            if (PointCloudExporter.ExistsIn(keyframeDir))
            {
                Logger.Info("Reusing existing point cloud from package");
            }
            else
            {
                Logger.Info("Exporting point cloud...");
                await PointCloudExporter.ExportAsync(keyframeDir);
            }

            Logger.Info("Uploading training data to PC server...");
            bool uploaded = await _serverClient.UploadTrainingData(keyframeDir, keyframeRelocation);
            if (!uploaded) { Logger.Error("Upload failed"); return null; }

            Logger.Info("Waiting for server training to complete...");
            bool success = await _serverClient.PollUntilDone();
            if (!success) { Logger.Error("Server training failed"); return null; }

            Logger.Info("Downloading trained Gaussians...");
            byte[] plyData = await _serverClient.DownloadResult();
            if (plyData == null || plyData.Length == 0) { Logger.Error("Download empty"); return null; }

            Logger.Info($"Trained splat downloaded ({plyData.Length / (1024 * 1024f):F1}MB)");
            return plyData;
        }

        public async Task<byte[]> EnhanceAtlasAsync(byte[] pngBytes, int scale, bool inpaint)
        {
            if (_serverClient == null) return null;
            return await _serverClient.EnhanceAtlasAsync(pngBytes, scale, inpaint);
        }

        public async Task<byte[]> EnhanceMeshAsync(byte[] meshBin, int smoothIterations, bool enablePlaneSnap)
        {
            if (_serverClient == null) return null;
            return await _serverClient.EnhanceMeshAsync(meshBin, smoothIterations, enablePlaneSnap: enablePlaneSnap);
        }

        public async Task<bool> UploadTrainingDataAsync(string keyframeDir, UnityEngine.Matrix4x4 keyframeRelocation)
        {
            if (_serverClient == null) return false;
            return await _serverClient.UploadTrainingData(keyframeDir, keyframeRelocation);
        }

        // ── IGSplatProvider: server / training status ──

        public string ServerUrl
        {
            get => _serverClient != null ? _serverClient.ServerUrl : "";
            set { if (_serverClient != null) _serverClient.ServerUrl = value; }
        }

        public bool IsUploading => _serverClient != null && _serverClient.IsUploading;
        public bool IsDownloading => _serverClient != null && _serverClient.IsDownloading;
        public bool IsPolling => _serverClient != null && _serverClient.IsPolling;

        public async Task CancelTraining()
        {
            if (_serverClient != null)
                await _serverClient.CancelTraining();
        }

        public string TrainingState => _serverClient?.LastStatus?.state ?? "idle";
        public float TrainingProgress => _serverClient?.LastStatus?.progress ?? 0f;
        public string TrainingMessage => _serverClient?.LastStatus?.message;
        public string TrainingBackend => _serverClient?.LastStatus?.backend;
        public int CurrentIteration => _serverClient?.LastStatus?.current_iteration ?? 0;
        public int TotalIterations => _serverClient?.LastStatus?.total_iterations ?? 0;
        public float ElapsedSeconds => _serverClient?.LastStatus?.elapsed_seconds ?? 0f;
    }
}
