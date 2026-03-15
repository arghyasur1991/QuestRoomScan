using System;
using System.IO;
using System.Threading.Tasks;
using Genesis.RoomScan.GSplat;
using Genesis.RoomScan.UI;
using UnityEngine;

namespace Genesis.RoomScan
{
    public enum ScanMode
    {
        Passive,
        Guided
    }

    public enum ScanVisualization
    {
        VertexColored,
        Wireframe,
        OcclusionOnly,
        Hidden
    }

    public enum ScanRenderMode
    {
        Mesh,
        Splat,
        Both
    }

    /// <summary>
    /// Top-level orchestrator for room scanning. All sibling components live on
    /// the same GameObject and are resolved automatically via GetComponent.
    /// Input bindings are handled by <see cref="RoomScanInputHandler"/> (optional).
    /// </summary>
    [RequireComponent(typeof(DepthCapture), typeof(VolumeIntegrator), typeof(MeshExtractor))]
    [RequireComponent(typeof(PassthroughCameraProvider), typeof(TriplanarCache), typeof(RoomScanPersistence))]
    [RequireComponent(typeof(KeyframeCollector), typeof(PointCloudExporter), typeof(GSplatManager))]
    [RequireComponent(typeof(GSplatServerClient))]
    public class RoomScanner : MonoBehaviour
    {
        public static RoomScanner Instance { get; private set; }

        [Header("Scan Settings")]
        [SerializeField] private ScanMode mode = ScanMode.Passive;
        [SerializeField] private ScanVisualization visualization = ScanVisualization.VertexColored;
        [SerializeField] private bool autoStartOnLoad = true;

        [Header("Passive Mode Rates")]
        [SerializeField] private float passiveIntegrationHz = 30f;
        [SerializeField] private float passiveMeshExtractionHz = 30f;

        [Header("Guided Mode Rates")]
        [SerializeField] private float guidedIntegrationHz = 30f;
        [SerializeField] private float guidedMeshExtractionHz = 30f;

        [Header("Mesh Quality")]
        [SerializeField] private int minIntegrationsBeforeMesh = 5;

        [Header("Render Mode")]
        [SerializeField] private ScanRenderMode renderMode = ScanRenderMode.Mesh;

        [Header("Guided Mode")]
        [SerializeField] private float guidedTimeoutSeconds = 60f;

        [Header("Persistence")]
        [SerializeField, Tooltip("Seconds between autosaves during scanning (0 = disabled)")]
        private float autoSaveIntervalSeconds = 10f;

        // ─────────────────────────────────────────────────────────────
        //  Sibling component cache (resolved in Awake)
        // ─────────────────────────────────────────────────────────────

        private DepthCapture _depthCapture;
        private VolumeIntegrator _volumeIntegrator;
        private MeshExtractor _meshExtractor;
        private PassthroughCameraProvider _cameraProvider;
        private TriplanarCache _triplanarCache;
        private RoomScanPersistence _persistence;
        private KeyframeCollector _keyframeCollector;
        private PointCloudExporter _pointCloudExporter;
        private GSplatManager _gsplatManager;
        private GSplatServerClient _gsplatServerClient;
        private DebugMenuController _debugMenu;
        private ICameraProvider _customCameraProvider;

        // ─────────────────────────────────────────────────────────────
        //  Public read-only state
        // ─────────────────────────────────────────────────────────────

        public ScanMode Mode
        {
            get => mode;
            set => SetMode(value);
        }

        public ScanVisualization Visualization
        {
            get => visualization;
            set => SetVisualization(value);
        }

        public bool IsScanning { get; private set; }
        public ScanRenderMode CurrentRenderMode => renderMode;
        public bool IsGsTrainingInProgress => _serverTrainingInProgress;
        public DebugMenuController DebugMenu => _debugMenu;

        // ─────────────────────────────────────────────────────────────
        //  Events
        // ─────────────────────────────────────────────────────────────

        public event Action<ScanMode> ModeChanged;
        public event Action ScanStarted;
        public event Action ScanStopped;
        public event Action<ScanRenderMode> RenderModeChanged;

        // ─────────────────────────────────────────────────────────────
        //  Private state
        // ─────────────────────────────────────────────────────────────

        private float _lastIntegrationTime;
        private float _lastMeshTime;
        private float _guidedStartTime;
        private float _lastAutoSaveTime;
        private bool _started;
        private bool _serverTrainingInProgress;

        private float IntegrationInterval => 1f / (mode == ScanMode.Guided ? guidedIntegrationHz : passiveIntegrationHz);
        private float MeshInterval => 1f / (mode == ScanMode.Guided ? guidedMeshExtractionHz : passiveMeshExtractionHz);

        // ─────────────────────────────────────────────────────────────
        //  Lifecycle
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            Instance = this;
            CacheComponents();
            SetSafeShaderDefaults();
        }

        private void Start()
        {
            SetupHeadExclusion();
            OnRoomReady();
        }

        private void CacheComponents()
        {
            _depthCapture = GetComponent<DepthCapture>();
            _volumeIntegrator = GetComponent<VolumeIntegrator>();
            _meshExtractor = GetComponent<MeshExtractor>();
            _cameraProvider = GetComponent<PassthroughCameraProvider>();
            _triplanarCache = GetComponent<TriplanarCache>();
            _persistence = GetComponent<RoomScanPersistence>();
            _keyframeCollector = GetComponent<KeyframeCollector>();
            _pointCloudExporter = GetComponent<PointCloudExporter>();
            _gsplatManager = GetComponent<GSplatManager>();
            _gsplatServerClient = GetComponent<GSplatServerClient>();
            _debugMenu = GetComponentInChildren<DebugMenuController>();
        }

        private async void OnRoomReady()
        {
            if (_persistence != null && _persistence.HasSavedScan())
            {
                Debug.Log("[RoomScan] Found saved scan, loading...");
                await _persistence.LoadAsync();
            }

            if (autoStartOnLoad)
                StartScanning();

            _started = true;
            Debug.Log("[RoomScan] Room ready, scanning started");
        }

        private void OnEnable()
        {
            if (_started && autoStartOnLoad && !IsScanning)
                StartScanning();
        }

        private void OnDisable()
        {
            StopScanning();
        }

        private async void OnApplicationPause(bool paused)
        {
            if (!paused) return;

            if (_persistence != null && _started && _volumeIntegrator != null
                && _volumeIntegrator.IntegrationCount > _volumeIntegrator.WarmupIntegrations)
            {
                Debug.Log("[RoomScan] App pausing, saving scan...");
                await _persistence.SaveAsync();
            }
        }

        private async void OnApplicationQuit()
        {
            if (_persistence != null && _started && _volumeIntegrator != null
                && _volumeIntegrator.IntegrationCount > _volumeIntegrator.WarmupIntegrations
                && !_persistence.IsSaving)
            {
                Debug.Log("[RoomScan] App quitting, saving scan...");
                await _persistence.SaveAsync();
            }
        }

        private float _lastScannerLog;
        private int _integrateCount;

        private void Update()
        {
            if (!IsScanning || !DepthCapture.DepthAvailable) return;

            float t = Time.time;

            if (t - _lastIntegrationTime >= IntegrationInterval)
            {
                _lastIntegrationTime = t;

                ProvideColorFrame();
                _volumeIntegrator.Integrate();
                _integrateCount++;

                int effectiveCount = _volumeIntegrator.IntegrationCount - _volumeIntegrator.WarmupIntegrations;
                if (effectiveCount >= minIntegrationsBeforeMesh
                    && t - _lastMeshTime >= MeshInterval)
                {
                    _lastMeshTime = t;
                    _meshExtractor.Extract();
                }
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Debug.Log($"[RoomScan] Scanner: integrations={_integrateCount}, mode={mode}, " +
                    $"depthAvail={DepthCapture.DepthAvailable}");
            }

            if (mode == ScanMode.Guided && t - _guidedStartTime >= guidedTimeoutSeconds)
                SetMode(ScanMode.Passive);

            if (autoSaveIntervalSeconds > 0 && _persistence != null && !_persistence.IsSaving
                && t - _lastAutoSaveTime >= autoSaveIntervalSeconds
                && _volumeIntegrator.IntegrationCount > _volumeIntegrator.WarmupIntegrations)
            {
                _lastAutoSaveTime = t;
                _ = _persistence.SaveAsync();
            }
        }

        // ═════════════════════════════════════════════════════════════
        //  PUBLIC API — call from any client, input handler, or UI
        // ═════════════════════════════════════════════════════════════

        public void StartScanning()
        {
            if (IsScanning) return;
            IsScanning = true;

            float t = Time.time;
            _lastIntegrationTime = t;
            _lastMeshTime = t;
            _lastAutoSaveTime = t;

            ICameraProvider provider = GetActiveCameraProvider();
            provider?.StartCapture();

            ScanStarted?.Invoke();
        }

        public void StopScanning()
        {
            if (!IsScanning) return;
            IsScanning = false;

            ICameraProvider provider = GetActiveCameraProvider();
            provider?.StopCapture();

            ScanStopped?.Invoke();
        }

        public void ToggleScanning()
        {
            if (IsScanning) StopScanning();
            else StartScanning();
        }

        public void SetMode(ScanMode newMode)
        {
            if (mode == newMode) return;
            mode = newMode;

            if (mode == ScanMode.Guided)
                _guidedStartTime = Time.time;

            ModeChanged?.Invoke(mode);
        }

        public void SetVisualization(ScanVisualization vis)
        {
            visualization = vis;
            ApplyVisualization();
        }

        public void ClearScan()
        {
            _volumeIntegrator.Clear();
            _meshExtractor.Reinitialize();
            if (_triplanarCache != null) _triplanarCache.Clear();
        }

        /// <summary>
        /// Clears all persisted data: in-memory scan, saved scan files, triplanar
        /// textures, and GSExport (keyframes + point cloud). Safe to call at runtime.
        /// </summary>
        public void ClearAllData()
        {
            StopScanning();
            ClearScan();

            if (_persistence != null)
                _persistence.DeleteSavedScan();

            if (_keyframeCollector != null)
                _keyframeCollector.ClearExport();

            string gsExportDir = Path.Combine(Application.persistentDataPath, "GSExport");
            if (Directory.Exists(gsExportDir))
                Directory.Delete(gsExportDir, true);

            Debug.Log("[RoomScan] All scan + export data cleared");

            if (autoStartOnLoad)
                StartScanning();
        }

        public void SetRenderMode(ScanRenderMode newMode)
        {
            renderMode = newMode;
            ApplyRenderMode();
            RenderModeChanged?.Invoke(renderMode);
            Debug.Log($"[RoomScan] Render mode: {renderMode}");
        }

        public void CycleRenderMode()
        {
            var next = renderMode switch
            {
                ScanRenderMode.Mesh => ScanRenderMode.Splat,
                ScanRenderMode.Splat => ScanRenderMode.Both,
                _ => ScanRenderMode.Mesh,
            };
            SetRenderMode(next);
        }

        public void FreezeInView()
        {
            if (_volumeIntegrator == null) return;
            if (!TryGetCameraIntrinsics(out var pose, out var focal, out var principal,
                    out var sensor, out var current)) return;

            _volumeIntegrator.FreezeInView(pose.position, pose.rotation,
                focal, principal, sensor, current);
        }

        public void UnfreezeInView()
        {
            if (_volumeIntegrator == null) return;
            if (!TryGetCameraIntrinsics(out var pose, out var focal, out var principal,
                    out var sensor, out var current)) return;

            _volumeIntegrator.UnfreezeInView(pose.position, pose.rotation,
                focal, principal, sensor, current);
        }

        public async Task ExportPointCloudAsync()
        {
            if (_pointCloudExporter != null)
                await _pointCloudExporter.ExportAsync();
        }

        public void StartServerTraining()
        {
            if (_serverTrainingInProgress) return;
            RunServerTrainingAsync();
        }

        public void ToggleDebugMenu()
        {
            if (_debugMenu != null) _debugMenu.Toggle();
        }

        /// <summary>
        /// Set a custom camera provider (overrides PassthroughCameraProvider).
        /// </summary>
        public void SetCameraProvider(ICameraProvider provider)
        {
            _customCameraProvider = provider;
        }

        public void AddExclusionZone(Transform t)
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ExclusionZones.Add(t);
        }

        public void RemoveExclusionZone(Transform t)
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ExclusionZones.Remove(t);
        }

        // ─────────────────────────────────────────────────────────────
        //  Internal helpers
        // ─────────────────────────────────────────────────────────────

        private bool TryGetCameraIntrinsics(out Pose pose, out Vector2 focal,
            out Vector2 principal, out Vector2 sensor, out Vector2 current)
        {
            pose = default;
            focal = principal = sensor = current = default;

            ICameraProvider provider = GetActiveCameraProvider();
            if (provider is not PassthroughCameraProvider pcp || !pcp.IsReady)
                return false;

            pose = pcp.CameraPose;
            focal = pcp.FocalLength;
            principal = pcp.PrincipalPoint;
            sensor = pcp.SensorResolution;
            current = pcp.CurrentResolution;
            return true;
        }

        private void SetupHeadExclusion()
        {
            if (_volumeIntegrator == null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                AddExclusionZone(cam.transform);
                Debug.Log($"[RoomScan] Head exclusion zone added: {cam.gameObject.name}");
            }
            else
            {
                Debug.LogWarning("[RoomScan] No main camera found for head exclusion zone");
            }
        }

        private void SetSafeShaderDefaults()
        {
            Shader.SetGlobalFloat(Shader.PropertyToID("_RSTriAvailable"), 0f);
        }

        private int _colorFrameLog;
        private void ProvideColorFrame()
        {
            ICameraProvider provider = GetActiveCameraProvider();

            if (provider is PassthroughCameraProvider pcp && pcp.IsReady)
            {
                Texture frame = pcp.CurrentFrame;
                if (frame != null)
                {
                    Pose pose = pcp.CameraPose;
                    Vector2 focal = pcp.FocalLength;
                    Vector2 principal = pcp.PrincipalPoint;
                    Vector2 sensor = pcp.SensorResolution;
                    Vector2 current = pcp.CurrentResolution;

                    _volumeIntegrator.SetCameraData(
                        frame, pose.position, pose.rotation,
                        focal, principal, sensor, current);

                    _keyframeCollector.TrySaveKeyframe(frame, pose.position, pose.rotation,
                        focal, principal, sensor, current);

                    if (_triplanarCache != null)
                    {
                        _triplanarCache.DispatchBake(frame, pose.position, pose.rotation,
                            focal, principal, sensor, current,
                            _volumeIntegrator.CameraExposure,
                            _volumeIntegrator.ExclusionZones);
                    }

                    _colorFrameLog++;
                    if (_colorFrameLog <= 3 || _colorFrameLog % 50 == 0)
                        Debug.Log($"[RoomScan] ColorFrame #{_colorFrameLog}: " +
                            $"frame={frame.width}x{frame.height}, " +
                            $"triCache={_triplanarCache != null}");

                    return;
                }
            }

            _colorFrameLog++;
            if (_colorFrameLog <= 5)
                Debug.Log($"[RoomScan] ColorFrame #{_colorFrameLog}: NO CAMERA " +
                    $"provider={provider?.GetType().Name ?? "null"}, " +
                    $"isPcp={provider is PassthroughCameraProvider}, " +
                    $"isReady={((provider as PassthroughCameraProvider)?.IsReady ?? false)}");

            _volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private void ApplyRenderMode()
        {
            var gpuRenderer = _meshExtractor != null ? _meshExtractor.GetComponent<GPUMeshRenderer>() : null;
            var splatRend = _gsplatManager != null
                ? _gsplatManager.GetComponent<GSRenderer>()
                : null;

            if (gpuRenderer != null)
                gpuRenderer.enabled = renderMode == ScanRenderMode.Mesh || renderMode == ScanRenderMode.Both;
            if (splatRend != null)
                splatRend.enabled = renderMode == ScanRenderMode.Splat || renderMode == ScanRenderMode.Both;
        }

        private async void RunServerTrainingAsync()
        {
            if (_serverTrainingInProgress) return;
            _serverTrainingInProgress = true;

            try
            {
                Debug.Log("[RoomScan] Starting server-side GS training pipeline...");

                Debug.Log("[RoomScan] Exporting point cloud...");
                await _pointCloudExporter.ExportAsync();

                Debug.Log("[RoomScan] Uploading training data to PC server...");
                bool uploaded = await _gsplatServerClient.UploadTrainingData();
                if (!uploaded) { Debug.LogError("[RoomScan] Upload failed"); return; }

                Debug.Log("[RoomScan] Waiting for server training to complete...");
                bool success = await _gsplatServerClient.PollUntilDone();
                if (!success) { Debug.LogError("[RoomScan] Server training failed"); return; }

                Debug.Log("[RoomScan] Downloading trained Gaussians...");
                byte[] plyData = await _gsplatServerClient.DownloadResult();
                if (plyData == null || plyData.Length == 0) { Debug.LogError("[RoomScan] Download empty"); return; }

                _gsplatManager.LoadTrainedPly(plyData);
                Debug.Log("[RoomScan] Trained Gaussians loaded and ready for rendering");
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] Server training pipeline error: {e.Message}\n{e.StackTrace}");
            }
            finally
            {
                _serverTrainingInProgress = false;
            }
        }

        private ICameraProvider GetActiveCameraProvider()
        {
            if (_customCameraProvider != null) return _customCameraProvider;
            return _cameraProvider;
        }

        private void ApplyVisualization()
        {
            if (_meshExtractor == null) return;
            var gpuRenderer = _meshExtractor.GetComponent<GPUMeshRenderer>();
            if (gpuRenderer == null) return;

            gpuRenderer.enabled = visualization != ScanVisualization.Hidden;
        }
    }
}
