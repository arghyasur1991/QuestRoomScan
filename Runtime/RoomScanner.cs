using System;
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
        Lit,
        Wireframe,
        OcclusionOnly,
        Hidden
    }

    /// <summary>
    /// Top-level orchestrator for room scanning. Manages the scanning pipeline:
    /// DepthCapture → VolumeIntegrator → MeshExtractor.
    /// Supports passive (background) and guided (active) scan modes.
    /// </summary>
    public class RoomScanner : MonoBehaviour
    {
        public static RoomScanner Instance { get; private set; }

        [Header("Components")]
        [SerializeField] private DepthCapture depthCapture;
        [SerializeField] private VolumeIntegrator volumeIntegrator;
        [SerializeField] private MeshExtractor meshExtractor;
        [SerializeField] private TriplanarCache triplanarCache;
        [SerializeField] private KeyframeStore keyframeStore;
        [SerializeField] private RoomScanPersistence persistence;
        [SerializeField] private KeyframeCollector keyframeCollector;
        [SerializeField] private PointCloudExporter pointCloudExporter;
        [SerializeField] private PlaneDetector planeDetector;
        [SerializeField] private LightEstimator lightEstimator;

        [Header("Camera")]
        [SerializeField] private PassthroughCameraProvider cameraProvider;
        private ICameraProvider _customCameraProvider;

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

        [Header("Guided Mode")]
        [SerializeField] private float guidedTimeoutSeconds = 60f;

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

        public event Action<ScanMode> ModeChanged;
        public event Action ScanStarted;
        public event Action ScanStopped;

        private float _lastIntegrationTime;
        private float _lastMeshTime;
        private float _guidedStartTime;
        private float _lastAutoSaveTime;
        private bool _started;

        [Header("Persistence")]
        [SerializeField, Tooltip("Seconds between autosaves during scanning (0 = disabled)")]
        private float autoSaveIntervalSeconds = 10f;

        private float IntegrationInterval => 1f / (mode == ScanMode.Guided ? guidedIntegrationHz : passiveIntegrationHz);
        private float MeshInterval => 1f / (mode == ScanMode.Guided ? guidedMeshExtractionHz : passiveMeshExtractionHz);

        private void Awake()
        {
            Instance = this;
            SetSafeShaderDefaults();
        }

        private async void Start()
        {
            ValidateComponents();
            SetupHeadExclusion();

            if (persistence != null && persistence.HasSavedScan())
            {
                Debug.Log("[RoomScan] Found saved scan, loading...");
                await persistence.LoadAsync();
            }

            if (autoStartOnLoad)
                StartScanning();

            _started = true;
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

            Debug.Log($"[RoomScan] OnApplicationPause: persistence={persistence != null}, " +
                      $"started={_started}, vi={volumeIntegrator != null}, " +
                      $"intCount={volumeIntegrator?.IntegrationCount ?? -1}, " +
                      $"warmup={volumeIntegrator?.WarmupIntegrations ?? -1}");

            if (persistence != null && _started && volumeIntegrator != null
                && volumeIntegrator.IntegrationCount > volumeIntegrator.WarmupIntegrations)
            {
                Debug.Log("[RoomScan] App pausing, saving scan...");
                await persistence.SaveAsync();
            }
        }

        private async void OnApplicationQuit()
        {
            Debug.Log($"[RoomScan] OnApplicationQuit: persistence={persistence != null}, " +
                      $"started={_started}, intCount={volumeIntegrator?.IntegrationCount ?? -1}");

            if (persistence != null && _started && volumeIntegrator != null
                && volumeIntegrator.IntegrationCount > volumeIntegrator.WarmupIntegrations
                && !persistence.IsSaving)
            {
                Debug.Log("[RoomScan] App quitting, saving scan...");
                await persistence.SaveAsync();
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
                volumeIntegrator.Integrate();
                _integrateCount++;

                int effectiveCount = volumeIntegrator.IntegrationCount - volumeIntegrator.WarmupIntegrations;
                if (effectiveCount >= minIntegrationsBeforeMesh
                    && t - _lastMeshTime >= MeshInterval)
                {
                    _lastMeshTime = t;
                    meshExtractor.Extract();

                    if (planeDetector != null)
                        planeDetector.OnMeshCycleComplete();

                    if (lightEstimator != null)
                        lightEstimator.OnMeshCycleComplete();
                }
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                ICameraProvider camProv = GetActiveCameraProvider();
                Debug.Log($"[RoomScan] Scanner: integrations={_integrateCount}, mode={mode}, " +
                    $"depthAvail={DepthCapture.DepthAvailable}");
            }

            if (mode == ScanMode.Guided && t - _guidedStartTime >= guidedTimeoutSeconds)
            {
                SetMode(ScanMode.Passive);
            }

            if (autoSaveIntervalSeconds > 0 && persistence != null && !persistence.IsSaving
                && t - _lastAutoSaveTime >= autoSaveIntervalSeconds
                && volumeIntegrator != null
                && volumeIntegrator.IntegrationCount > volumeIntegrator.WarmupIntegrations)
            {
                _lastAutoSaveTime = t;
                _ = persistence.SaveAsync();
            }

#if HAS_META_XR_SDK
            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.LTouch))
                CycleVisualization();
#endif
            if (Input.GetKeyDown(KeyCode.L))
                CycleVisualization();
        }

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

        /// <summary>
        /// Cycle through visualization modes. Call from controller input or UI.
        /// Order: VertexColored → Lit → Wireframe → OcclusionOnly → Hidden → repeat.
        /// </summary>
        public void CycleVisualization()
        {
            int count = System.Enum.GetValues(typeof(ScanVisualization)).Length;
            int next = ((int)visualization + 1) % count;
            SetVisualization((ScanVisualization)next);
            Debug.Log($"[RoomScan] Visualization: {visualization}");
        }

        public void ClearScan()
        {
            volumeIntegrator.Clear();
            meshExtractor.Reinitialize();
            if (triplanarCache != null) triplanarCache.Clear();
        }

        /// <summary>
        /// Set a custom camera provider (overrides WebCamProvider).
        /// Call before StartScanning or during runtime.
        /// </summary>
        public void SetCameraProvider(ICameraProvider provider)
        {
            _customCameraProvider = provider;
        }

        /// <summary>
        /// Add a transform to the exclusion zone list (e.g. player head, tracked controller).
        /// </summary>
        public void AddExclusionZone(Transform t)
        {
            if (volumeIntegrator != null)
                volumeIntegrator.ExclusionZones.Add(t);
        }

        public void RemoveExclusionZone(Transform t)
        {
            if (volumeIntegrator != null)
                volumeIntegrator.ExclusionZones.Remove(t);
        }

        private void ValidateComponents()
        {
            if (depthCapture == null) depthCapture = FindFirstObjectByType<DepthCapture>();
            if (volumeIntegrator == null) volumeIntegrator = FindFirstObjectByType<VolumeIntegrator>();
            if (meshExtractor == null) meshExtractor = FindFirstObjectByType<MeshExtractor>();
            if (triplanarCache == null) triplanarCache = FindFirstObjectByType<TriplanarCache>();
            if (keyframeStore == null) keyframeStore = FindFirstObjectByType<KeyframeStore>();
            if (persistence == null) persistence = FindFirstObjectByType<RoomScanPersistence>();
            if (keyframeCollector == null) keyframeCollector = FindFirstObjectByType<KeyframeCollector>();
            if (pointCloudExporter == null) pointCloudExporter = FindFirstObjectByType<PointCloudExporter>();
            if (planeDetector == null) planeDetector = FindFirstObjectByType<PlaneDetector>();
            if (lightEstimator == null) lightEstimator = FindFirstObjectByType<LightEstimator>();

            if (depthCapture == null) Debug.LogError("[RoomScan] DepthCapture not found");
            if (volumeIntegrator == null) Debug.LogError("[RoomScan] VolumeIntegrator not found");
            if (meshExtractor == null) Debug.LogError("[RoomScan] MeshExtractor not found");
        }

        private void SetupHeadExclusion()
        {
            if (volumeIntegrator == null) return;

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
            Shader.SetGlobalInt(Shader.PropertyToID("_RSKeyframeCount"), 0);
            Shader.SetGlobalFloat(Shader.PropertyToID("_RSCamExposure"), 3f);
            Shader.SetGlobalFloat(Shader.PropertyToID("_RSTriAvailable"), 0f);

            var dummyVecs = new Vector4[112];
            Shader.SetGlobalVectorArray(Shader.PropertyToID("_RSKeyframeData"), dummyVecs);
        }

        private int _colorFrameLog;
        private void ProvideColorFrame()
        {
            if (volumeIntegrator == null) return;
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

                    volumeIntegrator.SetCameraData(
                        frame, pose.position, pose.rotation,
                        focal, principal, sensor, current);

                    if (keyframeStore != null)
                    {
                        keyframeStore.SetLiveFrame(frame, pose.position, pose.rotation,
                            focal, principal, sensor, current);
                        keyframeStore.TryInsertKeyframe(frame, pose.position, pose.rotation,
                            focal, principal, sensor, current);
                    }

                    if (keyframeCollector != null)
                    {
                        keyframeCollector.TrySaveKeyframe(frame, pose.position, pose.rotation,
                            focal, principal, sensor, current);
                    }

                    if (triplanarCache != null)
                    {
                        triplanarCache.DispatchBake(frame, pose.position, pose.rotation,
                            focal, principal, sensor, current,
                            volumeIntegrator.CameraExposure,
                            volumeIntegrator.ExclusionZones);
                    }

                    _colorFrameLog++;
                    if (_colorFrameLog <= 3 || _colorFrameLog % 50 == 0)
                        Debug.Log($"[RoomScan] ColorFrame #{_colorFrameLog}: " +
                            $"frame={frame.width}x{frame.height}, " +
                            $"triCache={triplanarCache != null}, " +
                            $"keyframes={keyframeStore != null}");

                    return;
                }
            }

            _colorFrameLog++;
            if (_colorFrameLog <= 5)
                Debug.Log($"[RoomScan] ColorFrame #{_colorFrameLog}: NO CAMERA " +
                    $"provider={provider?.GetType().Name ?? "null"}, " +
                    $"isPcp={provider is PassthroughCameraProvider}, " +
                    $"isReady={((provider as PassthroughCameraProvider)?.IsReady ?? false)}");

            volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private ICameraProvider GetActiveCameraProvider()
        {
            if (_customCameraProvider != null) return _customCameraProvider;
            return cameraProvider;
        }

        private void ApplyVisualization()
        {
            if (meshExtractor == null) return;
            var gpuRenderer = meshExtractor.GetComponent<GPUMeshRenderer>();
            if (gpuRenderer == null) return;

            gpuRenderer.enabled = visualization != ScanVisualization.Hidden;

            bool lit = visualization == ScanVisualization.Lit;
            gpuRenderer.SetLitMode(lit);

            if (lightEstimator != null)
                lightEstimator.SetMarkersVisible(lit);
        }
    }
}
