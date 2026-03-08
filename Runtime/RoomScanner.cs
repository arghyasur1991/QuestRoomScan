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
        Wireframe,
        OcclusionOnly,
        Hidden
    }

    /// <summary>
    /// Top-level orchestrator for room scanning. Manages the scanning pipeline:
    /// DepthCapture → VolumeIntegrator → ChunkManager → TextureProjector.
    /// Supports passive (background) and guided (active) scan modes.
    /// </summary>
    public class RoomScanner : MonoBehaviour
    {
        public static RoomScanner Instance { get; private set; }

        [Header("Components")]
        [SerializeField] private DepthCapture depthCapture;
        [SerializeField] private VolumeIntegrator volumeIntegrator;
        [SerializeField] private ChunkManager chunkManager;
        [SerializeField] private TextureProjector textureProjector;
        [SerializeField] private TriplanarCache triplanarCache;
        [SerializeField] private KeyframeStore keyframeStore;
        [SerializeField] private RoomScanPersistence persistence;

        [Header("Camera")]
        [SerializeField] private PassthroughCameraProvider cameraProvider;
        private ICameraProvider _customCameraProvider;

        [Header("Scan Settings")]
        [SerializeField] private ScanMode mode = ScanMode.Passive;
        [SerializeField] private ScanVisualization visualization = ScanVisualization.VertexColored;
        [SerializeField] private bool autoStartOnLoad = true;
        [SerializeField] private bool enableTextureProjection = true;

        [Header("Passive Mode Rates")]
        [SerializeField] private float passiveIntegrationHz = 3f;
        [SerializeField] private float passiveMeshExtractionHz = 1f;
        [SerializeField] private float passiveTextureProjectionHz = 5f;

        [Header("Guided Mode Rates")]
        [SerializeField] private float guidedIntegrationHz = 8f;
        [SerializeField] private float guidedMeshExtractionHz = 3f;
        [SerializeField] private float guidedTextureProjectionHz = 15f;

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
        private float _lastTextureTime;
        private float _guidedStartTime;
        private float _lastAutoSaveTime;
        private bool _started;

        [Header("Persistence")]
        [SerializeField, Tooltip("Seconds between autosaves during scanning (0 = disabled)")]
        private float autoSaveIntervalSeconds = 10f;

        private float IntegrationInterval => 1f / (mode == ScanMode.Guided ? guidedIntegrationHz : passiveIntegrationHz);
        private float MeshInterval => 1f / (mode == ScanMode.Guided ? guidedMeshExtractionHz : passiveMeshExtractionHz);
        private float TextureInterval => 1f / (mode == ScanMode.Guided ? guidedTextureProjectionHz : passiveTextureProjectionHz);

        private void Awake()
        {
            Instance = this;
            SetSafeShaderDefaults();
        }

        private async void Start()
        {
            ValidateComponents();
            SetupCameraProvider();
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
                    chunkManager.UpdateDirtyChunks();
                }
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                ICameraProvider camProv = GetActiveCameraProvider();
                Debug.Log($"[RoomScan] Scanner: integrations={_integrateCount}, mode={mode}, " +
                    $"depthAvail={DepthCapture.DepthAvailable}");
            }

            if (enableTextureProjection && t - _lastTextureTime >= TextureInterval)
            {
                _lastTextureTime = t;
                textureProjector.ProjectFrame();
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
        }

        public void StartScanning()
        {
            if (IsScanning) return;
            IsScanning = true;

            float t = Time.time;
            _lastIntegrationTime = t;
            _lastMeshTime = t;
            _lastTextureTime = t;
            _lastAutoSaveTime = t;

            ICameraProvider provider = GetActiveCameraProvider();
            if (enableTextureProjection && provider != null)
                provider.StartCapture();

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

        public void ClearScan()
        {
            volumeIntegrator.Clear();
            chunkManager.ClearAllChunks();
            if (triplanarCache != null) triplanarCache.Clear();
        }

        /// <summary>
        /// Set a custom camera provider (overrides WebCamProvider).
        /// Call before StartScanning or during runtime.
        /// </summary>
        public void SetCameraProvider(ICameraProvider provider)
        {
            _customCameraProvider = provider;
            if (textureProjector != null)
                textureProjector.SetCameraProvider(provider);
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
            if (chunkManager == null) chunkManager = FindFirstObjectByType<ChunkManager>();
            if (textureProjector == null) textureProjector = FindFirstObjectByType<TextureProjector>();
            if (triplanarCache == null) triplanarCache = FindFirstObjectByType<TriplanarCache>();
            if (keyframeStore == null) keyframeStore = FindFirstObjectByType<KeyframeStore>();
            if (persistence == null) persistence = FindFirstObjectByType<RoomScanPersistence>();

            if (depthCapture == null) Debug.LogError("[RoomScan] DepthCapture not found");
            if (volumeIntegrator == null) Debug.LogError("[RoomScan] VolumeIntegrator not found");
            if (chunkManager == null) Debug.LogError("[RoomScan] ChunkManager not found");
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

        private void ProvideColorFrame()
        {
            if (!enableTextureProjection || volumeIntegrator == null) return;
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

                    if (triplanarCache != null)
                    {
                        triplanarCache.DispatchBake(frame, pose.position, pose.rotation,
                            focal, principal, sensor, current,
                            volumeIntegrator.CameraExposure,
                            volumeIntegrator.ExclusionZones);
                    }

                    return;
                }
            }

            volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private void SetupCameraProvider()
        {
            ICameraProvider provider = GetActiveCameraProvider();
            if (textureProjector != null && provider != null)
                textureProjector.SetCameraProvider(provider);
        }

        private ICameraProvider GetActiveCameraProvider()
        {
            if (_customCameraProvider != null) return _customCameraProvider;
            return cameraProvider;
        }

        private void ApplyVisualization()
        {
            // Visualization is applied by iterating chunk renderers
            if (chunkManager == null) return;

            foreach (MeshChunkData chunk in chunkManager.GetPopulatedChunks())
            {
                if (chunk.GameObject == null) continue;
                var renderer = chunk.GameObject.GetComponent<MeshRenderer>();
                if (renderer == null) continue;

                switch (visualization)
                {
                    case ScanVisualization.VertexColored:
                        renderer.enabled = true;
                        break;
                    case ScanVisualization.Hidden:
                        renderer.enabled = false;
                        break;
                    case ScanVisualization.OcclusionOnly:
                        renderer.enabled = true;
                        // Consuming project should assign an occlusion material via ChunkManager
                        break;
                    case ScanVisualization.Wireframe:
                        renderer.enabled = true;
                        break;
                }
            }
        }
    }
}
