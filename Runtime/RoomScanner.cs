using System;
using Genesis.RoomScan.GSplat;
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
        [SerializeField] private RoomScanPersistence persistence;
        [SerializeField] private KeyframeCollector keyframeCollector;
        [SerializeField] private PointCloudExporter pointCloudExporter;
        [SerializeField] private GSplatManager gsplatManager;
        [SerializeField] private GSplat.GSplatServerClient gsplatServerClient;

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

        [Header("Render Mode")]
        [SerializeField] private ScanRenderMode renderMode = ScanRenderMode.Mesh;

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

        private void Start()
        {
            ValidateComponents();
            SetupHeadExclusion();

            OnRoomReady();
        }

        private async void OnRoomReady()
        {
            if (persistence != null && persistence.HasSavedScan())
            {
                Debug.Log("[RoomScan] Found saved scan, loading...");
                await persistence.LoadAsync();
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

            PollFreezeInput();
            PollTrainingTrigger();

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
            if (persistence == null) persistence = FindFirstObjectByType<RoomScanPersistence>();
            if (keyframeCollector == null) keyframeCollector = FindFirstObjectByType<KeyframeCollector>();
            if (pointCloudExporter == null) pointCloudExporter = FindFirstObjectByType<PointCloudExporter>();
            if (gsplatManager == null) gsplatManager = FindFirstObjectByType<GSplatManager>();
            if (gsplatServerClient == null) gsplatServerClient = FindFirstObjectByType<GSplat.GSplatServerClient>();

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
            Shader.SetGlobalFloat(Shader.PropertyToID("_RSTriAvailable"), 0f);
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
                            $"triCache={triplanarCache != null}");

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

        private bool _serverTrainingInProgress;

        private void PollTrainingTrigger()
        {
            // Right controller B button = cycle render mode
            if (OVRInput.GetDown(OVRInput.Button.Two))
            {
                renderMode = renderMode switch
                {
                    ScanRenderMode.Mesh => ScanRenderMode.Splat,
                    ScanRenderMode.Splat => ScanRenderMode.Both,
                    _ => ScanRenderMode.Mesh,
                };
                ApplyRenderMode();
                Debug.Log($"[RoomScan] Render mode: {renderMode}");
            }

            if (_serverTrainingInProgress) return;
            if (gsplatServerClient == null || pointCloudExporter == null) return;

            // Right controller A button = start server-side GS training
            if (OVRInput.GetDown(OVRInput.Button.One))
            {
                Debug.Log("[RoomScan] GS training trigger pressed (A button)");
                RunServerTrainingAsync();
            }
        }

        private void ApplyRenderMode()
        {
            var gpuRenderer = meshExtractor?.GetComponent<GPUMeshRenderer>();
            var splatRend = gsplatManager != null
                ? gsplatManager.GetComponent<GSplat.GSRenderer>()
                  ?? FindFirstObjectByType<GSplat.GSRenderer>()
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

                // Step 1: Export latest point cloud
                Debug.Log("[RoomScan] Exporting point cloud...");
                await pointCloudExporter.ExportAsync();

                // Step 2: Upload to server
                Debug.Log("[RoomScan] Uploading training data to PC server...");
                bool uploaded = await gsplatServerClient.UploadTrainingData();
                if (!uploaded)
                {
                    Debug.LogError("[RoomScan] Upload failed, aborting training");
                    return;
                }

                // Step 3: Poll until done
                Debug.Log("[RoomScan] Waiting for server training to complete...");
                bool success = await gsplatServerClient.PollUntilDone();
                if (!success)
                {
                    Debug.LogError("[RoomScan] Server training failed or was cancelled");
                    return;
                }

                // Step 4: Download trained PLY
                Debug.Log("[RoomScan] Downloading trained Gaussians...");
                byte[] plyData = await gsplatServerClient.DownloadResult();
                if (plyData == null || plyData.Length == 0)
                {
                    Debug.LogError("[RoomScan] Download returned no data");
                    return;
                }

                // Step 5: Load into renderer
                if (gsplatManager != null)
                {
                    gsplatManager.LoadTrainedPly(plyData);
                    Debug.Log("[RoomScan] Trained Gaussians loaded and ready for rendering");
                }
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

        private void PollFreezeInput()
        {
            if (volumeIntegrator == null) return;

            bool freeze = OVRInput.GetDown(OVRInput.Button.Three);   // X on left controller
            bool unfreeze = OVRInput.GetDown(OVRInput.Button.Four);  // Y on left controller
            if (!freeze && !unfreeze) return;

            ICameraProvider provider = GetActiveCameraProvider();
            if (provider is not PassthroughCameraProvider pcp || !pcp.IsReady) return;

            Pose pose = pcp.CameraPose;
            Vector2 focal = pcp.FocalLength;
            Vector2 principal = pcp.PrincipalPoint;
            Vector2 sensor = pcp.SensorResolution;
            Vector2 current = pcp.CurrentResolution;

            if (freeze)
                volumeIntegrator.FreezeInView(pose.position, pose.rotation,
                    focal, principal, sensor, current);
            else
                volumeIntegrator.UnfreezeInView(pose.position, pose.rotation,
                    focal, principal, sensor, current);
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
        }
    }
}
