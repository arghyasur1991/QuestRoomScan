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
        private bool _started;

        private float IntegrationInterval => 1f / (mode == ScanMode.Guided ? guidedIntegrationHz : passiveIntegrationHz);
        private float MeshInterval => 1f / (mode == ScanMode.Guided ? guidedMeshExtractionHz : passiveMeshExtractionHz);
        private float TextureInterval => 1f / (mode == ScanMode.Guided ? guidedTextureProjectionHz : passiveTextureProjectionHz);

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            ValidateComponents();
            SetupCameraProvider();

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

        private float _lastScannerLog;
        private int _integrateCount;

        private void Update()
        {
            if (!IsScanning || !DepthCapture.DepthAvailable) return;

            float t = Time.time;

            if (t - _lastIntegrationTime >= IntegrationInterval)
            {
                _lastIntegrationTime = t;
                volumeIntegrator.Integrate();
                _integrateCount++;

                if (t - _lastMeshTime >= MeshInterval)
                {
                    _lastMeshTime = t;
                    chunkManager.UpdateDirtyChunks();
                }
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Debug.Log($"[RoomScan] Scanner: integrations={_integrateCount}, mode={mode}, depthAvail={DepthCapture.DepthAvailable}");
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
        }

        public void StartScanning()
        {
            if (IsScanning) return;
            IsScanning = true;

            float t = Time.time;
            _lastIntegrationTime = t;
            _lastMeshTime = t;
            _lastTextureTime = t;

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

            if (depthCapture == null) Debug.LogError("[RoomScan] DepthCapture not found");
            if (volumeIntegrator == null) Debug.LogError("[RoomScan] VolumeIntegrator not found");
            if (chunkManager == null) Debug.LogError("[RoomScan] ChunkManager not found");
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
