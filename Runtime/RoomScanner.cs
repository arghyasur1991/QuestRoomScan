using System;
using System.IO;
using System.Threading.Tasks;
using Genesis.RoomScan.GSplat;
using Genesis.RoomScan.UI;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

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
        Textured,
        Refined,
        HQRefined,
        Splat
    }

    /// <summary>
    /// Top-level orchestrator for room scanning. All sibling components live on
    /// the same GameObject and are resolved automatically via GetComponent.
    /// Input bindings are handled by <see cref="RoomScanInputHandler"/> (optional).
    /// </summary>
    [RequireComponent(typeof(DepthCapture), typeof(VolumeIntegrator), typeof(MeshExtractor))]
    [RequireComponent(typeof(PassthroughCameraProvider), typeof(TriplanarCache), typeof(RoomScanPersistence))]
    [RequireComponent(typeof(KeyframeCollector), typeof(PointCloudExporter), typeof(GSplatManager))]
    [RequireComponent(typeof(GSplatServerClient), typeof(RoomAnchorManager))]
    public class RoomScanner : MonoBehaviour
    {
        public static RoomScanner Instance { get; private set; }

        [Header("Scan Settings")]
        [SerializeField] private ScanMode mode = ScanMode.Passive;
        [SerializeField] private ScanVisualization visualization = ScanVisualization.VertexColored;

        [SerializeField, FormerlySerializedAs("autoStartOnLoad"), Tooltip(
            "When enabled, depth/color integration starts as soon as the scene loads. " +
            "When disabled (default), scanning stays paused until you tap Start Scanning in the debug menu — " +
            "avoids overwriting a restored scan and makes Load Scan easier to test.")]
        private bool startScanningAutomatically = false;

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
        [SerializeField, Tooltip("Auto-save scan data when the application quits")]
        private bool saveOnQuit = false;

        [Header("Texture Refinement")]
        [SerializeField] internal Shader refinedMeshShader;
        [SerializeField] internal ComputeShader atlasBakeCompute;
        [Tooltip("Force CPU bake path instead of GPU compute (for comparison)")]
        [SerializeField] internal bool forceCpuBake = false;
        [Tooltip("Skip denoise pass after baking (GPU bake has fewer speckles)")]
        [SerializeField] internal bool skipDenoise = true;
        [Tooltip("Multi-view blend: 2-pass GPU bake that blends top views per texel for smoother textures")]
        [SerializeField] internal bool multiViewBlend = true;
        [Tooltip("Unsharp mask strength to restore crispness after multi-view blending (0 = off)")]
        [Range(0f, 2f)]
        [SerializeField] internal float sharpenStrength = 0.8f;
        [Tooltip("Server-side atlas super-resolution scale (1 = SR off, just inpaint)")]
        [Range(1, 4)]
        [SerializeField] internal int hqRefineScale = 2;

        [Header("Unwrap Performance")]
        [Tooltip("Simplify mesh before UV unwrap (0.1 = 10% of tris, 1.0 = no simplification). " +
                 "Warning: values below 1.0 cause larger UV triangles which degrade GPU bake " +
                 "performance (warp divergence) and texture alignment. Keep at 1.0 unless testing.")]
        [Range(0.1f, 1f)]
        [SerializeField] internal float decimationRatio = 1f;
        [Tooltip("Align charts to 4x4 blocks for faster packing")]
        [SerializeField] internal bool useBlockAlign = true;
        [Tooltip("Chart growth cost limit — lower = more charts, faster unwrap")]
        [Range(0.5f, 4f)]
        [SerializeField] internal float xatlasMaxCost = 1.5f;

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
        private RoomAnchorManager _roomAnchor;

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
        private bool _started;
        private bool _serverTrainingInProgress;

        // ─────────────────────────────────────────────────────────────
        //  Texture refinement state
        // ─────────────────────────────────────────────────────────────

        private MeshFilter _refinedMeshFilter;
        private MeshRenderer _refinedRenderer;
        private Texture2D _refinedAtlasTexture;
        private Texture2D _hqAtlasTexture;
        private Mesh _refinedMesh;
        private UnwrappedMeshResult? _cachedUnwrap;

        public bool HasRefinedTexture { get; private set; }
        public bool HasHQRefinedTexture { get; private set; }
        public bool IsRefining { get; private set; }
        public bool IsHQRefining { get; private set; }
        public bool IsMeshEnhancing { get; private set; }
        public bool HasEnhancedMesh { get; internal set; }
        public string RefineStatus { get; private set; }
        public string HQRefineStatus { get; private set; }
        public string MeshEnhanceStatus { get; private set; }

        /// <summary>
        /// Shared UV mesh data used by persistence to save/restore refinement results.
        /// </summary>
        internal RefinedTextureResult? LastRefinedResult { get; set; }

        /// <summary>
        /// Relocation matrix from the last scan load. Transforms old-session world-space
        /// poses to current world-space. Identity when no relocation occurred or when
        /// running a live (non-reloaded) scan.
        /// </summary>
        internal Matrix4x4 KeyframeRelocation { get; set; } = Matrix4x4.identity;

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

            if (_roomAnchor != null && _roomAnchor.enabled)
            {
                if (_roomAnchor.IsRoomLoaded)
                    CompleteRoomStartup();
                else
                    _roomAnchor.RoomReady += OnRoomAnchorReady;
            }
            else
                CompleteRoomStartup();
        }

        private void OnRoomAnchorReady()
        {
            if (_roomAnchor != null)
                _roomAnchor.RoomReady -= OnRoomAnchorReady;
            if (_started)
                return;
            CompleteRoomStartup();
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
            _roomAnchor = GetComponent<RoomAnchorManager>();
        }

        /// <summary>
        /// Finishes startup after MRUK room is ready (or immediately if <see cref="RoomAnchorManager"/> is disabled).
        /// </summary>
        private void CompleteRoomStartup()
        {
            if (_started)
                return;

            if (startScanningAutomatically)
                StartScanning();

            _started = true;
            Debug.Log(startScanningAutomatically
                ? "[RoomScan] Room ready, scanning started automatically"
                : "[RoomScan] Room ready, scanning paused — use debug menu Start Scanning");
        }

        private void OnEnable()
        {
            if (_started && startScanningAutomatically && !IsScanning)
                StartScanning();
        }

        private void OnDisable()
        {
            StopScanning();
        }

        private async void OnApplicationQuit()
        {
            if (!saveOnQuit || _persistence == null || !_started) return;
            if (_volumeIntegrator == null
                || _volumeIntegrator.IntegrationCount <= _volumeIntegrator.WarmupIntegrations) return;
            if (_persistence.IsSaving) return;

            Debug.Log("[RoomScan] App quitting, saving scan...");
            await _persistence.SaveToNewPackageAsync();
        }

        private float _lastScannerLog;
        private int _integrateCount;

        private void Update()
        {
            if (_clearDone)
            {
                _clearDone = false;
                _clearInProgress = false;

                // Re-create GPU mesh pipeline (deferred from ClearAllDataAsync to
                // give the GPU a frame to finish using the old buffers).
                if (_meshExtractor != null && !_meshExtractor.IsInitialized)
                    _meshExtractor.Reinitialize();

                if (_keyframeCollector != null)
                    _keyframeCollector.ReinitExportDir();
                Debug.Log("[RoomScan] All scan + export data cleared");
                if (startScanningAutomatically)
                    StartScanning();
                _clearDoneCallback?.Invoke();
                _clearDoneCallback = null;
            }

            if (_reinitExportPending)
            {
                _reinitExportPending = false;
                if (_keyframeCollector != null)
                    _keyframeCollector.ReinitExportDir();
            }

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
        }

        // ═════════════════════════════════════════════════════════════
        //  PUBLIC API — call from any client, input handler, or UI
        // ═════════════════════════════════════════════════════════════

        public void StartScanning()
        {
            if (IsScanning) return;
            IsScanning = true;
            KeyframeRelocation = Matrix4x4.identity;
            _cachedUnwrap = null;
            if (_persistence != null) _persistence.ClearActivePackage();

            if (_keyframeCollector != null)
            {
                _keyframeCollector.ClearInMemory();
                if (_pointCloudExporter != null)
                    _pointCloudExporter.ResetTimer();
                string gsExportDir = Path.Combine(Application.persistentDataPath, "GSExport");
                System.Threading.ThreadPool.QueueUserWorkItem(_ =>
                {
                    try
                    {
                        if (Directory.Exists(gsExportDir))
                            Directory.Delete(gsExportDir, true);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"[RoomScan] StartScanning clear error: {e.Message}");
                    }
                    finally
                    {
                        _reinitExportPending = true;
                    }
                });
            }

            float t = Time.time;
            _lastIntegrationTime = t;
            _lastMeshTime = t;

            ICameraProvider provider = GetActiveCameraProvider();
            provider?.StartCapture();

            Debug.Log($"[RoomScan] StartScanning — integrationCount={_volumeIntegrator.IntegrationCount}");
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
            if (_triplanarCache != null)
                _triplanarCache.Clear();
        }

        /// <summary>
        /// Clears all persisted data: in-memory scan, saved scan files, triplanar
        /// textures, and GSExport (keyframes + point cloud). Safe to call at runtime.
        /// File I/O runs on a background thread via ThreadPool to avoid main-thread
        /// stalls and potential SynchronizationContext deadlocks on Quest/IL2CPP.
        /// GPU resources are disposed without immediate re-allocation to avoid
        /// Vulkan stalls when the GPU is still referencing the previous frame's buffers.
        /// Re-initialization happens lazily on the next <see cref="StartScanning"/> or load.
        /// </summary>
        public void ClearAllDataAsync(Action onComplete = null)
        {
            if (_clearInProgress) return;
            _clearInProgress = true;

            string gsExportDir = Path.Combine(Application.persistentDataPath, "GSExport");

            try
            {
                StopScanning();

                _gsplatManager?.ClearSplat();
                _gsplatManager?.ResetSplatTransform();
                _downloadedPlyData = null;
                HasRefinedTexture = false;
                HasHQRefinedTexture = false;
                HasEnhancedMesh = false;
                LastRefinedResult = null;
                _cachedUnwrap = null;
                _refinedMesh = null;

                _meshExtractor.DisposeOnly();
                _volumeIntegrator.Clear();
                if (_triplanarCache != null)
                    _triplanarCache.Clear();

                if (_keyframeCollector != null)
                    _keyframeCollector.ClearInMemory();

                if (_persistence != null) _persistence.ClearActivePackage();
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] ClearAllData sync error: {e.Message}\n{e.StackTrace}");
                _clearInProgress = false;
                return;
            }

            System.Threading.ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    if (Directory.Exists(gsExportDir))
                        Directory.Delete(gsExportDir, true);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[RoomScan] ClearAllData I/O error: {e.Message}");
                }
                finally
                {
                    _clearDoneCallback = onComplete;
                    _clearDone = true;
                }
            });
        }

        private volatile bool _clearInProgress;
        private volatile bool _clearDone;
        private volatile bool _reinitExportPending;
        private Action _clearDoneCallback;

        public void SetRenderMode(ScanRenderMode newMode)
        {
            renderMode = newMode;
            ApplyRenderMode();
            RenderModeChanged?.Invoke(renderMode);
            Debug.Log($"[RoomScan] Render mode: {renderMode}");
        }

        public void CycleRenderMode()
        {
            // Ordered cycle; skip modes whose data isn't available yet
            ScanRenderMode[] order =
            {
                ScanRenderMode.Mesh, ScanRenderMode.Textured,
                ScanRenderMode.Refined, ScanRenderMode.HQRefined,
                ScanRenderMode.Splat
            };

            int cur = Array.IndexOf(order, renderMode);
            for (int i = 1; i < order.Length; i++)
            {
                var candidate = order[(cur + i) % order.Length];
                if (candidate == ScanRenderMode.Refined && !HasRefinedTexture) continue;
                if (candidate == ScanRenderMode.HQRefined && !HasHQRefinedTexture) continue;

                if (candidate == ScanRenderMode.Splat && HasDownloadedSplat && !_gsplatManager.HasServerTrainedSplats)
                {
                    LoadDownloadedSplat();
                    return;
                }

                SetRenderMode(candidate);
                return;
            }
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

        public async Task<bool> SaveScanAsync()
        {
            if (_persistence == null) return false;
            return await _persistence.SaveToNewPackageAsync();
        }

        public async Task<bool> LoadPackageAsync(string pkgId)
        {
            if (_persistence == null) return false;
            return await _persistence.LoadPackageAsync(pkgId);
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
        //  Texture Refinement
        // ─────────────────────────────────────────────────────────────

        public async void StartTextureRefinement()
        {
            if (IsRefining) return;
            IsRefining = true;
            RefineStatus = "Starting...";

            TextureRefinement.StatusChanged += s => RefineStatus = s;
            TextureRefinement.EnableMultiViewBlend = multiViewBlend;
            TextureRefinement.SharpenStrength = sharpenStrength;
            try
            {
                string keyframeDir = KeyframeDirectory;
                var unwrap = await EnsureUnwrappedAsync();
                byte[] atlasPixels = await TextureRefinement.BakeAtlasAsync(
                    unwrap, keyframeDir, KeyframeRelocation,
                    forceCpuBake ? null : atlasBakeCompute,
                    skipDenoise);

                var result = new RefinedTextureResult
                {
                    Positions = unwrap.Positions,
                    Normals = unwrap.Normals,
                    UVs = unwrap.UVs,
                    Indices = unwrap.Indices,
                    AtlasPixels = atlasPixels,
                    AtlasWidth = unwrap.AtlasWidth,
                    AtlasHeight = unwrap.AtlasHeight
                };

                ApplyRefinedAtlas(result);
                LastRefinedResult = result;
                HasRefinedTexture = true;
                SetRenderMode(ScanRenderMode.Refined);

                if (_persistence != null && _persistence.HasActivePackage)
                    await _persistence.SaveArtifactAsync(ArtifactType.Refined, null, result);

                Debug.Log("[RoomScan] On-device texture refinement complete");
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] Texture refinement failed: {e.Message}\n{e.StackTrace}");
                RefineStatus = "Failed";
            }
            finally
            {
                TextureRefinement.StatusChanged -= s => RefineStatus = s;
                IsRefining = false;
            }
        }

        public async void StartHQRefinement()
        {
            if (IsHQRefining) return;
            IsHQRefining = true;
            HQRefineStatus = "Starting...";

            try
            {
                if (_gsplatServerClient == null)
                {
                    HQRefineStatus = "No server configured";
                    return;
                }

                // Auto-trigger on-device refinement if not done yet
                if (!LastRefinedResult.HasValue)
                {
                    HQRefineStatus = "Running on-device refine first...";
                    TextureRefinement.StatusChanged += s => HQRefineStatus = s;
                    try
                    {
                        StartTextureRefinement();
                        while (IsRefining) await Task.Yield();
                    }
                    finally { TextureRefinement.StatusChanged -= s => HQRefineStatus = s; }

                    if (!LastRefinedResult.HasValue)
                    {
                        HQRefineStatus = "On-device refine failed";
                        return;
                    }
                }

                // Encode current refined atlas to PNG
                HQRefineStatus = "Encoding atlas...";
                byte[] pngBytes;
                var r = LastRefinedResult.Value;
                if (r.AtlasPixels != null)
                {
                    var srcTex = new Texture2D(r.AtlasWidth, r.AtlasHeight, TextureFormat.RGBA32, false);
                    srcTex.SetPixelData(r.AtlasPixels, 0);
                    srcTex.Apply();
                    pngBytes = ImageConversion.EncodeToPNG(srcTex);
                    UnityEngine.Object.Destroy(srcTex);
                }
                else if (_refinedAtlasTexture != null)
                {
                    pngBytes = ImageConversion.EncodeToPNG(_refinedAtlasTexture);
                }
                else
                {
                    HQRefineStatus = "No atlas data available";
                    return;
                }

                Debug.Log($"[RoomScan] HQ refine: uploading {pngBytes.Length / 1024}KB atlas, scale={hqRefineScale}");
                HQRefineStatus = $"Uploading ({pngBytes.Length / 1024}KB)...";

                byte[] resultPng = await _gsplatServerClient.EnhanceAtlasAsync(pngBytes, hqRefineScale, inpaint: true);

                if (resultPng == null || resultPng.Length == 0)
                {
                    HQRefineStatus = "Server returned no data";
                    return;
                }

                HQRefineStatus = "Applying atlas...";
                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (ImageConversion.LoadImage(tex, resultPng))
                {
                    _hqAtlasTexture = tex;
                    HasHQRefinedTexture = true;
                    EnsureRefinedRenderer();
                    SetRenderMode(ScanRenderMode.HQRefined);

                    if (_persistence != null && _persistence.HasActivePackage)
                        await _persistence.SaveArtifactAsync(ArtifactType.HQRefined, resultPng);

                    HQRefineStatus = "Done";
                    Debug.Log($"[RoomScan] HQ atlas enhancement complete: {tex.width}x{tex.height}");
                }
                else
                {
                    UnityEngine.Object.Destroy(tex);
                    HQRefineStatus = "Failed to decode atlas";
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] HQ refinement failed: {e.Message}\n{e.StackTrace}");
                HQRefineStatus = "Failed";
            }
            finally
            {
                IsHQRefining = false;
            }
        }

        public async void StartMeshEnhancement()
        {
            if (IsMeshEnhancing) return;
            IsMeshEnhancing = true;
            MeshEnhanceStatus = "Starting...";

            try
            {
                if (_gsplatServerClient == null)
                {
                    MeshEnhanceStatus = "No server configured";
                    return;
                }

                if (!LastRefinedResult.HasValue)
                {
                    MeshEnhanceStatus = "No refined mesh — refine first";
                    return;
                }

                MeshEnhanceStatus = "Serializing mesh...";
                var r = LastRefinedResult.Value;
                byte[] meshBin = await Task.Run(() => SerializeRefinedMesh(r));
                Debug.Log($"[RoomScan] Mesh enhance: uploading {meshBin.Length / 1024}KB ({r.Positions.Length} verts)");

                MeshEnhanceStatus = $"Uploading ({meshBin.Length / 1024}KB)...";
                byte[] resultBin = await _gsplatServerClient.EnhanceMeshAsync(
                    meshBin, smoothIterations: 3, enablePlaneSnap: false);

                if (resultBin == null || resultBin.Length == 0)
                {
                    MeshEnhanceStatus = "Server returned no data";
                    return;
                }

                MeshEnhanceStatus = "Applying enhanced mesh...";
                var enhanced = await Task.Run(() => DeserializeRefinedMesh(resultBin));
                enhanced.AtlasPixels = r.AtlasPixels;

                ApplyEnhancedMesh(enhanced);
                HasEnhancedMesh = true;

                if (_persistence != null && _persistence.HasActivePackage)
                    await _persistence.SaveArtifactAsync(ArtifactType.EnhancedMesh, null, enhanced);

                MeshEnhanceStatus = "Done";
                Debug.Log($"[RoomScan] Mesh enhancement complete: {enhanced.Positions.Length} verts");
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] Mesh enhancement failed: {e.Message}\n{e.StackTrace}");
                MeshEnhanceStatus = "Failed";
            }
            finally
            {
                IsMeshEnhancing = false;
            }
        }

        private static byte[] SerializeRefinedMesh(RefinedTextureResult r)
        {
            int vertCount = r.Positions.Length;
            int idxCount = r.Indices.Length;
            int size = 24 + vertCount * 32 + idxCount * 4;

            using var ms = new MemoryStream(size);
            using var w = new BinaryWriter(ms);

            w.Write((uint)0x46524D52); // RMRF magic
            w.Write(1);                // version
            w.Write(vertCount);
            w.Write(idxCount);
            w.Write(r.AtlasWidth);
            w.Write(r.AtlasHeight);

            for (int i = 0; i < vertCount; i++)
            {
                w.Write(r.Positions[i].x); w.Write(r.Positions[i].y); w.Write(r.Positions[i].z);
                w.Write(r.Normals[i].x); w.Write(r.Normals[i].y); w.Write(r.Normals[i].z);
                w.Write(r.UVs[i].x); w.Write(r.UVs[i].y);
            }
            foreach (int idx in r.Indices) w.Write(idx);

            return ms.ToArray();
        }

        private static RefinedTextureResult DeserializeRefinedMesh(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var r = new BinaryReader(ms);

            uint magic = r.ReadUInt32();
            if (magic == 0x46524D52) r.ReadInt32(); // skip version
            else ms.Position = 4; // no header, rewind past magic (treat as vertCount)

            int vertCount = magic == 0x46524D52 ? r.ReadInt32() : (int)magic;
            int idxCount = r.ReadInt32();
            int atlasW = r.ReadInt32();
            int atlasH = r.ReadInt32();

            var positions = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];
            for (int i = 0; i < vertCount; i++)
            {
                positions[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                normals[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                uvs[i] = new Vector2(r.ReadSingle(), r.ReadSingle());
            }

            int[] indices = new int[idxCount];
            for (int i = 0; i < idxCount; i++) indices[i] = r.ReadInt32();

            return new RefinedTextureResult
            {
                Positions = positions, Normals = normals, UVs = uvs,
                Indices = indices, AtlasWidth = atlasW, AtlasHeight = atlasH,
            };
        }

        private void ApplyEnhancedMesh(RefinedTextureResult enhanced)
        {
            if (_refinedMesh == null)
            {
                _refinedMesh = new Mesh { name = "EnhancedScanMesh", indexFormat = IndexFormat.UInt32 };
            }

            _refinedMesh.Clear();
            _refinedMesh.SetVertices(enhanced.Positions);
            _refinedMesh.SetNormals(enhanced.Normals);
            _refinedMesh.SetUVs(0, enhanced.UVs);
            _refinedMesh.SetTriangles(enhanced.Indices, 0);

            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = _refinedMesh;

            if (_refinedAtlasTexture != null)
                _refinedRenderer.material.mainTexture = _refinedAtlasTexture;
        }

        private void ApplyRefinedAtlas(RefinedTextureResult result)
        {
            _refinedAtlasTexture = new Texture2D(result.AtlasWidth, result.AtlasHeight,
                TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear };
            _refinedAtlasTexture.SetPixelData(result.AtlasPixels, 0);
            _refinedAtlasTexture.Apply();

            Debug.Log($"[RoomScan] Refined atlas applied: " +
                $"{result.Positions.Length} verts, {result.Indices.Length / 3} tris, " +
                $"atlas {result.AtlasWidth}x{result.AtlasHeight}");

            EnsureRefinedRenderer();
            _refinedRenderer.material.mainTexture = _refinedAtlasTexture;
        }

        /// <summary>
        /// Applies pre-loaded atlas and mesh data (called from persistence load).
        /// </summary>
        internal void ApplyRefinedTexture(Texture2D atlas, Mesh mesh)
        {
            _refinedAtlasTexture = atlas;
            _refinedMesh = mesh;
            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = mesh;
            _refinedRenderer.material.mainTexture = atlas;
            HasRefinedTexture = true;
        }

        internal void ApplyHQTexture(Texture2D atlas)
        {
            _hqAtlasTexture = atlas;
            EnsureRefinedRenderer();
            HasHQRefinedTexture = true;
        }

        /// <summary>
        /// Returns the cached UV-unwrapped mesh, or runs the unwrap if not yet done.
        /// Shared by both on-device refine and HQ refine paths.
        /// </summary>
        private async Task<UnwrappedMeshResult> EnsureUnwrappedAsync()
        {
            if (_cachedUnwrap.HasValue)
            {
                Debug.Log("[RoomScan] Reusing cached UV-unwrapped mesh");
                return _cachedUnwrap.Value;
            }

            // Reconstruct from persisted refined mesh if available (skip xatlas)
            if (LastRefinedResult.HasValue)
            {
                var r = LastRefinedResult.Value;
                var unwrap = ReconstructUnwrapFromResult(r);
                _cachedUnwrap = unwrap;
                EnsureRefinedMesh(unwrap);
                Debug.Log("[RoomScan] Reconstructed UV mesh from persisted refined_mesh.bin (no xatlas needed)");
                return unwrap;
            }

            string kfDir = KeyframeDirectory;
            var opts = XAtlasWrapper.UnwrapOptions.Default;
            opts.MaxCost = xatlasMaxCost;
            opts.BlockAlign = useBlockAlign;
            var unwrap2 = await TextureRefinement.UnwrapMeshAsync(
                kfDir, KeyframeRelocation, opts, decimationRatio);
            _cachedUnwrap = unwrap2;

            EnsureRefinedMesh(unwrap2);

            LastRefinedResult = new RefinedTextureResult
            {
                Positions = unwrap2.Positions,
                Normals = unwrap2.Normals,
                UVs = unwrap2.UVs,
                Indices = unwrap2.Indices,
                AtlasWidth = unwrap2.AtlasWidth,
                AtlasHeight = unwrap2.AtlasHeight,
                AtlasPixels = null,
            };

            return unwrap2;
        }

        /// <summary>
        /// Rebuilds an UnwrappedMeshResult from a persisted RefinedTextureResult.
        /// RawUVs are reconstructed from normalized UVs * atlas dimensions.
        /// OrigPositions/OrigIndices use the unwrapped mesh as fallback (sufficient for depth buffer).
        /// </summary>
        private static UnwrappedMeshResult ReconstructUnwrapFromResult(RefinedTextureResult r)
        {
            int vertCount = r.Positions.Length;
            float[] rawUVs = new float[vertCount * 2];
            for (int i = 0; i < vertCount; i++)
            {
                rawUVs[i * 2] = r.UVs[i].x * r.AtlasWidth;
                rawUVs[i * 2 + 1] = r.UVs[i].y * r.AtlasHeight;
            }

            return new UnwrappedMeshResult
            {
                Positions = r.Positions,
                Normals = r.Normals,
                UVs = r.UVs,
                RawUVs = rawUVs,
                Indices = r.Indices,
                AtlasWidth = r.AtlasWidth,
                AtlasHeight = r.AtlasHeight,
                OrigPositions = r.Positions,
                OrigNormals = r.Normals,
                OrigIndices = r.Indices,
            };
        }

        private void EnsureRefinedMesh(UnwrappedMeshResult unwrap)
        {
            if (_refinedMesh != null) return;
            _refinedMesh = new Mesh { name = "RefinedScanMesh", indexFormat = IndexFormat.UInt32 };
            _refinedMesh.SetVertices(unwrap.Positions);
            _refinedMesh.SetNormals(unwrap.Normals);
            _refinedMesh.SetUVs(0, unwrap.UVs);
            _refinedMesh.SetTriangles(unwrap.Indices, 0);
            EnsureRefinedRenderer();
            _refinedMeshFilter.mesh = _refinedMesh;
        }

        private void EnsureRefinedRenderer()
        {
            if (_refinedRenderer != null) return;

            var go = new GameObject("RefinedMeshRenderer");
            go.transform.SetParent(transform, false);
            _refinedMeshFilter = go.AddComponent<MeshFilter>();
            _refinedRenderer = go.AddComponent<MeshRenderer>();

            var shader = refinedMeshShader;
            if (shader == null) shader = Shader.Find("Genesis/RefinedMesh");
            if (shader == null)
            {
                Debug.LogWarning("[RoomScan] Genesis/RefinedMesh shader not found, using URP/Unlit");
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            }
            _refinedRenderer.material = new Material(shader);
            _refinedRenderer.enabled = false;
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
            if (_depthCapture != null)
                pose = _depthCapture.TrackingToWorld(pose);
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
                    _depthCapture?.SetRGBGuide(frame);

                    Pose pose = pcp.CameraPose;
                    if (_depthCapture != null)
                        pose = _depthCapture.TrackingToWorld(pose);
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

        private static readonly int NoFreezeTintID = Shader.PropertyToID("_RSNoFreezeTint");

        private void ApplyRenderMode()
        {
            var gpuRenderer = _meshExtractor != null ? _meshExtractor.GetComponent<GPUMeshRenderer>() : null;

            bool gpuMeshVisible = renderMode == ScanRenderMode.Mesh || renderMode == ScanRenderMode.Textured;
            bool refinedVisible = renderMode == ScanRenderMode.Refined || renderMode == ScanRenderMode.HQRefined;

            if (gpuRenderer != null)
                gpuRenderer.RenderVisible = gpuMeshVisible;
            if (_gsplatManager != null)
                _gsplatManager.RenderVisible = renderMode == ScanRenderMode.Splat;

            // Show/hide the UV-mapped refined mesh renderer
            if (_refinedRenderer != null)
            {
                _refinedRenderer.enabled = refinedVisible;
                if (refinedVisible)
                {
                    var tex = renderMode == ScanRenderMode.HQRefined && _hqAtlasTexture != null
                        ? _hqAtlasTexture
                        : _refinedAtlasTexture;
                    if (tex != null)
                        _refinedRenderer.material.mainTexture = tex;
                }
            }

            Shader.SetGlobalFloat(NoFreezeTintID, renderMode == ScanRenderMode.Textured ? 1f : 0f);
        }

        /// <summary>
        /// Whether a trained splat has been downloaded and is ready to load/render.
        /// </summary>
        public bool HasDownloadedSplat => _downloadedPlyData != null && _downloadedPlyData.Length > 0;

        private byte[] _downloadedPlyData;

        /// <summary>
        /// Raw PLY bytes from server training or loaded from disk.
        /// Used by persistence to save/restore Gaussian Splat data.
        /// </summary>
        public byte[] DownloadedPlyData
        {
            get => _downloadedPlyData;
            set => _downloadedPlyData = value;
        }

        /// <summary>
        /// Keyframe directory. Uses active package's keyframes/ if loaded, else GSExport/.
        /// </summary>
        public string KeyframeDirectory
        {
            get
            {
                if (_persistence != null && _persistence.HasActivePackage)
                {
                    string pkgKf = Path.Combine(_persistence.ActivePackageDirectory, "keyframes");
                    if (Directory.Exists(pkgKf)) return pkgKf;
                }
                return Path.Combine(Application.persistentDataPath, "GSExport");
            }
        }

        /// <summary>
        /// Deletes the artifact matching the current render mode from the active package.
        /// </summary>
        public void DeleteActiveArtifact()
        {
            if (_persistence == null || !_persistence.HasActivePackage) return;

            switch (renderMode)
            {
                case ScanRenderMode.Splat:
                    _persistence.DeleteArtifactFromPackage(ArtifactType.Splat);
                    _gsplatManager?.ClearSplat();
                    _downloadedPlyData = null;
                    SetRenderMode(ScanRenderMode.Mesh);
                    break;

                case ScanRenderMode.Refined:
                    if (HasEnhancedMesh)
                    {
                        DeleteEnhancedMesh();
                    }
                    else
                    {
                        _persistence.DeleteArtifactFromPackage(ArtifactType.Refined);
                        HasRefinedTexture = false;
                        LastRefinedResult = null;
                        _cachedUnwrap = null;
                        _refinedMesh = null;
                        SetRenderMode(ScanRenderMode.Textured);
                    }
                    break;

                case ScanRenderMode.HQRefined:
                    _persistence.DeleteArtifactFromPackage(ArtifactType.HQRefined);
                    HasHQRefinedTexture = false;
                    _hqAtlasTexture = null;
                    SetRenderMode(HasRefinedTexture ? ScanRenderMode.Refined : ScanRenderMode.Textured);
                    break;
            }
        }

        /// <summary>
        /// Removes the enhanced mesh overlay and restores the original refined mesh geometry.
        /// </summary>
        public void DeleteEnhancedMesh()
        {
            if (!HasEnhancedMesh) return;

            _persistence?.DeleteArtifactFromPackage(ArtifactType.EnhancedMesh);
            HasEnhancedMesh = false;

            if (LastRefinedResult.HasValue)
            {
                var r = LastRefinedResult.Value;
                if (_refinedMesh != null)
                {
                    _refinedMesh.Clear();
                    _refinedMesh.SetVertices(r.Positions);
                    _refinedMesh.SetNormals(r.Normals);
                    _refinedMesh.SetUVs(0, r.UVs);
                    _refinedMesh.SetTriangles(r.Indices, 0);
                    EnsureRefinedRenderer();
                    _refinedMeshFilter.mesh = _refinedMesh;
                }
            }

            Debug.Log("[RoomScan] Enhanced mesh deleted, original restored");
        }

        /// <summary>
        /// Loads the downloaded splat into GPU buffers and switches to splat render mode.
        /// Call after training + download completes.
        /// </summary>
        public void LoadDownloadedSplat()
        {
            if (_downloadedPlyData == null || _downloadedPlyData.Length == 0)
            {
                Debug.LogWarning("[RoomScan] No downloaded splat data to load");
                return;
            }
            _gsplatManager.LoadTrainedPly(_downloadedPlyData);
            SetRenderMode(ScanRenderMode.Splat);
            Debug.Log("[RoomScan] Trained Gaussians loaded and rendering");
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
                bool uploaded = await _gsplatServerClient.UploadTrainingData(KeyframeRelocation);
                if (!uploaded) { Debug.LogError("[RoomScan] Upload failed"); return; }

                Debug.Log("[RoomScan] Waiting for server training to complete...");
                bool success = await _gsplatServerClient.PollUntilDone();
                if (!success) { Debug.LogError("[RoomScan] Server training failed"); return; }

                Debug.Log("[RoomScan] Downloading trained Gaussians...");
                byte[] plyData = await _gsplatServerClient.DownloadResult();
                if (plyData == null || plyData.Length == 0) { Debug.LogError("[RoomScan] Download empty"); return; }

                _downloadedPlyData = plyData;
                Debug.Log($"[RoomScan] Trained splat downloaded ({plyData.Length / (1024 * 1024f):F1}MB)");

                if (_persistence != null && _persistence.HasActivePackage)
                    await _persistence.SaveArtifactAsync(ArtifactType.Splat, plyData);

                LoadDownloadedSplat();
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

            gpuRenderer.RenderVisible = visualization != ScanVisualization.Hidden;
        }
    }
}
