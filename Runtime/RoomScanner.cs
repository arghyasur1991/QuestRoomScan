using System;
using System.IO;
using System.Threading.Tasks;
using Genesis.RoomScan.GSplat;
using Genesis.RoomScan.UI;
using UnityEngine;
using UnityEngine.Networking;
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
        public string RefineStatus { get; private set; }
        public string HQRefineStatus { get; private set; }

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
                LastRefinedResult = null;

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

                TextureRefinement.StatusChanged += s => HQRefineStatus = s;
                UnwrappedMeshResult unwrap;
                try { unwrap = await EnsureUnwrappedAsync(); }
                finally { TextureRefinement.StatusChanged -= s => HQRefineStatus = s; }

                HQRefineStatus = "Packaging...";
                string serverUrl = _gsplatServerClient.ServerUrl;
                string keyframeDir = KeyframeDirectory;
                byte[] zipData = await Task.Run(() => PackRefinementZip(keyframeDir, unwrap));

                if (zipData == null || zipData.Length == 0)
                {
                    HQRefineStatus = "No data to upload";
                    return;
                }

                // Phase 1: Upload
                HQRefineStatus = "Uploading...";
                using (var upload = new UnityWebRequest($"{serverUrl}/refine-texture", "POST"))
                {
                    upload.uploadHandler = new UploadHandlerRaw(zipData);
                    upload.downloadHandler = new DownloadHandlerBuffer();
                    upload.SetRequestHeader("Content-Type", "application/octet-stream");
                    upload.timeout = 300;
                    await upload.SendWebRequest();

                    if (upload.result != UnityWebRequest.Result.Success)
                    {
                        HQRefineStatus = $"Upload failed: {upload.error}";
                        Debug.LogError($"[RoomScan] HQ refine upload failed: {upload.error}");
                        return;
                    }
                    Debug.Log($"[RoomScan] HQ refine upload OK: {upload.downloadHandler.text}");
                }

                // Phase 2: Poll until done
                while (true)
                {
                    await Task.Delay(3000);
                    using var poll = UnityWebRequest.Get($"{serverUrl}/refine-texture/status");
                    poll.timeout = 10;
                    await poll.SendWebRequest();

                    if (poll.result != UnityWebRequest.Result.Success)
                    {
                        HQRefineStatus = $"Poll error: {poll.error}";
                        Debug.LogWarning($"[RoomScan] HQ refine poll error: {poll.error}");
                        continue;
                    }

                    var status = JsonUtility.FromJson<HQRefineServerStatus>(poll.downloadHandler.text);
                    if (status.state == "done")
                    {
                        HQRefineStatus = "Downloading result...";
                        break;
                    }
                    if (status.state == "error")
                    {
                        HQRefineStatus = $"Server error: {status.message}";
                        Debug.LogError($"[RoomScan] HQ refine server error: {status.message}");
                        return;
                    }

                    int pct = Mathf.RoundToInt(status.progress * 100f);
                    HQRefineStatus = $"Processing ({pct}%)...";
                }

                // Phase 3: Download result
                using var download = UnityWebRequest.Get($"{serverUrl}/refine-texture/result");
                download.timeout = 120;
                await download.SendWebRequest();

                if (download.result != UnityWebRequest.Result.Success)
                {
                    HQRefineStatus = $"Download failed: {download.error}";
                    Debug.LogError($"[RoomScan] HQ refine download failed: {download.error}");
                    return;
                }

                byte[] pngData = download.downloadHandler.data;
                HQRefineStatus = "Applying atlas...";

                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (ImageConversion.LoadImage(tex, pngData))
                {
                    _hqAtlasTexture = tex;
                    HasHQRefinedTexture = true;
                    EnsureRefinedRenderer();
                    SetRenderMode(ScanRenderMode.HQRefined);

                    if (_persistence != null && _persistence.HasActivePackage)
                    {
                        byte[] rawPixels = tex.GetRawTextureData();
                        await _persistence.SaveArtifactAsync(ArtifactType.HQRefined, rawPixels);
                    }

                    HQRefineStatus = "Done";
                    Debug.Log($"[RoomScan] HQ texture refinement complete: {tex.width}x{tex.height}");
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

        [Serializable]
        private class HQRefineServerStatus
        {
            public string state;
            public float progress;
            public string message;
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

            string kfDir = KeyframeDirectory;
            var opts = XAtlasWrapper.UnwrapOptions.Default;
            opts.MaxCost = xatlasMaxCost;
            opts.BlockAlign = useBlockAlign;
            var unwrap = await TextureRefinement.UnwrapMeshAsync(
                kfDir, KeyframeRelocation, opts, decimationRatio);
            _cachedUnwrap = unwrap;

            EnsureRefinedMesh(unwrap);

            LastRefinedResult = new RefinedTextureResult
            {
                Positions = unwrap.Positions,
                Normals = unwrap.Normals,
                UVs = unwrap.UVs,
                Indices = unwrap.Indices,
                AtlasWidth = unwrap.AtlasWidth,
                AtlasHeight = unwrap.AtlasHeight,
                AtlasPixels = null,
            };

            return unwrap;
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

        private byte[] PackRefinementZip(string keyframeDir, UnwrappedMeshResult mesh)
        {
            string framesPath = Path.Combine(keyframeDir, "frames.jsonl");
            if (!File.Exists(framesPath)) return null;

            using var ms = new MemoryStream();
            using (var archive = new System.IO.Compression.ZipArchive(ms,
                System.IO.Compression.ZipArchiveMode.Create, true))
            {
                var entry = archive.CreateEntry("frames.jsonl");
                using (var es = entry.Open())
                {
                    byte[] data = TextureRefinement.RelocateFramesJsonl(keyframeDir, KeyframeRelocation);
                    if (data == null) return null;
                    es.Write(data, 0, data.Length);
                }

                string imagesDir = Path.Combine(keyframeDir, "images");
                if (Directory.Exists(imagesDir))
                {
                    foreach (string img in Directory.GetFiles(imagesDir, "*.jpg"))
                    {
                        string name = "images/" + Path.GetFileName(img);
                        var imgEntry = archive.CreateEntry(name);
                        using var ies = imgEntry.Open();
                        byte[] imgData = File.ReadAllBytes(img);
                        ies.Write(imgData, 0, imgData.Length);
                    }
                }

                var meshEntry = archive.CreateEntry("refined_mesh.bin");
                using var mes = meshEntry.Open();
                using var bw = new BinaryWriter(mes);
                bw.Write(mesh.Positions.Length);
                bw.Write(mesh.Indices.Length);
                bw.Write(mesh.AtlasWidth);
                bw.Write(mesh.AtlasHeight);
                for (int i = 0; i < mesh.Positions.Length; i++)
                {
                    bw.Write(mesh.Positions[i].x); bw.Write(mesh.Positions[i].y); bw.Write(mesh.Positions[i].z);
                    bw.Write(mesh.Normals[i].x); bw.Write(mesh.Normals[i].y); bw.Write(mesh.Normals[i].z);
                    bw.Write(mesh.UVs[i].x); bw.Write(mesh.UVs[i].y);
                }
                foreach (int idx in mesh.Indices)
                    bw.Write(idx);
            }

            return ms.ToArray();
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
                    _persistence.DeleteArtifactFromPackage(ArtifactType.Refined);
                    HasRefinedTexture = false;
                    LastRefinedResult = null;
                    _cachedUnwrap = null;
                    SetRenderMode(ScanRenderMode.Textured);
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
