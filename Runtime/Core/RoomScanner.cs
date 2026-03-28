using System;
using System.IO;
using System.Threading.Tasks;
using Genesis.RoomScan.UI;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Which representation of the scan to display. <see cref="RoomScanner.CycleRenderMode"/>
    /// automatically skips modes whose backing data or module is not present.
    /// </summary>
    public enum ScanRenderMode
    {
        /// <summary>Live GPU mesh with vertex colors only (triplanar forced off).</summary>
        Vertex,
        /// <summary>Live GPU mesh with triplanar-projected camera textures (falls back to vertex colors where data is missing).</summary>
        Triplanar,
        /// <summary>UV-unwrapped mesh with on-device baked atlas.</summary>
        Refined,
        /// <summary>UV-unwrapped mesh with server-enhanced high-resolution atlas.</summary>
        HQRefined,
        /// <summary>Gaussian Splat point cloud rendered from server-trained PLY data.</summary>
        Splat,
        /// <summary>All scan rendering disabled.</summary>
        None,
        /// <summary>Live GPU mesh wireframe via barycentric edge detection.</summary>
        Wireframe
    }

    /// <summary>
    /// Top-level orchestrator for room scanning. All sibling components live on
    /// the same GameObject and are resolved automatically via GetComponent.
    /// Input bindings are handled by <see cref="RoomScanInputHandler"/> (optional).
    /// </summary>
    [RequireComponent(typeof(DepthCapture), typeof(VolumeIntegrator), typeof(MeshExtractor))]
    [RequireComponent(typeof(RoomScanPersistence), typeof(RoomAnchorManager))]
    public class RoomScanner : MonoBehaviour
    {
        public static RoomScanner Instance { get; private set; }

        [Header("Scan Rates")]
        [SerializeField] private float integrationHz = 30f;
        [SerializeField] private float meshExtractionHz = 30f;

        [Header("Render Mode")]
        [SerializeField] private ScanRenderMode renderMode = ScanRenderMode.Vertex;

        [SerializeField, Range(0.2f, 5f), Tooltip("Wireframe line thickness multiplier")]
        private float wireThickness = 1.5f;

        [SerializeField, Tooltip("Show blue tint overlay on frozen voxels (Vertex/Triplanar/Wireframe modes)")]
        private bool showFreezeTint = true;

        [Header("Logging")]
        [SerializeField] private LogLevel logLevel = LogLevel.Info;

        // ─────────────────────────────────────────────────────────────
        //  Sibling component cache (resolved in Awake)
        // ─────────────────────────────────────────────────────────────

        private DepthCapture _depthCapture;
        private VolumeIntegrator _volumeIntegrator;
        private MeshExtractor _meshExtractor;
        private RoomScanPersistence _persistence;
        private RoomAnchorManager _roomAnchor;

        // Optional modules (discovered, not required)
        private PassthroughCameraProvider _cameraProvider;
        private TriplanarCache _triplanarCache;
        private KeyframeCollector _keyframeCollector;
        private PointCloudExporter _pointCloudExporter;
        private IGSplatProvider _gsplatProvider;
        private TextureRefinement _textureRefinement;
        private DebugMenuController _debugMenu;
        private ICameraProvider _customCameraProvider;
        private IRoomScanModule[] _modules;

        // ─────────────────────────────────────────────────────────────
        //  Public read-only state
        // ─────────────────────────────────────────────────────────────

        /// <summary>Toggle blue tint overlay on frozen voxels in live mesh modes.</summary>
        public bool ShowFreezeTint
        {
            get => showFreezeTint;
            set { showFreezeTint = value; Shader.SetGlobalFloat(NoFreezeTintID, value ? 0f : 1f); }
        }

        public bool IsScanning { get; private set; }
        public ScanRenderMode CurrentRenderMode => renderMode;
        public bool IsGsTrainingInProgress => _serverTrainingInProgress;
        public DebugMenuController DebugMenu => _debugMenu;

        /// <summary>The core volume integrator component.</summary>
        public VolumeIntegrator VolumeIntegrator => _volumeIntegrator;
        /// <summary>The core depth capture component.</summary>
        public DepthCapture DepthCapture => _depthCapture;
        /// <summary>The core mesh extractor component.</summary>
        public MeshExtractor MeshExtractor => _meshExtractor;
        /// <summary>The active camera provider (custom or passthrough).</summary>
        public ICameraProvider ActiveCameraProvider => GetActiveCameraProvider();
        /// <summary>The optional Gaussian Splat provider, or null if the GSplat module is not attached.</summary>
        public IGSplatProvider GSplatProvider => _gsplatProvider;
        /// <summary>True when the TextureRefinement module is attached.</summary>
        public bool HasTextureRefinementModule => _textureRefinement != null;

        // ─────────────────────────────────────────────────────────────
        //  Events
        // ─────────────────────────────────────────────────────────────

        public event Action ScanStarted;
        public event Action ScanStopped;
        public event Action<ScanRenderMode> RenderModeChanged;

        /// <summary>
        /// Raised each frame a passthrough camera frame is fed to the volume integrator.
        /// Parameters: frame texture, camera pose, focal length, principal point, sensor resolution, current resolution.
        /// </summary>
        public event Action<Texture, Pose, Vector2, Vector2, Vector2, Vector2> ColorFrameProvided;

        /// <summary>Raised after each depth integration pass.</summary>
        public event Action Integrated;
        /// <summary>Raised after each mesh extraction pass.</summary>
        public event Action MeshExtracted;

        // ─────────────────────────────────────────────────────────────
        //  Private state
        // ─────────────────────────────────────────────────────────────

        private float _lastIntegrationTime;
        private float _lastMeshTime;
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

        private float IntegrationInterval => 1f / integrationHz;
        private float MeshInterval => 1f / meshExtractionHz;

        // ─────────────────────────────────────────────────────────────
        //  Lifecycle
        // ─────────────────────────────────────────────────────────────

        private void Awake()
        {
            Instance = this;
            Logger.Level = logLevel;
            CacheComponents();
            SetSafeShaderDefaults();
        }

        private void Start()
        {
            _modules = GetComponents<IRoomScanModule>();
            foreach (var m in _modules) m.OnModuleInitialize(this);

            if (_persistence != null)
                _persistence.LoadCompleted += ApplyRenderMode;

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
            _gsplatProvider = GetComponent<IGSplatProvider>();
            _textureRefinement = GetComponent<TextureRefinement>();
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
            _started = true;
            Logger.Info("Room ready — call StartScanning() to begin");
        }

        private void OnDisable()
        {
            StopScanning();
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
                Logger.Info("All scan + export data cleared");
                _clearDoneCallback?.Invoke();
                _clearDoneCallback = null;
            }

            if (_reinitExportPending)
            {
                _reinitExportPending = false;
                if (_keyframeCollector != null)
                    _keyframeCollector.ReinitExportDir();
            }

            // Enforce triplanar override every frame (not just while scanning)
            // so loaded scans don't flash triplanar before ApplyRenderMode runs.
            if (renderMode == ScanRenderMode.Vertex)
                Shader.SetGlobalFloat(TriAvailableID, 0f);

            if (!IsScanning || !DepthCapture.DepthAvailable) return;

            float t = Time.time;

            if (t - _lastIntegrationTime >= IntegrationInterval)
            {
                _lastIntegrationTime = t;

                ProvideColorFrame();
                _volumeIntegrator.Integrate();
                Integrated?.Invoke();
                _integrateCount++;

                if (t - _lastMeshTime >= MeshInterval)
                {
                    _lastMeshTime = t;
                    _meshExtractor.Extract();
                    MeshExtracted?.Invoke();
                }
            }

            if (t - _lastScannerLog >= 5f)
            {
                _lastScannerLog = t;
                Logger.Verbose($"Scanner: integrations={_integrateCount}, " +
                    $"depthAvail={DepthCapture.DepthAvailable}");
            }
        }

        // ═════════════════════════════════════════════════════════════
        //  PUBLIC API — call from any client, input handler, or UI
        // ═════════════════════════════════════════════════════════════

        /// <summary>
        /// Begins depth integration and mesh extraction. Resets relocation state,
        /// clears in-memory keyframes, and starts the active camera provider.
        /// </summary>
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
                        Logger.Error($"StartScanning clear error: {e.Message}");
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

            Logger.Info($"StartScanning — integrationCount={_volumeIntegrator.IntegrationCount}");
            ScanStarted?.Invoke();
            if (_modules != null)
                foreach (var m in _modules) m.OnScanStarted();
        }

        /// <summary>
        /// Pauses depth integration and stops the camera provider.
        /// </summary>
        public void StopScanning()
        {
            if (!IsScanning) return;
            IsScanning = false;

            ICameraProvider provider = GetActiveCameraProvider();
            provider?.StopCapture();

            ScanStopped?.Invoke();
            if (_modules != null)
                foreach (var m in _modules) m.OnScanStopped();
        }

        public void ToggleScanning()
        {
            if (IsScanning) StopScanning();
            else StartScanning();
        }

        /// <summary>
        /// Clears the TSDF volume and reinitializes the GPU mesh pipeline.
        /// Does not delete persisted files — use <see cref="ClearAllDataAsync"/> for a full wipe.
        /// </summary>
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

                _gsplatProvider?.ClearSplat();
                _gsplatProvider?.ResetSplatTransform();
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
                Logger.Error($"ClearAllData sync error: {e.Message}\n{e.StackTrace}");
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
                    Logger.Error($"ClearAllData I/O error: {e.Message}");
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

        /// <summary>
        /// Switches the active render mode and updates mesh/splat visibility accordingly.
        /// </summary>
        public void SetRenderMode(ScanRenderMode newMode)
        {
            renderMode = newMode;
            ApplyRenderMode();
            RenderModeChanged?.Invoke(renderMode);
            Logger.Info($"Render mode: {renderMode}");
        }

        /// <summary>
        /// Advances to the next available render mode, skipping modes whose backing
        /// data or module is not present.
        /// </summary>
        public void CycleRenderMode()
        {
            ScanRenderMode[] order =
            {
                ScanRenderMode.Wireframe, ScanRenderMode.Vertex, ScanRenderMode.Triplanar,
                ScanRenderMode.Refined, ScanRenderMode.HQRefined,
                ScanRenderMode.Splat, ScanRenderMode.None
            };

            int cur = Array.IndexOf(order, renderMode);
            if (cur < 0) cur = 0;

            for (int i = 1; i <= order.Length; i++)
            {
                var candidate = order[(cur + i) % order.Length];
                if (!IsModeAvailable(candidate)) continue;

                if (candidate == ScanRenderMode.Splat && HasDownloadedSplat
                    && (_gsplatProvider == null || !_gsplatProvider.HasServerTrainedSplats))
                {
                    LoadDownloadedSplat();
                    return;
                }

                SetRenderMode(candidate);
                return;
            }
        }

        /// <summary>
        /// Returns true if the given render mode's backing module/data is present.
        /// </summary>
        public bool IsModeAvailable(ScanRenderMode mode)
        {
            return mode switch
            {
                ScanRenderMode.Vertex => true,
                ScanRenderMode.Wireframe => true,
                ScanRenderMode.None => true,
                ScanRenderMode.Triplanar => _triplanarCache != null,
                ScanRenderMode.Refined => HasRefinedTexture,
                ScanRenderMode.HQRefined => HasHQRefinedTexture,
                ScanRenderMode.Splat => (_gsplatProvider != null && _gsplatProvider.HasServerTrainedSplats) || HasDownloadedSplat,
                _ => false
            };
        }

        /// <summary>
        /// Freezes voxels currently visible in the camera frustum, preventing further integration updates.
        /// </summary>
        public void FreezeInView()
        {
            if (_volumeIntegrator == null) return;
            if (!TryGetCameraIntrinsics(out var pose, out var focal, out var principal,
                    out var sensor, out var current)) return;

            _volumeIntegrator.FreezeInView(pose.position, pose.rotation,
                focal, principal, sensor, current);
        }

        /// <summary>
        /// Unfreezes previously frozen voxels in the current camera frustum, allowing integration to resume.
        /// </summary>
        public void UnfreezeInView()
        {
            if (_volumeIntegrator == null) return;
            if (!TryGetCameraIntrinsics(out var pose, out var focal, out var principal,
                    out var sensor, out var current)) return;

            _volumeIntegrator.UnfreezeInView(pose.position, pose.rotation,
                focal, principal, sensor, current);
        }

        /// <summary>
        /// Persists the current scan (TSDF volume, keyframes, artifacts) to a new package on disk.
        /// </summary>
        public async Task<bool> SaveScanAsync()
        {
            if (_persistence == null) return false;
            return await _persistence.SaveToNewPackageAsync();
        }

        /// <summary>
        /// Loads a previously saved scan package by ID, restoring volume data and artifacts.
        /// </summary>
        public async Task<bool> LoadPackageAsync(string pkgId)
        {
            if (_persistence == null) return false;
            return await _persistence.LoadPackageAsync(pkgId);
        }

        /// <summary>
        /// Exports the current keyframe point cloud to disk via <see cref="PointCloudExporter"/>.
        /// </summary>
        public async Task ExportPointCloudAsync()
        {
            if (_pointCloudExporter != null)
                await _pointCloudExporter.ExportAsync();
        }

        /// <summary>
        /// Kicks off the server-side Gaussian Splat training pipeline. Downloads the trained PLY on completion.
        /// </summary>
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

        /// <summary>
        /// Registers a transform as an exclusion zone; voxels near it are skipped during integration (e.g. the user's head).
        /// </summary>
        public void AddExclusionZone(Transform t)
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ExclusionZones.Add(t);
        }

        /// <summary>
        /// Unregisters a previously added exclusion zone.
        /// </summary>
        public void RemoveExclusionZone(Transform t)
        {
            if (_volumeIntegrator != null)
                _volumeIntegrator.ExclusionZones.Remove(t);
        }

        // ─────────────────────────────────────────────────────────────
        //  Texture Refinement
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Runs the on-device texture refinement pipeline: UV-unwraps the mesh via xatlas,
        /// bakes a texture atlas from collected keyframes, and switches to Refined render mode.
        /// </summary>
        public async void StartTextureRefinement()
        {
            if (IsRefining) return;
            if (_textureRefinement == null)
            {
                Logger.Warning("TextureRefinement module not attached — skipping refinement");
                return;
            }
            IsRefining = true;
            RefineStatus = "Starting...";

            Action<string> statusHandler = s => RefineStatus = s;
            _textureRefinement.StatusChanged += statusHandler;
            try
            {
                string keyframeDir = KeyframeDirectory;
                var unwrap = await EnsureUnwrappedAsync();
                byte[] atlasPixels = await _textureRefinement.BakeAtlasAsync(
                    unwrap, keyframeDir, KeyframeRelocation);

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

                Logger.Info("On-device texture refinement complete");
            }
            catch (Exception e)
            {
                Logger.Error($"Texture refinement failed: {e.Message}\n{e.StackTrace}");
                RefineStatus = "Failed";
            }
            finally
            {
                _textureRefinement.StatusChanged -= statusHandler;
                IsRefining = false;
            }
        }

        /// <summary>
        /// Uploads the on-device atlas to the server for super-resolution enhancement.
        /// Triggers on-device refinement first if it hasn't been run yet.
        /// </summary>
        public async void StartHQRefinement()
        {
            if (IsHQRefining) return;
            if (_textureRefinement == null)
            {
                Logger.Warning("TextureRefinement module not attached — skipping HQ refinement");
                return;
            }
            IsHQRefining = true;
            HQRefineStatus = "Starting...";

            try
            {
                if (_gsplatProvider == null)
                {
                    HQRefineStatus = "No server configured";
                    return;
                }

                // Auto-trigger on-device refinement if not done yet
                if (!LastRefinedResult.HasValue)
                {
                    HQRefineStatus = "Running on-device refine first...";
                    Action<string> hqStatusHandler = s => HQRefineStatus = s;
                    _textureRefinement.StatusChanged += hqStatusHandler;
                    try
                    {
                        StartTextureRefinement();
                        while (IsRefining) await Task.Yield();
                    }
                    finally { _textureRefinement.StatusChanged -= hqStatusHandler; }

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

                int scale = _textureRefinement != null ? _textureRefinement.hqRefineScale : 2;
                Logger.Info($"HQ refine: uploading {pngBytes.Length / 1024}KB atlas, scale={scale}");
                HQRefineStatus = $"Uploading ({pngBytes.Length / 1024}KB)...";

                byte[] resultPng = await _gsplatProvider.EnhanceAtlasAsync(pngBytes, scale, inpaint: true);

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
                    Logger.Info($"HQ atlas enhancement complete: {tex.width}x{tex.height}");
                }
                else
                {
                    UnityEngine.Object.Destroy(tex);
                    HQRefineStatus = "Failed to decode atlas";
                }
            }
            catch (Exception e)
            {
                Logger.Error($"HQ refinement failed: {e.Message}\n{e.StackTrace}");
                HQRefineStatus = "Failed";
            }
            finally
            {
                IsHQRefining = false;
            }
        }

        /// <summary>
        /// Uploads the refined mesh to the server for smoothing and plane-snapping enhancement.
        /// Requires on-device refinement to have completed first.
        /// </summary>
        public async void StartMeshEnhancement()
        {
            if (IsMeshEnhancing) return;
            if (_textureRefinement == null)
            {
                Logger.Warning("TextureRefinement module not attached — skipping mesh enhancement");
                return;
            }
            IsMeshEnhancing = true;
            MeshEnhanceStatus = "Starting...";

            try
            {
                if (_gsplatProvider == null)
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
                Logger.Info($"Mesh enhance: uploading {meshBin.Length / 1024}KB ({r.Positions.Length} verts)");

                MeshEnhanceStatus = $"Uploading ({meshBin.Length / 1024}KB)...";
                byte[] resultBin = await _gsplatProvider.EnhanceMeshAsync(
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
                Logger.Info($"Mesh enhancement complete: {enhanced.Positions.Length} verts");
            }
            catch (Exception e)
            {
                Logger.Error($"Mesh enhancement failed: {e.Message}\n{e.StackTrace}");
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

            Logger.Info($"Refined atlas applied: " +
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
                Logger.Info("Reusing cached UV-unwrapped mesh");
                return _cachedUnwrap.Value;
            }

            // Reconstruct from persisted refined mesh if available (skip xatlas)
            if (LastRefinedResult.HasValue)
            {
                var r = LastRefinedResult.Value;
                var unwrap = ReconstructUnwrapFromResult(r);
                _cachedUnwrap = unwrap;
                EnsureRefinedMesh(unwrap);
                Logger.Info("Reconstructed UV mesh from persisted refined_mesh.bin (no xatlas needed)");
                return unwrap;
            }

            string kfDir = KeyframeDirectory;
            var unwrap2 = await _textureRefinement.UnwrapMeshAsync(kfDir, KeyframeRelocation);
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

            var shader = _textureRefinement != null ? _textureRefinement.refinedMeshShader : null;
            if (shader == null) shader = Shader.Find("Genesis/RefinedMesh");
            if (shader == null)
            {
                Logger.Warning("Genesis/RefinedMesh shader not found, using URP/Unlit");
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
            if (provider != null && provider.IsReady)
            {
                pose = provider.CameraPose;
                if (_depthCapture != null)
                    pose = _depthCapture.TrackingToWorld(pose);
                focal = provider.FocalLength;
                principal = provider.PrincipalPoint;
                sensor = provider.SensorResolution;
                current = provider.CurrentResolution;
                return true;
            }

            // Fallback to main camera intrinsics when no provider is available
            var cam = Camera.main;
            if (cam == null) return false;

            var ct = cam.transform;
            pose = new Pose(ct.position, ct.rotation);
            float w = cam.pixelWidth;
            float h = cam.pixelHeight;
            float vFovRad = cam.fieldOfView * Mathf.Deg2Rad;
            float fy = h / (2f * Mathf.Tan(vFovRad * 0.5f));
            focal = new Vector2(fy, fy);
            principal = new Vector2(w * 0.5f, h * 0.5f);
            sensor = new Vector2(w, h);
            current = new Vector2(w, h);
            return true;
        }

        private void SetupHeadExclusion()
        {
            if (_volumeIntegrator == null) return;

            var cam = Camera.main;
            if (cam != null)
            {
                AddExclusionZone(cam.transform);
                Logger.Info($"Head exclusion zone added: {cam.gameObject.name}");
            }
            else
            {
                Logger.Warning("No main camera found for head exclusion zone");
            }
        }

        private static readonly int NormalFallbackID = Shader.PropertyToID("_RSNormalFallback");

        private void SetSafeShaderDefaults()
        {
            Shader.SetGlobalFloat(TriAvailableID, 0f);
            Shader.SetGlobalFloat(NormalFallbackID, 0f);
            Shader.SetGlobalFloat(WireframeID, 0f);
            Shader.SetGlobalFloat(WireThicknessID, wireThickness);
            Shader.SetGlobalFloat(NoFreezeTintID, showFreezeTint ? 0f : 1f);
        }

        private int _colorFrameLog;
        private bool _cameraAvailable;
        private void ProvideColorFrame()
        {
            ICameraProvider provider = GetActiveCameraProvider();

            // IsPlaying = camera subsystem running (stable signal).
            // IsReady = new frame available this tick (toggles at camera fps < app fps).
            bool cameraPlaying = provider != null && provider.IsPlaying;

            if (cameraPlaying && !_cameraAvailable)
            {
                _cameraAvailable = true;
                Shader.SetGlobalFloat(NormalFallbackID, 0f);
                Logger.Info("Camera playing — disabling normal fallback");
            }
            else if (!cameraPlaying && (_cameraAvailable || _colorFrameLog == 0))
            {
                _cameraAvailable = false;
                Shader.SetGlobalFloat(NormalFallbackID, 1f);
                Logger.Info("Camera not playing — enabling normal fallback rendering");
            }

            if (provider != null && provider.IsReady)
            {
                Texture frame = provider.CurrentFrame;
                if (frame != null)
                {
                    _depthCapture?.SetRGBGuide(frame);

                    Pose pose = provider.CameraPose;
                    if (_depthCapture != null)
                        pose = _depthCapture.TrackingToWorld(pose);
                    Vector2 focal = provider.FocalLength;
                    Vector2 principal = provider.PrincipalPoint;
                    Vector2 sensor = provider.SensorResolution;
                    Vector2 current = provider.CurrentResolution;

                    _volumeIntegrator.SetCameraData(
                        frame, pose.position, pose.rotation,
                        focal, principal, sensor, current);

                    ColorFrameProvided?.Invoke(frame, new Pose(pose.position, pose.rotation),
                        focal, principal, sensor, current);

                    _colorFrameLog++;
                    if (_colorFrameLog <= 3 || _colorFrameLog % 50 == 0)
                        Logger.Verbose($"ColorFrame #{_colorFrameLog}: " +
                            $"frame={frame.width}x{frame.height}");

                    return;
                }
            }

            _colorFrameLog++;
            if (_colorFrameLog <= 5)
                Logger.Verbose($"ColorFrame #{_colorFrameLog}: NO FRAME " +
                    $"playing={cameraPlaying}, " +
                    $"isReady={provider?.IsReady ?? false}");

            _volumeIntegrator.SetCameraData(null, Vector3.zero, Quaternion.identity,
                Vector2.one, Vector2.zero, Vector2.one, Vector2.one);
        }

        private static readonly int NoFreezeTintID = Shader.PropertyToID("_RSNoFreezeTint");
        private static readonly int TriAvailableID = Shader.PropertyToID("_RSTriAvailable");
        private static readonly int WireframeID = Shader.PropertyToID("_RSWireframe");
        private static readonly int WireThicknessID = Shader.PropertyToID("_RSWireThickness");

        private void ApplyRenderMode()
        {
            var gpuRenderer = _meshExtractor != null ? _meshExtractor.GetComponent<GPUMeshRenderer>() : null;

            bool gpuMeshVisible = renderMode == ScanRenderMode.Vertex
                               || renderMode == ScanRenderMode.Triplanar
                               || renderMode == ScanRenderMode.Wireframe;
            bool refinedVisible = renderMode == ScanRenderMode.Refined
                               || renderMode == ScanRenderMode.HQRefined;

            if (gpuRenderer != null)
                gpuRenderer.RenderVisible = gpuMeshVisible;
            if (_gsplatProvider != null)
                _gsplatProvider.RenderVisible = renderMode == ScanRenderMode.Splat;
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

            // Vertex mode: force triplanar off; Triplanar mode: reassert globals from cache
            if (renderMode == ScanRenderMode.Vertex)
                Shader.SetGlobalFloat(TriAvailableID, 0f);
            else if (renderMode == ScanRenderMode.Triplanar && _triplanarCache != null)
                _triplanarCache.UpdateShaderGlobals();

            Shader.SetGlobalFloat(WireframeID, renderMode == ScanRenderMode.Wireframe ? 1f : 0f);
            Shader.SetGlobalFloat(WireThicknessID, wireThickness);
            Shader.SetGlobalFloat(NoFreezeTintID, showFreezeTint ? 0f : 1f);
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
                    _gsplatProvider?.ClearSplat();
                    _downloadedPlyData = null;
                    SetRenderMode(ScanRenderMode.Vertex);
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
                        SetRenderMode(ScanRenderMode.Vertex);
                    }
                    break;

                case ScanRenderMode.HQRefined:
                    _persistence.DeleteArtifactFromPackage(ArtifactType.HQRefined);
                    HasHQRefinedTexture = false;
                    _hqAtlasTexture = null;
                    SetRenderMode(HasRefinedTexture ? ScanRenderMode.Refined : ScanRenderMode.Vertex);
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

            Logger.Info("Enhanced mesh deleted, original restored");
        }

        /// <summary>
        /// Loads the downloaded splat into GPU buffers and switches to splat render mode.
        /// Call after training + download completes.
        /// </summary>
        public void LoadDownloadedSplat()
        {
            if (_downloadedPlyData == null || _downloadedPlyData.Length == 0)
            {
                Logger.Warning("No downloaded splat data to load");
                return;
            }
            _gsplatProvider?.LoadTrainedPly(_downloadedPlyData);
            SetRenderMode(ScanRenderMode.Splat);
            Logger.Info("Trained Gaussians loaded and rendering");
        }

        private async void RunServerTrainingAsync()
        {
            if (_serverTrainingInProgress) return;
            _serverTrainingInProgress = true;

            try
            {
                if (_gsplatProvider == null)
                {
                    Logger.Error("No GSplat provider available");
                    return;
                }

                Logger.Info("Starting server-side GS training pipeline...");
                byte[] plyData = await _gsplatProvider.RunServerTrainingAsync(KeyframeDirectory, KeyframeRelocation);
                if (plyData == null || plyData.Length == 0) return;

                _downloadedPlyData = plyData;

                if (_persistence != null && _persistence.HasActivePackage)
                    await _persistence.SaveArtifactAsync(ArtifactType.Splat, plyData);

                LoadDownloadedSplat();
            }
            catch (Exception e)
            {
                Logger.Error($"Server training pipeline error: {e.Message}\n{e.StackTrace}");
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

    }
}
