using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Manages the GPU TSDF + color volume and dispatches compute-shader integration, pruning,
    /// and freeze/unfreeze passes. Voxels are integrated from depth and optional camera color
    /// each frame, with configurable convergence, exclusion zones, and warmup clearing.
    /// </summary>
    public class VolumeIntegrator : MonoBehaviour
    {
        public static VolumeIntegrator Instance { get; private set; }

        [SerializeField] private ComputeShader compute;

        [Header("Volume")]
        [SerializeField] private int3 voxelCount = new(256, 256, 256);
        [SerializeField] private float voxelSize = 0.05f;
        [SerializeField] private float voxelDistance = 0.15f;
        [SerializeField] private float voxelMin = 0.1f;

        [Header("Integration")]
        [SerializeField] private float depthDisparityThreshold = 0.5f;
        [SerializeField] private float maxUpdateDist = 5f;
        [SerializeField] private float minUpdateDist = 0.5f;
        [SerializeField] private int maxFrustumPositions = 1000000;

        [Header("Convergence")]
        [Tooltip("Blend strength. Higher = faster convergence and correction. (default 0.8)")]
        [SerializeField, Range(0.1f, 2f)] private float blendRate = 0.8f;
        [Tooltip("Weight resistance to blending. Lower = faster corrections but less stable. (default 2.5)")]
        [SerializeField, Range(0.5f, 10f)] private float stability = 2.5f;
        [Tooltip("How fast weight accumulates per frame. Lower = bad data builds less confidence. (default 0.025)")]
        [SerializeField, Range(0.005f, 0.1f)] private float weightGrowth = 0.025f;
        [Tooltip("Maximum weight any voxel can reach. Lower = all areas correct equally fast. (default 0.5)")]
        [SerializeField, Range(0.1f, 1f)] private float maxWeight = 0.5f;

        [Header("Meshing")]
        [Tooltip("Min voxel confidence weight for Surface Nets to generate mesh. Higher = fewer phantom surfaces. (default 0.08)")]
        [SerializeField, Range(0.01f, 0.5f)] private float minMeshWeight = 0.08f;
        public float MinMeshWeight => minMeshWeight;

        [Header("Camera Color")]
        [Tooltip("Exposure boost for camera texture. Quest 3 passthrough cameras produce dim images. (default 3.0)")]
        [SerializeField, Range(1f, 10f)] private float cameraExposure = 3f;

        private RenderTexture _volume;
        private RenderTexture _colorVolume;

        /// <summary>3D RenderTexture (R8G8_SNorm) storing the truncated signed distance field.</summary>
        public RenderTexture Volume => _volume;
        /// <summary>3D RenderTexture (RGBA8_UNorm) storing per-voxel accumulated color.</summary>
        public RenderTexture ColorVolume => _colorVolume;
        public int3 VoxelCount => voxelCount;
        public float VoxelSize => voxelSize;
        public float VoxelDistance => voxelDistance;

        private static readonly int VolumeRWID = Shader.PropertyToID("gsVolumeRW");
        private static readonly int VolumeID = Shader.PropertyToID("gsVolume");
        private static readonly int ColorVolumeRWID = Shader.PropertyToID("gsColorVolumeRW");
        private static readonly int ColorVolumeID = Shader.PropertyToID("gsColorVolume");
        private static readonly int VoxCountID = Shader.PropertyToID("gsVoxCount");
        private static readonly int VoxSizeID = Shader.PropertyToID("gsVoxSize");
        private static readonly int VoxMinID = Shader.PropertyToID("gsVoxMin");
        private static readonly int VoxDistID = Shader.PropertyToID("gsVoxDist");
        private static readonly int FrustumVolumeID = Shader.PropertyToID("gsFrustumVolume");
        private static readonly int DepthDispThreshID = Shader.PropertyToID("gsDepthDispThresh");
        private static readonly int NumExclusionsID = Shader.PropertyToID("gsNumExclusions");
        private static readonly int ExclusionP0ID = Shader.PropertyToID("gsExclusionP0");
        private static readonly int ExclusionP1ID = Shader.PropertyToID("gsExclusionP1");
        private static readonly int EraseBodyID = Shader.PropertyToID("gsEraseBody");
        private static readonly int EraseMaxWeightID = Shader.PropertyToID("gsEraseMaxWeight");
        private static readonly int MaxUpdateDistID = Shader.PropertyToID("gsMaxUpdateDist");
        private static readonly int BlendRateID = Shader.PropertyToID("gsBlendRate");
        private static readonly int StabilityID = Shader.PropertyToID("gsStability");
        private static readonly int WeightGrowthID = Shader.PropertyToID("gsWeightGrowth");
        private static readonly int MaxWeightID = Shader.PropertyToID("gsMaxWeight");
        private static readonly int CamRGBID = Shader.PropertyToID("gsCamRGB");
        private static readonly int CamAvailableID = Shader.PropertyToID("gsCamAvailable");
        private static readonly int CamPosID = Shader.PropertyToID("gsCamPos");
        private static readonly int CamInvRotID = Shader.PropertyToID("gsCamInvRot");
        private static readonly int CamFocalLenID = Shader.PropertyToID("gsCamFocalLen");
        private static readonly int CamPrincipalPtID = Shader.PropertyToID("gsCamPrincipalPt");
        private static readonly int CamSensorResID = Shader.PropertyToID("gsCamSensorRes");
        private static readonly int CamCurrentResID = Shader.PropertyToID("gsCamCurrentRes");
        private static readonly int CamExposureID = Shader.PropertyToID("gsCamExposure");
        private static readonly int ConfineToRoomID = Shader.PropertyToID("gsConfineToRoom");
        private static readonly int NumRoomClipPlanesID = Shader.PropertyToID("gsNumRoomClipPlanes");
        private static readonly int RoomClipPlanesID = Shader.PropertyToID("gsRoomClipPlanes");
        private static readonly int NumScreenStampsID = Shader.PropertyToID("gsNumScreenStamps");
        private static readonly int ScreenCenterID = Shader.PropertyToID("gsScreenCenter");
        private static readonly int ScreenInwardID = Shader.PropertyToID("gsScreenInward");
        private static readonly int ScreenAxisID = Shader.PropertyToID("gsScreenAxis");
        private static readonly int ScreenBitangentID = Shader.PropertyToID("gsScreenBitangent");
        private static readonly int UseRoomAabbID = Shader.PropertyToID("gsUseRoomAabb");
        private static readonly int RoomAabbMinID = Shader.PropertyToID("gsRoomAabbMin");
        private static readonly int RoomAabbMaxID = Shader.PropertyToID("gsRoomAabbMax");
        private static readonly int StampVoxMinID = Shader.PropertyToID("gsStampVoxMin");
        private static readonly int StampVoxMaxID = Shader.PropertyToID("gsStampVoxMax");

        public float CameraExposure => cameraExposure;

        [Header("Warmup")]
        [Tooltip("Clear the volume after this many integrations to discard sensor startup noise. 0 = disabled.")]
        [SerializeField] private int warmupIntegrations = 3;

        [Header("Body exclusion")]
        [Tooltip("Torso capsule radius around the head, world-up. Smaller than 0.6 so leaning into a corner is not carved.")]
        [SerializeField] private float torsoRadius = 0.35f;
        [SerializeField] private float torsoAbove = 0.25f;
        [SerializeField] private float torsoBelow = 1.7f;
        [SerializeField] private float handRadius = 0.14f;
        [SerializeField] private float handHalfLength = 0.08f;
        [SerializeField] private float forearmRadius = 0.08f;
        [SerializeField] private float forearmLength = 0.28f;
        [SerializeField] private float shoulderDrop = 0.2f;
        [SerializeField] private float shoulderLateral = 0.2f;
        [Tooltip("After Integrate, zero non-frozen voxels inside hand/forearm capsules whose weight is below eraseMaxWeight. Off by default — exclusion already skips those voxels.")]
        [SerializeField] private bool eraseBodyBlobs = false;
        [Tooltip("Weight ceiling for the optional eraser. Must sit above SEED_WEIGHT (0.10) or a freshly seeded hand voxel is never cleared. Independent of MinMeshWeight.")]
        [SerializeField, Range(0.1f, 0.5f)] private float eraseMaxWeight = 0.2f;

        [Header("Shell coverage")]
        [Tooltip("Soft-stamp the captured plane over small uncovered wall/floor/ceiling patches whose neighbours lie on one plane.")]
        [SerializeField] private bool autoFillShellGaps = true;
        [Tooltip("Largest gap (in shell cells) the auto-fill will close. 6 cells at 10 cm is ~25×25 cm.")]
        [SerializeField, Range(1, 16)] private int fillMaxCells = 6;
        [Tooltip("Weight written by auto-fill. Above MinMeshWeight so it meshes; below anything real depth accumulates so real data overrides it.")]
        [SerializeField, Range(0.1f, 0.5f)] private float fillWeight = 0.15f;
        [Tooltip("Max spread (m) of neighbour hit offsets for a gap to count as 'on one plane'.")]
        [SerializeField] private float fillNeighborSpreadMax = 0.06f;
        [Tooltip("Max fill dispatches per coverage tick (~1 Hz).")]
        [SerializeField, Range(1, 32)] private int fillMaxPerTick = 8;
        [Tooltip("Cluster-local 6-neighbour close for small furniture gaps.")]
        [SerializeField] private bool closeFurnitureHoles = true;

        public bool AutoFillShellGaps { get => autoFillShellGaps; set => autoFillShellGaps = value; }
        public bool CloseFurnitureHoles { get => closeFurnitureHoles; set => closeFurnitureHoles = value; }
        public int FillMaxCells => fillMaxCells;
        public float FillNeighborSpreadMax => fillNeighborSpreadMax;
        public int FillMaxPerTick => fillMaxPerTick;

        [Header("Pruning")]
        [SerializeField] private float pruneIntervalSeconds = 3f;

        private ComputeKernelHelper _clearKernel;
        private ComputeKernelHelper _integrateKernel;
        private ComputeKernelHelper _stampKernel;
        private ComputeKernelHelper _pruneKernel;
        private ComputeKernelHelper _freezeKernel;
        private ComputeKernelHelper _unfreezeKernel;
        private ComputeKernelHelper _eraseKernel;
        private ComputeKernelHelper _shellMarchKernel;
        private ComputeKernelHelper _fillPatchKernel;
        private ComputeKernelHelper _closeHolesKernel;

        private ComputeBuffer _shellPos;
        private ComputeBuffer _shellNrm;
        private ComputeBuffer _shellResult;
        private int _shellCount;
        private bool _shellReadbackPending;
        private int _shellReadbackGeneration;
        private readonly uint[] _shellResultCpu = new uint[ShellCellSet.MaxCells];

        /// <summary>
        /// Bumped by every <see cref="SetShellCells"/> / <see cref="ClearShellCells"/>.
        /// A readback carries the generation it was dispatched under so a
        /// result that raced a cell rebuild can be dropped instead of being
        /// decoded against the new cell order.
        /// </summary>
        public int ShellGeneration { get; private set; }

        private static readonly int CoverCellPosID = Shader.PropertyToID("gsCoverCellPos");
        private static readonly int CoverCellNrmID = Shader.PropertyToID("gsCoverCellNrm");
        private static readonly int CoverResultID = Shader.PropertyToID("gsCoverResult");
        private static readonly int CoverCellCountID = Shader.PropertyToID("gsCoverCellCount");
        private static readonly int CoverMinWeightID = Shader.PropertyToID("gsCoverMinWeight");
        private static readonly int FillCenterID = Shader.PropertyToID("gsFillCenter");
        private static readonly int FillInwardID = Shader.PropertyToID("gsFillInward");
        private static readonly int FillAxisID = Shader.PropertyToID("gsFillAxis");
        private static readonly int FillBitangentID = Shader.PropertyToID("gsFillBitangent");
        private static readonly int FillWeightID = Shader.PropertyToID("gsFillWeight");

        /// <summary>
        /// Raised on the main thread after each shell-coverage readback with
        /// the per-cell result words (bit0 covered, bit1 observed-empty,
        /// bits 8..15 hit step), the count, and the <see cref="ShellGeneration"/>
        /// the march ran under. The array is reused; consume synchronously.
        /// </summary>
        public event Action<uint[], int, int> ShellResultReady;

        private ComputeBuffer _frustumVolume;
        private bool _frustumReady;
        private float _lastPruneTime;

        // ── Scan analysis (coverage counters + mesh closure) ───────────
        // One time-sliced GPU cycle per analysisIntervalSeconds, one 128-byte
        // readback at its end. See ScanAnalysis / StepAnalysis.
        private ComputeKernelHelper _coverageKernel;
        private ComputeKernelHelper _ccResetKernel;
        private ComputeKernelHelper _ccInsertKernel;
        private ComputeKernelHelper _ccLinkKernel;
        private ComputeKernelHelper _ccJumpKernel;
        private ComputeKernelHelper _ccClassifyKernel;
        private ComputeKernelHelper _ccFinalizeKernel;
        private ComputeKernelHelper _ccCentroidKernel;
        private ComputeBuffer _analysisResult;
        private GraphicsBuffer _ccEdgesSnap;
        private GraphicsBuffer _ccCountersSnap;
        private ComputeBuffer _ccHashKey;
        private ComputeBuffer _ccHashVal;
        private ComputeBuffer _ccLabel;
        private ComputeBuffer _ccComp;
        private AnalysisState _analysisState;
        private int _analysisRound;
        private int _analysisExtractSeen = -1;
        private float _lastAnalysisTime;
        private readonly uint[] _analysisCpu = new uint[AnalysisWords];

        private const int AnalysisWords = 32;
        private const int CcHashSize = GPUSurfaceNets.MaxOpenEdges * 2;

        private static readonly int AnalysisResultID = Shader.PropertyToID("_AnalysisResult");
        private static readonly int ConfidentWeightID = Shader.PropertyToID("gsConfidentWeight");
        private static readonly int ColorVolumeReadID = Shader.PropertyToID("gsColorVolumeRead");
        private static readonly int CcEdgesID = Shader.PropertyToID("gsCcEdges");
        private static readonly int CcCountersID = Shader.PropertyToID("gsCcCounters");
        private static readonly int CcHashKeyID = Shader.PropertyToID("gsCcHashKey");
        private static readonly int CcHashValID = Shader.PropertyToID("gsCcHashVal");
        private static readonly int CcLabelID = Shader.PropertyToID("gsCcLabel");
        private static readonly int CcCompID = Shader.PropertyToID("gsCcComp");
        private static readonly int CcCapID = Shader.PropertyToID("gsCcCap");
        private static readonly int CcHashMaskID = Shader.PropertyToID("gsCcHashMask");
        private static readonly int CcHashSizeID = Shader.PropertyToID("gsCcHashSize");
        private static readonly int CcMinEdgesID = Shader.PropertyToID("gsCcMinEdges");
        private static readonly int CcCutTolID = Shader.PropertyToID("gsCcCutTol");

        enum AnalysisState { Idle, Link, Classify, Finalize, Readback }

        [Header("Scan analysis")]
        [Tooltip("Seconds between analysis cycles. Each cycle is ~16 frames of small kernels and one 128-byte readback.")]
        [SerializeField, Range(0.25f, 5f)] private float analysisIntervalSeconds = 1f;
        [Tooltip("Surface voxels at or above this |weight| count as refined. Weight only grows with good observations (max 0.5) and the blend rate falls with it, so this is 'the surface has stopped moving'.")]
        [SerializeField, Range(0.1f, 0.5f)] private float confidentWeight = 0.3f;
        [Tooltip("Boundary loops shorter than this (metres) are ignored: a few-centimetre pit is not a leak.")]
        [SerializeField, Range(0.05f, 1f)] private float holeMinPerimeter = 0.2f;
        [Tooltip("Closure = 1 / (1 + openBoundary / (ref × √meshArea)). 0.3 puts 50 % at an open boundary of 0.3·√A metres (2.8 m for a 90 m² room) and 92 % at about one 8 cm hole.")]
        [SerializeField, Range(0.05f, 1f)] private float closureReference = 0.3f;
        [Tooltip("Boundary edges within this many voxels of a clip plane, the room AABB or the volume edge are cuts, not holes.")]
        [SerializeField, Range(1f, 4f)] private float cutToleranceVoxels = 2f;
        [Tooltip("Min-label link + pointer-jump rounds, one per frame. 12 converges loops of a few thousand edges.")]
        [SerializeField, Range(4, 32)] private int closureLinkRounds = 12;

        /// <summary>Number of voxels near the zero-crossing with sufficient weight (surface voxels).</summary>
        public int SurfaceVoxelCount { get; private set; }
        /// <summary>Number of surface voxels that are frozen (user-confirmed done).</summary>
        public int FrozenSurfaceCount { get; private set; }
        /// <summary>Number of surface voxels with camera color data (alpha &gt; 0.1).</summary>
        public int ColoredSurfaceCount { get; private set; }
        /// <summary>Surface voxels at or above <c>confidentWeight</c>.</summary>
        public int ConfidentSurfaceCount { get; private set; }
        /// <summary>Confident ÷ surface (0–1): how much of the surface has stopped moving.</summary>
        public float Refinement { get; private set; }
        /// <summary>Mesh closure from the last analysis cycle.</summary>
        public MeshClosure Closure { get; private set; }
        /// <summary>True once a cycle has completed for the current scan.</summary>
        public bool AnalysisAvailable { get; private set; }

        /// <summary>
        /// Extra torso capsules (legacy <see cref="RoomScanner.AddExclusionZone"/>).
        /// Head / hands are <see cref="HeadAnchor"/> / hand anchors, not this list.
        /// </summary>
        public readonly List<Transform> ExclusionZones = new();
        private readonly Vector4[] _exclusionP0 = new Vector4[BodyExclusion.Max];
        private readonly Vector4[] _exclusionP1 = new Vector4[BodyExclusion.Max];

        /// <summary>Head (or center-eye) used to build the torso capsule. World up, not head.up.</summary>
        public Transform HeadAnchor { get; set; }
        /// <summary>Left wrist or controller. Null skips that side's hand/forearm capsules.</summary>
        public Transform LeftHandAnchor { get; set; }
        /// <summary>Right wrist or controller. Null skips that side's hand/forearm capsules.</summary>
        public Transform RightHandAnchor { get; set; }
        /// <summary>
        /// When true, <see cref="RoomScanner"/> will not overwrite the three
        /// anchors from the camera rig each frame (host already wired them).
        /// </summary>
        public bool BodyAnchorsHostOwned { get; set; }

        /// <summary>
        /// Pin head and wrist transforms used to pack exclusion capsules.
        /// Pass <paramref name="hostOwned"/> true when the host refreshes
        /// Capsense / controller wrists itself.
        /// </summary>
        public void SetBodyExclusionAnchors(Transform head, Transform leftHand, Transform rightHand, bool hostOwned)
        {
            HeadAnchor = head;
            LeftHandAnchor = leftHand;
            RightHandAnchor = rightHand;
            BodyAnchorsHostOwned = hostOwned;
        }

        public const int MaxRoomClipPlanes = 32;
        public const int MaxScreenStamps = 4;

        private readonly Vector4[] _roomClipPlanes = new Vector4[MaxRoomClipPlanes];
        private readonly Vector4[] _screenCenter = new Vector4[MaxScreenStamps];
        private readonly Vector4[] _screenInward = new Vector4[MaxScreenStamps];
        private readonly Vector4[] _screenAxis = new Vector4[MaxScreenStamps];
        private readonly Vector4[] _screenBitangent = new Vector4[MaxScreenStamps];
        private int _roomClipCount;
        private int _screenStampCount;
        private bool _confineToRoom;
        private bool _useRoomAabb;
        private Vector3 _roomAabbMin;
        private Vector3 _roomAabbMax;

        /// <summary>Total number of integration passes dispatched since startup or the last clear.</summary>
        public int IntegrationCount { get; private set; }
        public int WarmupIntegrations => warmupIntegrations;

        /// <summary>Raised after each integration compute dispatch (before pruning).</summary>
        public event Action Integrated;
        /// <summary>Raised after the volume is cleared.</summary>
        public event Action Cleared;

        private Texture _pendingCamFrame;
        private Vector3 _pendingCamPos;
        private Quaternion _pendingCamRot;
        private Vector2 _pendingFocalLen;
        private Vector2 _pendingPrincipalPt;
        private Vector2 _pendingSensorRes;
        private Vector2 _pendingCurrentRes;
        private RenderTexture _camFrameCopy;
        private Texture2D _dummyCamTex;

        private void Awake()
        {
            Instance = this;
            // GPU resources allocate lazily on the first scan / save / full-load
            // path via ReallocateVolumes(). The lightweight LoadRefinedOnlyAsync
            // path (returning-player and editor-sim) never touches them, so a
            // pure replay session avoids the ~150 MB TSDF+color RT footprint.
        }

        private void Start()
        {
            // Intentionally empty — see Awake().
            //
            // Historic note: kernel helpers + 3D RTs used to be constructed
            // here unconditionally. They're now created on demand inside
            // ReallocateVolumes(), which is called by:
            //   * RoomScanner.StartScanning() (every scan begin)
            //   * RoomScanPersistence.SaveToNewPackageAsync (defensive)
            //   * RoomScanPersistence.LoadPackageAsync (full TSDF reload)
        }

        /// <summary>
        /// Build all compute-kernel helpers and bind them to the current
        /// <see cref="_volume"/> / <see cref="_colorVolume"/>. Idempotent —
        /// the first <see cref="ReallocateVolumes"/> call constructs them;
        /// subsequent allocations only need <see cref="RebindKernelTextures"/>.
        /// </summary>
        private void InitKernels()
        {
            _clearKernel = new ComputeKernelHelper(compute, "Clear");
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);

            _integrateKernel = new ComputeKernelHelper(compute, "Integrate");
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);

            _stampKernel = new ComputeKernelHelper(compute, "StampScreen");
            _stampKernel.Set(VolumeRWID, _volume);
            _stampKernel.Set(ColorVolumeRWID, _colorVolume);

            _pruneKernel = new ComputeKernelHelper(compute, "Prune");
            _pruneKernel.Set(VolumeRWID, _volume);
            _pruneKernel.Set(ColorVolumeRWID, _colorVolume);

            _freezeKernel = new ComputeKernelHelper(compute, "FreezeInFrustum");
            _freezeKernel.Set(VolumeRWID, _volume);

            _unfreezeKernel = new ComputeKernelHelper(compute, "UnfreezeInFrustum");
            _unfreezeKernel.Set(VolumeRWID, _volume);

            _eraseKernel = new ComputeKernelHelper(compute, "EraseInsideExclusions");
            _eraseKernel.Set(VolumeRWID, _volume);
            _eraseKernel.Set(ColorVolumeRWID, _colorVolume);

            _coverageKernel = new ComputeKernelHelper(compute, "CountSurfaceCoverage");
            _coverageKernel.Set(VolumeRWID, _volume);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);
            InitAnalysisBuffers();

            _shellMarchKernel = new ComputeKernelHelper(compute, "MarchShellCells");
            _shellMarchKernel.Set(VolumeRWID, _volume);
            _fillPatchKernel = new ComputeKernelHelper(compute, "FillShellPatch");
            _fillPatchKernel.Set(VolumeRWID, _volume);
            _closeHolesKernel = new ComputeKernelHelper(compute, "CloseSmallHoles");
            _closeHolesKernel.Set(VolumeRWID, _volume);
            _shellPos ??= new ComputeBuffer(ShellCellSet.MaxCells, sizeof(float) * 4);
            _shellNrm ??= new ComputeBuffer(ShellCellSet.MaxCells, sizeof(float) * 4);
            _shellResult ??= new ComputeBuffer(ShellCellSet.MaxCells, sizeof(uint));
            _shellMarchKernel.Set(CoverCellPosID, _shellPos);
            _shellMarchKernel.Set(CoverCellNrmID, _shellNrm);
            _shellMarchKernel.Set(CoverResultID, _shellResult);

            _dummyCamTex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _dummyCamTex.SetPixel(0, 0, Color.black);
            _dummyCamTex.Apply(false, true);
        }

        private void OnDestroy()
        {
            ReleaseVolumes();
            ReleaseAnalysisBuffers();
            _shellPos?.Release();
            _shellNrm?.Release();
            _shellResult?.Release();
            _shellPos = _shellNrm = _shellResult = null;
            if (_camFrameCopy) Destroy(_camFrameCopy);
            if (_dummyCamTex) Destroy(_dummyCamTex);
        }

        /// <summary>
        /// Destroys the TSDF + color volume RenderTextures and the frustum buffer to free GPU memory.
        /// The component stays alive; calling <see cref="CreateVolume"/> + <see cref="SetShaderConstants"/>
        /// re-allocates everything (handled transparently by the integration path).
        /// </summary>
        public void ReleaseVolumes()
        {
            _frustumVolume?.Release();
            _frustumVolume = null;
            _frustumReady = false;
            if (_volume) { Destroy(_volume); _volume = null; }
            if (_colorVolume) { Destroy(_colorVolume); _colorVolume = null; }
            IntegrationCount = 0;
            Logger.Info("VolumeIntegrator: GPU volumes released");
        }

        /// <summary>True when volumes have been released and need re-allocation before integration.</summary>
        public bool VolumesReleased => _volume == null;

        /// <summary>
        /// Allocate (or re-allocate) TSDF + color volumes and bring kernels +
        /// shader constants up to date. Idempotent — early-returns if volumes
        /// already exist. Handles three scenarios:
        /// <list type="bullet">
        ///   <item><description><b>First-ever scan</b>: builds compute kernels
        ///   from scratch (deferred from the old eager <c>Awake</c>/<c>Start</c>
        ///   path), allocates RTs, sets globals.</description></item>
        ///   <item><description><b>Resume after <see cref="ReleaseVolumes"/></b>:
        ///   re-allocates RTs and rebinds existing kernels via
        ///   <see cref="RebindKernelTextures"/>.</description></item>
        ///   <item><description><b>Already allocated</b>: no-op.</description></item>
        /// </list>
        /// Called by <see cref="RoomScanner.StartScanningAsync"/> and the heavy
        /// <c>RoomScanPersistence</c> save/full-load paths. The lightweight
        /// <c>LoadRefinedOnlyAsync</c> path intentionally skips this.
        /// </summary>
        public void ReallocateVolumes()
        {
            if (_volume != null) return;

            // ComputeKernelHelper is a struct — use its readonly Shader
            // backing field as the "never initialized" sentinel.
            bool firstAlloc = (_clearKernel.Shader == null);

            CreateVolume();

            if (firstAlloc) InitKernels();
            else            RebindKernelTextures();

            SetShaderConstants();
            Clear();

            if (DepthCapture.Instance != null)
                DepthCapture.Instance.SetVoxelParams(voxelDistance, voxelSize);

            Logger.Info(firstAlloc
                ? "VolumeIntegrator: GPU resources allocated lazily on first scan/save/full-load."
                : "VolumeIntegrator: GPU volumes re-allocated after release.");
        }

        private void RebindKernelTextures()
        {
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _integrateKernel.Set(VolumeRWID, _volume);
            _integrateKernel.Set(ColorVolumeRWID, _colorVolume);
            _stampKernel.Set(VolumeRWID, _volume);
            _stampKernel.Set(ColorVolumeRWID, _colorVolume);
            _pruneKernel.Set(VolumeRWID, _volume);
            _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
            _freezeKernel.Set(VolumeRWID, _volume);
            _unfreezeKernel.Set(VolumeRWID, _volume);
            _eraseKernel.Set(VolumeRWID, _volume);
            _eraseKernel.Set(ColorVolumeRWID, _colorVolume);
            _coverageKernel.Set(VolumeRWID, _volume);
            compute.SetTexture(_coverageKernel.KernelIndex, ColorVolumeReadID, _colorVolume);
            _shellMarchKernel.Set(VolumeRWID, _volume);
            _fillPatchKernel.Set(VolumeRWID, _volume);
            _closeHolesKernel.Set(VolumeRWID, _volume);
        }

        // ── Scan analysis cycle ─────────────────────────────────────────

        void InitAnalysisBuffers()
        {
            const int cap = GPUSurfaceNets.MaxOpenEdges;
            _analysisResult ??= new ComputeBuffer(AnalysisWords, sizeof(uint));
            _ccEdgesSnap ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, cap, 16);
            _ccCountersSnap ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, GPUSurfaceNets.CounterCount, 4);
            _ccHashKey ??= new ComputeBuffer(CcHashSize, sizeof(uint));
            _ccHashVal ??= new ComputeBuffer(CcHashSize, sizeof(uint));
            _ccLabel ??= new ComputeBuffer(cap, sizeof(uint));
            _ccComp ??= new ComputeBuffer(cap * 4, sizeof(int));
            _ccCountersSnap.SetData(new uint[GPUSurfaceNets.CounterCount]);

            _ccResetKernel = new ComputeKernelHelper(compute, "ClosureReset");
            _ccInsertKernel = new ComputeKernelHelper(compute, "ClosureInsert");
            _ccLinkKernel = new ComputeKernelHelper(compute, "ClosureLink");
            _ccJumpKernel = new ComputeKernelHelper(compute, "ClosureJump");
            _ccClassifyKernel = new ComputeKernelHelper(compute, "ClosureClassify");
            _ccFinalizeKernel = new ComputeKernelHelper(compute, "ClosureFinalize");
            _ccCentroidKernel = new ComputeKernelHelper(compute, "ClosureCentroid");

            _coverageKernel.Set(AnalysisResultID, _analysisResult);
            foreach (var k in new[] { _ccResetKernel, _ccInsertKernel, _ccLinkKernel, _ccJumpKernel,
                         _ccClassifyKernel, _ccFinalizeKernel, _ccCentroidKernel })
            {
                k.Set(AnalysisResultID, _analysisResult);
                k.Set(CcEdgesID, _ccEdgesSnap);
                k.Set(CcCountersID, _ccCountersSnap);
                k.Set(CcHashKeyID, _ccHashKey);
                k.Set(CcHashValID, _ccHashVal);
                k.Set(CcLabelID, _ccLabel);
                k.Set(CcCompID, _ccComp);
            }
            compute.SetInt(CcCapID, cap);
            compute.SetInt(CcHashSizeID, CcHashSize);
            compute.SetInt(CcHashMaskID, CcHashSize - 1);
        }

        void ReleaseAnalysisBuffers()
        {
            _analysisResult?.Release();
            _ccEdgesSnap?.Release();
            _ccCountersSnap?.Release();
            _ccHashKey?.Release();
            _ccHashVal?.Release();
            _ccLabel?.Release();
            _ccComp?.Release();
            _analysisResult = null;
            _ccEdgesSnap = _ccCountersSnap = null;
            _ccHashKey = _ccHashVal = _ccLabel = _ccComp = null;
        }

        /// <summary>Forget the last cycle's numbers (scan start).</summary>
        public void ResetAnalysis()
        {
            AnalysisAvailable = false;
            _analysisState = AnalysisState.Idle;
            _analysisExtractSeen = -1;
            _lastAnalysisTime = 0f;
            SurfaceVoxelCount = FrozenSurfaceCount = ColoredSurfaceCount = ConfidentSurfaceCount = 0;
            Refinement = 0f;
            Closure = default;
        }

        /// <summary>
        /// Advance the analysis cycle by one frame. Call once per frame while
        /// scanning, after the extract for that frame has been issued.
        /// <para>
        /// Idle → (tick due, fresh extract) reset scratch, count surface voxels,
        /// GPU-copy the Surface Nets boundary edges + counters into the
        /// snapshot, hash-insert → <c>closureLinkRounds</c> frames of link +
        /// jump → classify → finalize + centroid → one readback of the 32-word
        /// result. The shell march rides the same tick.
        /// </para>
        /// </summary>
        internal void StepAnalysis(GPUSurfaceNets surfaceNets, int extractCount)
        {
            if (_volume == null || _analysisResult == null) return;

            switch (_analysisState)
            {
                case AnalysisState.Idle:
                {
                    if (Time.time - _lastAnalysisTime < analysisIntervalSeconds) return;
                    if (surfaceNets == null || surfaceNets.OpenEdgeBuffer == null) return;
                    if (extractCount == _analysisExtractSeen) return; // no new mesh since last cycle
                    _analysisExtractSeen = extractCount;

                    compute.SetFloat(ConfidentWeightID, confidentWeight);
                    compute.SetInt(CcMinEdgesID, Mathf.Max(1, Mathf.RoundToInt(holeMinPerimeter / voxelSize)));
                    compute.SetFloat(CcCutTolID, cutToleranceVoxels * voxelSize);
                    BindScanPriors(compute);

                    _ccResetKernel.DispatchFit(CcHashSize, 1);
                    _coverageKernel.Set(VolumeRWID, _volume);
                    _coverageKernel.DispatchFit(_volume);
                    Graphics.CopyBuffer(surfaceNets.OpenEdgeBuffer, _ccEdgesSnap);
                    Graphics.CopyBuffer(surfaceNets.CountersBuffer, _ccCountersSnap);
                    _ccInsertKernel.DispatchFit(GPUSurfaceNets.MaxOpenEdges, 1);
                    DispatchShellMarch();

                    _analysisRound = 0;
                    _analysisState = AnalysisState.Link;
                    break;
                }
                case AnalysisState.Link:
                {
                    _ccLinkKernel.DispatchFit(GPUSurfaceNets.MaxOpenEdges, 1);
                    _ccJumpKernel.DispatchFit(GPUSurfaceNets.MaxOpenEdges, 1);
                    if (++_analysisRound >= closureLinkRounds)
                        _analysisState = AnalysisState.Classify;
                    break;
                }
                case AnalysisState.Classify:
                {
                    _ccJumpKernel.DispatchFit(GPUSurfaceNets.MaxOpenEdges, 1);
                    _ccClassifyKernel.DispatchFit(GPUSurfaceNets.MaxOpenEdges, 1);
                    _analysisState = AnalysisState.Finalize;
                    break;
                }
                case AnalysisState.Finalize:
                {
                    _ccFinalizeKernel.DispatchFit(GPUSurfaceNets.MaxOpenEdges, 1);
                    _ccCentroidKernel.DispatchFit(1, 1);
                    AsyncGPUReadback.Request(_analysisResult, OnAnalysisReadback);
                    _analysisState = AnalysisState.Readback;
                    break;
                }
                case AnalysisState.Readback:
                    break; // waiting on the GPU
            }
        }

        private void OnAnalysisReadback(AsyncGPUReadbackRequest request)
        {
            _analysisState = AnalysisState.Idle;
            _lastAnalysisTime = Time.time;
            if (request.hasError || _analysisResult == null) return;
            var data = request.GetData<uint>();
            if (data.Length < AnalysisWords) return;
            NativeArray<uint>.Copy(data, 0, _analysisCpu, 0, AnalysisWords);
            var r = _analysisCpu;

            SurfaceVoxelCount = (int)r[0];
            FrozenSurfaceCount = (int)r[1];
            ColoredSurfaceCount = (int)r[2];
            ConfidentSurfaceCount = (int)r[3];
            Refinement = SurfaceVoxelCount > 0 ? (float)ConfidentSurfaceCount / SurfaceVoxelCount : 0f;

            int openTotal = (int)r[4];
            int openUsed = (int)r[5];
            int sigEdges = (int)r[8];
            // Edges past the snapshot capacity were never analysed; count
            // them as open boundary so an overflowing frontier reads low.
            sigEdges += Mathf.Max(0, openTotal - openUsed);
            uint packed = r[10];
            int largestEdges = (int)(packed >> 16);
            Vector3 largestCenter = new Vector3(
                (int)r[11] * 0.001f, (int)r[12] * 0.001f, (int)r[13] * 0.001f);
            int quads = (int)r[14] / 6;
            float area = quads * voxelSize * voxelSize;
            float openMetres = sigEdges * voxelSize;
            float reference = Mathf.Max(closureReference * Mathf.Sqrt(Mathf.Max(area, 0f)), 4f * voxelSize);
            float closure = area > 0f ? 1f / (1f + openMetres / reference) : 0f;

            Closure = new MeshClosure(
                closure, openMetres, area, (int)r[9], (int)r[7], (int)r[6], openTotal,
                new MeshHole(largestCenter, largestEdges * voxelSize, largestEdges));
            AnalysisAvailable = true;
        }

        // ── Shell coverage ──────────────────────────────────────────────

        /// <summary>
        /// Upload the captured-shell cells for the coverage march. Pass
        /// count 0 to disable. Arrays are read up to <paramref name="count"/>.
        /// </summary>
        public void SetShellCells(Vector4[] pos, Vector4[] nrm, int count)
        {
            if (_shellPos == null || _shellNrm == null)
            {
                _shellCount = 0;
                return;
            }
            count = Mathf.Clamp(count, 0, ShellCellSet.MaxCells);
            if (count > 0)
            {
                _shellPos.SetData(pos, 0, 0, count);
                _shellNrm.SetData(nrm, 0, 0, count);
            }
            _shellCount = count;
            ShellGeneration++;
        }

        public void ClearShellCells()
        {
            _shellCount = 0;
            ShellGeneration++;
        }

        void DispatchShellMarch()
        {
            if (_shellCount <= 0 || _shellReadbackPending || _shellResult == null
                || _shellMarchKernel.Shader == null)
                return;
            _shellReadbackPending = true;
            _shellReadbackGeneration = ShellGeneration;
            compute.SetInt(CoverCellCountID, _shellCount);
            compute.SetFloat(CoverMinWeightID, minMeshWeight);
            _shellMarchKernel.Set(VolumeRWID, _volume);
            _shellMarchKernel.DispatchFit(_shellCount, 1);
            AsyncGPUReadback.Request(_shellResult, _shellCount * sizeof(uint), 0, OnShellReadback);
        }

        private void OnShellReadback(AsyncGPUReadbackRequest request)
        {
            _shellReadbackPending = false;
            if (request.hasError || _shellCount <= 0) return;
            if (_shellReadbackGeneration != ShellGeneration) return; // cells rebuilt while in flight
            var data = request.GetData<uint>();
            int n = Mathf.Min(data.Length, _shellCount);
            if (n <= 0) return;
            NativeArray<uint>.Copy(data, 0, _shellResultCpu, 0, n);
            ShellResultReady?.Invoke(_shellResultCpu, n, _shellReadbackGeneration);
        }

        /// <summary>One auto-fill dispatch: a soft plane stamp or a furniture close, over a small voxel box.</summary>
        public struct ShellFillRequest
        {
            public bool Close;
            public Vector3 Center;
            public Vector3 Inward;
            public Vector3 Axis;
            public Vector3 Bitangent;
            public float HalfW;
            public float HalfH;
            public Vector3 BoxMin;
            public Vector3 BoxMax;
        }

        /// <summary>Dispatch up to <paramref name="count"/> fills. Returns how many ran.</summary>
        public int ApplyShellFills(ShellFillRequest[] requests, int count)
        {
            if (_volume == null || requests == null || count <= 0) return 0;
            if (_fillPatchKernel.Shader == null || _closeHolesKernel.Shader == null) return 0;

            compute.SetFloat(FillWeightID, fillWeight);
            compute.SetFloat(CoverMinWeightID, minMeshWeight);
            int applied = 0;
            for (int i = 0; i < count; i++)
            {
                var r = requests[i];
                if (r.Close)
                {
                    if (!closeFurnitureHoles) continue;
                    if (!TryVoxelBox(r.BoxMin, r.BoxMax, out int mx, out int my, out int mz, out int sx, out int sy, out int sz))
                        continue;
                    compute.SetInts(StampVoxMinID, mx, my, mz);
                    compute.SetInts(StampVoxMaxID, mx + sx, my + sy, mz + sz);
                    _closeHolesKernel.Set(VolumeRWID, _volume);
                    _closeHolesKernel.DispatchFit(sx, sy, sz);
                }
                else
                {
                    if (!autoFillShellGaps) continue;
                    Vector3 bmin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
                    Vector3 bmax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
                    for (int u = -1; u <= 1; u += 2)
                    for (int v = -1; v <= 1; v += 2)
                    for (int w = -1; w <= 1; w += 2)
                    {
                        Vector3 corner = r.Center + r.Axis * (u * (r.HalfW + voxelSize))
                            + r.Bitangent * (v * (r.HalfH + voxelSize))
                            + r.Inward * (w * voxelDistance);
                        bmin = Vector3.Min(bmin, corner);
                        bmax = Vector3.Max(bmax, corner);
                    }
                    if (!TryVoxelBox(bmin, bmax, out int mx, out int my, out int mz, out int sx, out int sy, out int sz))
                        continue;
                    compute.SetVector(FillCenterID, new Vector4(r.Center.x, r.Center.y, r.Center.z, r.HalfW));
                    compute.SetVector(FillInwardID, new Vector4(r.Inward.x, r.Inward.y, r.Inward.z, r.HalfH));
                    compute.SetVector(FillAxisID, r.Axis);
                    compute.SetVector(FillBitangentID, r.Bitangent);
                    compute.SetInts(StampVoxMinID, mx, my, mz);
                    compute.SetInts(StampVoxMaxID, mx + sx, my + sy, mz + sz);
                    _fillPatchKernel.Set(VolumeRWID, _volume);
                    _fillPatchKernel.DispatchFit(sx, sy, sz);
                }
                applied++;
            }
            return applied;
        }

        bool TryVoxelBox(Vector3 worldMin, Vector3 worldMax,
            out int minX, out int minY, out int minZ, out int sx, out int sy, out int sz)
        {
            Vector3 vmin = WorldToVoxelFloat(worldMin);
            Vector3 vmax = WorldToVoxelFloat(worldMax);
            minX = Mathf.Clamp(Mathf.FloorToInt(vmin.x) - 1, 0, voxelCount.x);
            minY = Mathf.Clamp(Mathf.FloorToInt(vmin.y) - 1, 0, voxelCount.y);
            minZ = Mathf.Clamp(Mathf.FloorToInt(vmin.z) - 1, 0, voxelCount.z);
            int maxX = Mathf.Clamp(Mathf.CeilToInt(vmax.x) + 1, 0, voxelCount.x);
            int maxY = Mathf.Clamp(Mathf.CeilToInt(vmax.y) + 1, 0, voxelCount.y);
            int maxZ = Mathf.Clamp(Mathf.CeilToInt(vmax.z) + 1, 0, voxelCount.z);
            sx = maxX - minX;
            sy = maxY - minY;
            sz = maxZ - minZ;
            return sx > 0 && sy > 0 && sz > 0;
        }

        private void CreateVolume()
        {
            long tsdfBytes = (long)voxelCount.x * voxelCount.y * voxelCount.z * 2;
            long colorBytes = (long)voxelCount.x * voxelCount.y * voxelCount.z * 4;
            Logger.Info($"TSDF volume: {voxelCount} RG8_SNorm = {tsdfBytes / (1024 * 1024)}MB");
            Logger.Info($"Color volume: {voxelCount} RGBA8_UNorm = {colorBytes / (1024 * 1024)}MB");

            _volume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8G8_SNorm, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _volume.Create();

            _colorVolume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8G8B8A8_UNorm, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _colorVolume.Create();
        }

        private void SetShaderConstants()
        {
            int3 s = voxelCount;
            compute.SetInts(VoxCountID, s.x, s.y, s.z);
            Shader.SetGlobalVector(VoxCountID, new Vector4(s.x, s.y, s.z, 0));

            compute.SetFloat(VoxSizeID, voxelSize);
            Shader.SetGlobalFloat(VoxSizeID, voxelSize);

            compute.SetFloat(VoxMinID, voxelMin);
            compute.SetFloat(VoxDistID, voxelDistance);
            Shader.SetGlobalFloat(VoxDistID, voxelDistance);

            compute.SetFloat(DepthDispThreshID, depthDisparityThreshold);
            compute.SetFloat(MaxUpdateDistID, maxUpdateDist);
            compute.SetFloat(BlendRateID, blendRate);
            compute.SetFloat(StabilityID, stability);
            compute.SetFloat(WeightGrowthID, weightGrowth);
            compute.SetFloat(MaxWeightID, maxWeight);

            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);
        }

        /// <summary>
        /// Zeros the TSDF and color volumes on the GPU. No-op if volumes
        /// haven't been allocated yet (lazy alloc — see
        /// <see cref="ReallocateVolumes"/>).
        /// </summary>
        public void Clear()
        {
            if (_volume == null || _clearKernel.Shader == null) return;
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.Set(ColorVolumeRWID, _colorVolume);
            _clearKernel.DispatchFit(_volume);
            Cleared?.Invoke();
        }

        /// <summary>
        /// Resample TSDF + color from the current (relocated) grid into a new identity grid.
        /// After this call the volume data lives in the current tracking/world frame.
        /// </summary>
        public void BakeRelocation(Matrix4x4 relocationMatrix)
        {
            if (_volume == null || _colorVolume == null || compute == null)
                return;

            Matrix4x4 invRelocation = relocationMatrix.inverse;
            int3 vc = voxelCount;

            var dstTsdf = new RenderTexture(vc.x, vc.y, 0, _volume.graphicsFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            dstTsdf.Create();

            var dstColor = new RenderTexture(vc.x, vc.y, 0, _colorVolume.graphicsFormat, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = vc.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            dstColor.Create();

            int kernel = compute.FindKernel("BakeRelocation");
            compute.SetInts(Shader.PropertyToID("gsVoxCount"), vc.x, vc.y, vc.z);
            compute.SetFloat(Shader.PropertyToID("gsVoxSize"), voxelSize);
            compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcTsdf"), _volume);
            compute.SetTexture(kernel, Shader.PropertyToID("gsBakeSrcColor"), _colorVolume);
            compute.SetTexture(kernel, VolumeRWID, dstTsdf);
            compute.SetTexture(kernel, ColorVolumeRWID, dstColor);
            compute.SetMatrix(Shader.PropertyToID("gsBakeInvRelocation"), invRelocation);

            int tx = Mathf.CeilToInt(vc.x / 4f);
            int ty = Mathf.CeilToInt(vc.y / 4f);
            int tz = Mathf.CeilToInt(vc.z / 4f);
            compute.Dispatch(kernel, tx, ty, tz);
            GL.Flush();

            // Swap volumes: destroy old, adopt baked textures.
            // Avoids Graphics.CopyTexture on 3D RTs which can silently fail on Vulkan/Quest.
            Destroy(_volume);
            Destroy(_colorVolume);
            _volume = dstTsdf;
            _colorVolume = dstColor;

            // Rebind global texture references (used by render shader for freeze tint etc.)
            Shader.SetGlobalTexture(VolumeID, _volume);
            Shader.SetGlobalTexture(ColorVolumeID, _colorVolume);

            // Rebind per-kernel UAV references so subsequent integrations/clears use new textures
            RebindVolumeTextures();

            Logger.Info($"BakeRelocation complete — resampled {vc} voxels, " +
                      $"reloc row0={relocationMatrix.GetRow(0)}, inv row0={invRelocation.GetRow(0)}");
        }

        private void RebindVolumeTextures()
        {
            if (_clearKernel.Shader == null) return;
            RebindKernelTextures();
        }

        /// <summary>
        /// Freeze all voxels currently visible in the camera frustum.
        /// Frozen voxels are encoded as negative weight and skip integration.
        /// Requires camera data to have been provided via SetCameraData.
        /// </summary>
        public void FreezeInView(Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (_volume == null || _freezeKernel.Shader == null)
            {
                Logger.Warning("FreezeInView called before GPU resources allocated; ignored.");
                return;
            }
            SetFrustumCameraUniforms(_freezeKernel, camPos, camRot,
                focalLen, principalPt, sensorRes, currentRes);
            BindExclusionUniforms(compute);
            _freezeKernel.Set(VolumeRWID, _volume);
            _freezeKernel.DispatchFit(_volume);
            Logger.Info("FreezeInView dispatched");
        }

        /// <summary>
        /// Unfreeze all frozen voxels currently visible in the camera frustum.
        /// </summary>
        public void UnfreezeInView(Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (_volume == null || _unfreezeKernel.Shader == null)
            {
                Logger.Warning("UnfreezeInView called before GPU resources allocated; ignored.");
                return;
            }
            SetFrustumCameraUniforms(_unfreezeKernel, camPos, camRot,
                focalLen, principalPt, sensorRes, currentRes);
            _unfreezeKernel.Set(VolumeRWID, _volume);
            _unfreezeKernel.DispatchFit(_volume);
            Logger.Info("UnfreezeInView dispatched");
        }

        private void SetFrustumCameraUniforms(ComputeKernelHelper kernel, Vector3 camPos,
            Quaternion camRot, Vector2 focalLen, Vector2 principalPt,
            Vector2 sensorRes, Vector2 currentRes)
        {
            compute.SetVector(CamPosID, camPos);
            compute.SetMatrix(CamInvRotID, Matrix4x4.Rotate(Quaternion.Inverse(camRot)));
            compute.SetVector(CamFocalLenID, focalLen);
            compute.SetVector(CamPrincipalPtID, principalPt);
            compute.SetVector(CamSensorResID, sensorRes);
            compute.SetVector(CamCurrentResID, currentRes);
        }

        /// <summary>
        /// Provide a camera frame and intrinsics for color integration this tick.
        /// Uses direct pinhole projection (matching Meta PCA samples) instead of VP matrix.
        /// Call before Integrate() each frame. Pass null frame to skip color.
        /// </summary>
        public void SetCameraData(Texture frame, Vector3 camPos, Quaternion camRot,
            Vector2 focalLength, Vector2 principalPoint, Vector2 sensorRes, Vector2 currentRes)
        {
            _pendingCamFrame = frame;
            _pendingCamPos = camPos;
            _pendingCamRot = camRot;
            _pendingFocalLen = focalLength;
            _pendingPrincipalPt = principalPoint;
            _pendingSensorRes = sensorRes;
            _pendingCurrentRes = currentRes;
        }

        /// <summary>
        /// Ensures _camFrameCopy exists and blits the pending frame to it.
        /// Called internally before Integrate() uses it for compute shader color integration.
        /// </summary>
        private void EnsureCamFrameCopy()
        {
            if (_pendingCamFrame == null) return;
            int w = _pendingCamFrame.width;
            int h = _pendingCamFrame.height;
            if (_camFrameCopy == null || _camFrameCopy.width != w || _camFrameCopy.height != h)
            {
                if (_camFrameCopy) Destroy(_camFrameCopy);
                _camFrameCopy = new RenderTexture(w, h, 0, GraphicsFormat.R8G8B8A8_SRGB, 0)
                {
                    enableRandomWrite = false,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                _camFrameCopy.Create();
            }
            Graphics.Blit(_pendingCamFrame, _camFrameCopy);
        }

        /// <summary>
        /// Builds the frustum sample positions buffer used by the Integrate kernel.
        /// Called lazily on first integration or after a volume clear/load.
        /// </summary>
        public void SetupFrustumVolume()
        {
            if (!DepthCapture.DepthAvailable) return;

            Matrix4x4 depthProj = Shader.GetGlobalMatrixArray(DepthCapture.ProjID)[0];
            FrustumPlanes frustum = depthProj.decomposeProjection;
            frustum.zFar = maxUpdateDist;

            var positions = new List<Vector3>(Mathf.Min(maxFrustumPositions, 200000));

            float ls = frustum.left / frustum.zNear;
            float rs = frustum.right / frustum.zNear;
            float ts = frustum.top / frustum.zNear;
            float bs = frustum.bottom / frustum.zNear;

            float step = voxelSize;
            bool capped = false;

            for (float z = frustum.zNear; z < frustum.zFar && !capped; z += step)
            {
                float xMin = ls * z + step;
                float xMax = rs * z - step;
                float yMin = bs * z + step;
                float yMax = ts * z - step;

                for (float x = xMin; x < xMax && !capped; x += step)
                for (float y = yMin; y < yMax; y += step)
                {
                    var v = new Vector3(x, y, -z);
                    float mag = v.magnitude;
                    if (mag > minUpdateDist && mag < maxUpdateDist)
                    {
                        positions.Add(v);
                        if (positions.Count >= maxFrustumPositions)
                        {
                            capped = true;
                            break;
                        }
                    }
                }
            }

            if (positions.Count == 0) return;

            Logger.Info($"Frustum volume: {positions.Count} positions ({positions.Count * 12 / 1024}KB)");

            _frustumVolume?.Release();
            _frustumVolume = new ComputeBuffer(positions.Count, sizeof(float) * 3);
            _frustumVolume.SetData(positions);
            _integrateKernel.Set(FrustumVolumeID, _frustumVolume);
            _frustumReady = true;
        }

        /// <summary>
        /// Dispatches one TSDF + color integration pass from the current depth frame.
        /// Handles frustum setup, exclusion zones, warmup clearing, and periodic pruning.
        /// </summary>
        public void Integrate()
        {
            var dc = DepthCapture.Instance;
            if (dc == null || !DepthCapture.DepthAvailable || dc.DepthTex == null) return;
            // Defensive: with lazy GPU alloc a stray Integrate() before
            // ReallocateVolumes can land here. RoomScanner.StartScanning()
            // always calls ReallocateVolumes first, so this is just a
            // safety net.
            if (_volume == null || _integrateKernel.Shader == null) return;
            if (!_frustumReady) SetupFrustumVolume();
            if (!_frustumReady) return;

            dc.UpdateDilationIfNeeded();

            compute.SetMatrixArray(DepthCapture.ViewID, dc.View);
            compute.SetMatrixArray(DepthCapture.ProjID, dc.Proj);
            compute.SetMatrixArray(DepthCapture.ViewInvID, dc.ViewInv);
            compute.SetMatrixArray(DepthCapture.ProjInvID, dc.ProjInv);

            BindExclusionUniforms(compute);
            BindScanPriors(compute);

            compute.SetFloat(BlendRateID, blendRate);
            compute.SetFloat(StabilityID, stability);
            compute.SetFloat(WeightGrowthID, weightGrowth);
            compute.SetFloat(MaxWeightID, maxWeight);

            BindCameraToKernel(_integrateKernel.KernelIndex);

            _integrateKernel.Set(DepthCapture.DepthTexID, dc.DepthTex);
            _integrateKernel.Set(DepthCapture.NormTexID, dc.NormTex);
            _integrateKernel.Set(DepthCapture.DilatedDepthTexID, dc.DilatedDepthTex);

            _integrateKernel.DispatchFit(_frustumVolume.count, 1);
            DispatchScreenStamps();
            DispatchEraseBodyBlobs();

            IntegrationCount++;
            _pendingCamFrame = null;

            if (warmupIntegrations > 0 && IntegrationCount == warmupIntegrations)
            {
                Logger.Info($"Warmup complete ({warmupIntegrations} frames), clearing volume to discard sensor startup noise");
                Clear();
            }

            float t = Time.time;
            if (t - _lastPruneTime >= pruneIntervalSeconds)
            {
                _lastPruneTime = t;
                _pruneKernel.Set(VolumeRWID, _volume);
                _pruneKernel.Set(ColorVolumeRWID, _colorVolume);
                _pruneKernel.DispatchFit(_volume);
            }

            Integrated?.Invoke();
        }

        void BindCameraToKernel(int kernelIndex)
        {
            EnsureCamFrameCopy();
            if (_pendingCamFrame != null && _camFrameCopy != null)
            {
                compute.SetTexture(kernelIndex, CamRGBID, _camFrameCopy);
                compute.SetInt(CamAvailableID, 1);
                compute.SetVector(CamPosID, _pendingCamPos);
                compute.SetMatrix(CamInvRotID, Matrix4x4.Rotate(Quaternion.Inverse(_pendingCamRot)));
                compute.SetVector(CamFocalLenID, _pendingFocalLen);
                compute.SetVector(CamPrincipalPtID, _pendingPrincipalPt);
                compute.SetVector(CamSensorResID, _pendingSensorRes);
                compute.SetVector(CamCurrentResID, _pendingCurrentRes);
                compute.SetFloat(CamExposureID, cameraExposure);
            }
            else
            {
                compute.SetTexture(kernelIndex, CamRGBID, _dummyCamTex);
                compute.SetInt(CamAvailableID, 0);
            }
        }

        /// <summary>
        /// Stamp SCREEN slabs as a 3D dispatch over each stamp's voxel AABB
        /// after Integrate so the analytic plane wins over glass depth.
        /// The frustum pass no longer searches for TVs.
        /// </summary>
        void DispatchScreenStamps()
        {
            if (_screenStampCount <= 0 || _stampKernel.Shader == null) return;
            _stampKernel.Set(VolumeRWID, _volume);
            _stampKernel.Set(ColorVolumeRWID, _colorVolume);
            BindCameraToKernel(_stampKernel.KernelIndex);
            for (int i = 0; i < _screenStampCount; i++)
            {
                if (!TryScreenStampVoxelBox(i, out int minX, out int minY, out int minZ,
                        out int sx, out int sy, out int sz))
                    continue;
                compute.SetInts(StampVoxMinID, minX, minY, minZ);
                compute.SetInts(StampVoxMaxID, minX + sx, minY + sy, minZ + sz);
                _stampKernel.DispatchFit(sx, sy, sz);
            }
        }

        bool TryScreenStampVoxelBox(
            int i,
            out int minX, out int minY, out int minZ,
            out int sx, out int sy, out int sz)
        {
            minX = minY = minZ = sx = sy = sz = 0;
            Vector3 c = _screenCenter[i];
            float ht = _screenCenter[i].w;
            Vector3 n = _screenInward[i];
            float hw = _screenInward[i].w;
            Vector3 tangent = _screenAxis[i];
            float hh = _screenAxis[i].w;
            Vector3 bitangent = _screenBitangent[i];

            Vector3 vmin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 vmax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int u = -1; u <= 1; u += 2)
            for (int v = -1; v <= 1; v += 2)
            for (int w = -1; w <= 1; w += 2)
            {
                Vector3 corner = c + n * (u * ht) + tangent * (v * hw) + bitangent * (w * hh);
                Vector3 vf = WorldToVoxelFloat(corner);
                vmin = Vector3.Min(vmin, vf);
                vmax = Vector3.Max(vmax, vf);
            }

            minX = Mathf.Clamp(Mathf.FloorToInt(vmin.x) - 1, 0, voxelCount.x);
            minY = Mathf.Clamp(Mathf.FloorToInt(vmin.y) - 1, 0, voxelCount.y);
            minZ = Mathf.Clamp(Mathf.FloorToInt(vmin.z) - 1, 0, voxelCount.z);
            int maxX = Mathf.Clamp(Mathf.CeilToInt(vmax.x) + 1, 0, voxelCount.x);
            int maxY = Mathf.Clamp(Mathf.CeilToInt(vmax.y) + 1, 0, voxelCount.y);
            int maxZ = Mathf.Clamp(Mathf.CeilToInt(vmax.z) + 1, 0, voxelCount.z);
            sx = maxX - minX;
            sy = maxY - minY;
            sz = maxZ - minZ;
            return sx > 0 && sy > 0 && sz > 0;
        }

        Vector3 WorldToVoxelFloat(Vector3 worldPos)
            => worldPos / voxelSize
            + new Vector3(voxelCount.x, voxelCount.y, voxelCount.z) * 0.5f;

        /// <summary>
        /// Upload room-clip half-spaces and SCREEN plane stamps for the next
        /// <see cref="Integrate"/> (and for <see cref="BindScanPriors"/> on
        /// other compute shaders that include <c>VolumeHelpers.hlsl</c>).
        /// Pass <paramref name="confine"/> false to keep unbounded TSDF;
        /// SCREEN stamps still apply. Null lists clear that buffer.
        /// <paramref name="useRoomAabb"/> is a conservative world AABB
        /// padded by the same 50 cm outward expand as the clip planes.
        /// </summary>
        public void SetScanPriors(
            bool confine,
            List<Vector4> clipPlanes,
            List<ScanScreenStamp> screenStamps,
            bool useRoomAabb = false,
            Vector3 roomAabbMin = default,
            Vector3 roomAabbMax = default)
        {
            _confineToRoom = confine;
            _roomClipCount = 0;
            if (clipPlanes != null)
            {
                int n = Mathf.Min(clipPlanes.Count, MaxRoomClipPlanes);
                for (int i = 0; i < n; i++)
                    _roomClipPlanes[i] = clipPlanes[i];
                for (int i = n; i < MaxRoomClipPlanes; i++)
                    _roomClipPlanes[i] = Vector4.zero;
                _roomClipCount = n;
            }
            else
            {
                for (int i = 0; i < MaxRoomClipPlanes; i++)
                    _roomClipPlanes[i] = Vector4.zero;
            }

            _screenStampCount = 0;
            if (screenStamps != null)
            {
                int n = Mathf.Min(screenStamps.Count, MaxScreenStamps);
                for (int i = 0; i < n; i++)
                {
                    var s = screenStamps[i];
                    _screenCenter[i] = new Vector4(
                        s.Center.x, s.Center.y, s.Center.z, s.HalfThickness);
                    _screenInward[i] = new Vector4(
                        s.Inward.x, s.Inward.y, s.Inward.z, s.HalfWidth);
                    _screenAxis[i] = new Vector4(
                        s.Tangent.x, s.Tangent.y, s.Tangent.z, s.HalfHeight);
                    _screenBitangent[i] = new Vector4(
                        s.Bitangent.x, s.Bitangent.y, s.Bitangent.z, 0f);
                }
                for (int i = n; i < MaxScreenStamps; i++)
                {
                    _screenCenter[i] = Vector4.zero;
                    _screenInward[i] = Vector4.zero;
                    _screenAxis[i] = Vector4.zero;
                    _screenBitangent[i] = Vector4.zero;
                }
                _screenStampCount = n;
            }
            else
            {
                for (int i = 0; i < MaxScreenStamps; i++)
                {
                    _screenCenter[i] = Vector4.zero;
                    _screenInward[i] = Vector4.zero;
                    _screenAxis[i] = Vector4.zero;
                    _screenBitangent[i] = Vector4.zero;
                }
            }

            _useRoomAabb = useRoomAabb && confine && _roomClipCount > 0;
            _roomAabbMin = roomAabbMin;
            _roomAabbMax = roomAabbMax;
        }

        /// <summary>Drop clip planes and SCREEN stamps (scan stop / unload).</summary>
        public void ClearScanPriors()
            => SetScanPriors(false, null, null);

        /// <summary>
        /// Bind the last <see cref="SetScanPriors"/> upload onto any compute
        /// shader that includes <c>VolumeHelpers.hlsl</c> (Integrate, triplanar
        /// bake). No-op when <paramref name="target"/> is null.
        /// </summary>
        public void BindScanPriors(ComputeShader target)
        {
            if (target == null) return;
            target.SetInt(ConfineToRoomID, _confineToRoom && _roomClipCount > 0 ? 1 : 0);
            target.SetInt(NumRoomClipPlanesID, _roomClipCount);
            target.SetVectorArray(RoomClipPlanesID, _roomClipPlanes);
            target.SetInt(NumScreenStampsID, _screenStampCount);
            target.SetVectorArray(ScreenCenterID, _screenCenter);
            target.SetVectorArray(ScreenInwardID, _screenInward);
            target.SetVectorArray(ScreenAxisID, _screenAxis);
            target.SetVectorArray(ScreenBitangentID, _screenBitangent);
            target.SetInt(UseRoomAabbID, _useRoomAabb ? 1 : 0);
            target.SetVector(RoomAabbMinID, _roomAabbMin);
            target.SetVector(RoomAabbMaxID, _roomAabbMax);
        }

        /// <summary>
        /// Pack current head / hand / extra-zone capsules and upload to any
        /// compute shader that includes <c>VolumeHelpers.hlsl</c>.
        /// </summary>
        public void BindExclusionUniforms(ComputeShader target)
        {
            if (target == null) return;
            int n = BodyExclusion.Pack(
                _exclusionP0, _exclusionP1,
                HeadAnchor, LeftHandAnchor, RightHandAnchor, ExclusionZones,
                torsoRadius, torsoAbove, torsoBelow,
                handRadius, handHalfLength,
                forearmRadius, forearmLength,
                shoulderDrop, shoulderLateral);
            target.SetInt(NumExclusionsID, n);
            target.SetVectorArray(ExclusionP0ID, _exclusionP0);
            target.SetVectorArray(ExclusionP1ID, _exclusionP1);
            target.SetInt(EraseBodyID, eraseBodyBlobs ? 1 : 0);
            target.SetFloat(EraseMaxWeightID, eraseMaxWeight);
        }

        void DispatchEraseBodyBlobs()
        {
            if (!eraseBodyBlobs || _volume == null || _eraseKernel.Shader == null) return;
            _eraseKernel.Set(VolumeRWID, _volume);
            _eraseKernel.Set(ColorVolumeRWID, _colorVolume);
            _eraseKernel.DispatchFit(_volume);
        }

        /// <summary>
        /// Uploads CPU TSDF/color blobs into the 3D RenderTextures.
        /// Uses <see cref="GraphicsFormat"/> matching the volume RTs so
        /// <see cref="Graphics.CopyTexture"/> is valid on Metal/Vulkan (RG16 Texture3D ≠ R8G8_SNorm layout).
        /// </summary>
        public bool LoadVolumes(byte[] tsdfBytes, byte[] colorBytes, int integrationCount)
        {
            if (_volume == null || _colorVolume == null)
            {
                Logger.Error("Cannot load volumes: textures not created");
                return false;
            }

            int3 s = voxelCount;
            int expectedTsdf = s.x * s.y * s.z * 2;
            int expectedColor = s.x * s.y * s.z * 4;

            if (tsdfBytes.Length != expectedTsdf)
            {
                Logger.Error($"TSDF size mismatch: got {tsdfBytes.Length}, expected {expectedTsdf}");
                return false;
            }
            if (colorBytes.Length != expectedColor)
            {
                Logger.Error($"Color volume size mismatch: got {colorBytes.Length}, expected {expectedColor}");
                return false;
            }

            // Must match CreateVolume(): R8G8_SNorm TSDF + RGBA8_UNorm color
            var tsdfTex = new Texture3D(s.x, s.y, s.z, GraphicsFormat.R8G8_SNorm, TextureCreationFlags.None);
            tsdfTex.SetPixelData(tsdfBytes, 0);
            tsdfTex.Apply(false, false);
            Graphics.CopyTexture(tsdfTex, _volume);
            Destroy(tsdfTex);

            var colorTex = new Texture3D(s.x, s.y, s.z, GraphicsFormat.R8G8B8A8_UNorm, TextureCreationFlags.None);
            colorTex.SetPixelData(colorBytes, 0);
            colorTex.Apply(false, false);
            Graphics.CopyTexture(colorTex, _colorVolume);
            Destroy(colorTex);

            GL.Flush();

            IntegrationCount = integrationCount;
            _frustumReady = false;

            Logger.Info($"Volumes loaded: {s}, integrationCount={integrationCount}");
            return true;
        }
    }
}
