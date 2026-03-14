using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders Gaussian splats via compute prepass → GPU sort → URP render pass.
    /// Zero-copy from training buffers. All sectors concatenated into one draw call.
    /// Uses radix sort (13 dispatches) with bitonic fallback for Quest compatibility.
    ///
    /// The prepass outputs world-space positions so the vertex shader can apply
    /// the late-latched per-eye VP matrix at render time, eliminating VR jitter.
    /// Drawing is deferred to <see cref="GSplatRenderFeature"/> which handles
    /// per-eye RT slice targeting required for Quest stereo.
    /// </summary>
    public class GSSectorRenderer : MonoBehaviour
    {
        [SerializeField] Material splatMaterial;
        [SerializeField] ComputeShader viewPrepassCompute;
        [SerializeField] ComputeShader sortCompute;
        [SerializeField] ComputeShader radixSortCompute;
        [SerializeField, Range(0.1f, 4f)] float splatSizeMultiplier = 1f;
        [SerializeField] int shDegree = 2;

        public static GSSectorRenderer ActiveInstance { get; private set; }

        SectorScheduler _scheduler;
        MaterialPropertyBlock _props;
        readonly List<(int id, GSplatBuffers buffers)> _readySectors = new();
        bool _ready;
        bool _loggedFirstDraw;

        int _preparedTotalCount;
        bool _preparedIsStereo;

        /// <summary>True when LateUpdate has prepared splat data for the current frame.</summary>
        public bool HasPreparedSplats => _preparedTotalCount > 0;

        /// <summary>Whether the current frame's prepared data is for stereo rendering.</summary>
        public bool IsStereoMode => _preparedIsStereo;

        GraphicsBuffer _viewDataBuffer;
        int _viewDataCapacity;

        // Radix sort buffers (separate key/payload)
        GraphicsBuffer _sortKeysBuffer;
        GraphicsBuffer _sortPayloadBuffer;
        int _sortCapacity;
        GpuRadixSort _radixSort;
        GpuRadixSort.Resources _radixResources;
        bool _useRadixSort;

        // Bitonic sort fallback
        GraphicsBuffer _bitonicSortBuffer;
        int _bitonicCapacity;

        int _prepassKernel = -1;
        int _initSortKernel = -1;
        int _initSortBitonicKernel = -1;
        int _bitonicStepKernel = -1;

        static readonly int ID_NumPoints     = Shader.PropertyToID("_NumPoints");
        static readonly int ID_WriteOffset   = Shader.PropertyToID("_WriteOffset");
        static readonly int ID_SplatSize     = Shader.PropertyToID("_SplatSize");
        static readonly int ID_SHDegree      = Shader.PropertyToID("_SHDegree");
        static readonly int ID_ViewMatrix    = Shader.PropertyToID("_ViewMatrix");
        static readonly int ID_Focal         = Shader.PropertyToID("_Focal");
        static readonly int ID_ScreenSize    = Shader.PropertyToID("_ScreenSize");
        static readonly int ID_CamPos        = Shader.PropertyToID("_CamPos");
        static readonly int ID_Means         = Shader.PropertyToID("_Means");
        static readonly int ID_Scales        = Shader.PropertyToID("_Scales");
        static readonly int ID_Quats         = Shader.PropertyToID("_Quats");
        static readonly int ID_FeaturesDC    = Shader.PropertyToID("_FeaturesDC");
        static readonly int ID_FeaturesRest  = Shader.PropertyToID("_FeaturesRest");
        static readonly int ID_Opacities     = Shader.PropertyToID("_Opacities");
        static readonly int ID_ViewData      = Shader.PropertyToID("_ViewData");
        static readonly int ID_SplatViewData = Shader.PropertyToID("_SplatViewData");
        static readonly int ID_SplatCount    = Shader.PropertyToID("_SplatCount");
        static readonly int ID_OrderBuffer   = Shader.PropertyToID("_OrderBuffer");
        static readonly int ID_SortKeys      = Shader.PropertyToID("_SortKeys");
        static readonly int ID_SortPayload   = Shader.PropertyToID("_SortPayload");
        static readonly int ID_Count         = Shader.PropertyToID("_Count");
        static readonly int ID_PaddedCount   = Shader.PropertyToID("_PaddedCount");
        static readonly int ID_StepSize      = Shader.PropertyToID("_StepSize");
        static readonly int ID_GroupSize     = Shader.PropertyToID("_GroupSize");
        static readonly int ID_SortBuffer    = Shader.PropertyToID("_SortBuffer");
        static readonly int ID_SplatVP = Shader.PropertyToID("_SplatVP");
        static readonly int ID_SplatView = Shader.PropertyToID("_SplatView");
        static readonly int ID_SplatFocal = Shader.PropertyToID("_SplatFocal");
        static readonly int ID_SplatScreen = Shader.PropertyToID("_SplatScreen");

        public Material SplatMaterial
        {
            get => splatMaterial;
            set => splatMaterial = value;
        }

        void OnEnable()
        {
            ActiveInstance = this;
        }

        public void Initialize(SectorScheduler scheduler)
        {
            _scheduler = scheduler;
            _props = new MaterialPropertyBlock();

            if (viewPrepassCompute != null)
                _prepassKernel = viewPrepassCompute.FindKernel("CSCalcViewData");
            else
                Debug.LogWarning("[GSSectorRenderer] No viewPrepassCompute shader assigned!");

            if (sortCompute != null)
            {
                _initSortKernel = sortCompute.FindKernel("CSInitSortKeys");
                _initSortBitonicKernel = sortCompute.FindKernel("CSInitSortKeysBitonic");
                _bitonicStepKernel = sortCompute.FindKernel("CSBitonicStep");
            }

            // Try radix sort first
            if (radixSortCompute != null)
            {
                _radixSort = new GpuRadixSort(radixSortCompute);
                _useRadixSort = _radixSort.Valid;
            }

            if (!_useRadixSort)
                Debug.LogWarning("[GSSectorRenderer] Radix sort unavailable, using bitonic fallback");

            if (splatMaterial != null)
            {
                if (_useRadixSort)
                    splatMaterial.EnableKeyword("_SORT_RADIX");
                else
                    splatMaterial.DisableKeyword("_SORT_RADIX");
            }

            _ready = true;
            Debug.Log($"[GSSectorRenderer] Initialized, material={splatMaterial?.name ?? "NULL"}, " +
                      $"prepass={viewPrepassCompute?.name ?? "NULL"}, " +
                      $"sort={(_useRadixSort ? "radix(13 dispatches)" : "bitonic(O(log²N))")}");
        }

        void EnsureBuffers(int totalCount)
        {
            if (_viewDataBuffer == null || _viewDataCapacity < totalCount)
            {
                _viewDataBuffer?.Release();
                _viewDataCapacity = Mathf.Max(totalCount, 1024);
                _viewDataBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _viewDataCapacity, 48);
            }

            if (_useRadixSort)
            {
                if (_sortKeysBuffer == null || _sortCapacity < totalCount)
                {
                    _sortKeysBuffer?.Release();
                    _sortPayloadBuffer?.Release();
                    _radixResources.Dispose();

                    _sortCapacity = Mathf.Max(totalCount, 1024);
                    _sortKeysBuffer = new GraphicsBuffer(
                        GraphicsBuffer.Target.Structured, _sortCapacity, 4) { name = "SplatSortKeys" };
                    _sortPayloadBuffer = new GraphicsBuffer(
                        GraphicsBuffer.Target.Structured, _sortCapacity, 4) { name = "SplatSortPayload" };
                    _radixResources = GpuRadixSort.Resources.Create(_sortCapacity);
                }
            }
            else
            {
                int padded = Mathf.NextPowerOfTwo(totalCount);
                if (_bitonicSortBuffer == null || _bitonicCapacity < padded)
                {
                    _bitonicSortBuffer?.Release();
                    _bitonicCapacity = padded;
                    _bitonicSortBuffer = new GraphicsBuffer(
                        GraphicsBuffer.Target.Structured, _bitonicCapacity, 8);
                }
            }
        }

        void LateUpdate()
        {
            _preparedTotalCount = 0;

            if (!_ready || _scheduler == null || splatMaterial == null || _prepassKernel < 0)
                return;

            var cam = Camera.main;
            if (cam == null) return;

            _scheduler.GetSplatReadySectors(_readySectors);
            if (_readySectors.Count == 0) return;

            int totalCount = 0;
            foreach (var (_, buf) in _readySectors)
                totalCount += buf.CurrentCount;
            if (totalCount <= 0) return;

            if (!_loggedFirstDraw)
            {
                _loggedFirstDraw = true;
                Debug.Log($"[GSSectorRenderer] First draw: {_readySectors.Count} sector(s), " +
                          $"{totalCount} total gaussians, splatSize={splatSizeMultiplier}, " +
                          $"sort={(_useRadixSort ? "radix" : "bitonic")}");
            }

            EnsureBuffers(totalCount);

            var camT = cam.transform;
            Matrix4x4 viewPosZ = Matrix4x4.TRS(camT.position, camT.rotation, Vector3.one).inverse;

            bool isStereo = XRSettings.enabled && cam.stereoEnabled;
            float screenW = isStereo ? XRSettings.eyeTextureWidth : cam.pixelWidth;
            float screenH = isStereo ? XRSettings.eyeTextureHeight : cam.pixelHeight;
            float fx = Mathf.Abs(cam.projectionMatrix[0, 0]) * screenW / 2f;
            float fy = Mathf.Abs(cam.projectionMatrix[1, 1]) * screenH / 2f;

            // --- Phase 1: Compute prepass (world-space output, VP-independent) ---
            viewPrepassCompute.SetFloat(ID_SplatSize, splatSizeMultiplier);
            viewPrepassCompute.SetInt(ID_SHDegree, shDegree);
            viewPrepassCompute.SetMatrix(ID_ViewMatrix, viewPosZ);
            viewPrepassCompute.SetVector(ID_Focal, new Vector4(fx, fy, 0, 0));
            viewPrepassCompute.SetVector(ID_ScreenSize, new Vector4(screenW, screenH, 0, 0));
            viewPrepassCompute.SetVector(ID_CamPos, (Vector4)camT.position);
            viewPrepassCompute.SetBuffer(_prepassKernel, ID_ViewData, _viewDataBuffer);

            int writeOffset = 0;
            foreach (var (id, buffers) in _readySectors)
            {
                int n = buffers.CurrentCount;
                if (n <= 0) continue;

                viewPrepassCompute.SetInt(ID_NumPoints, n);
                viewPrepassCompute.SetInt(ID_WriteOffset, writeOffset);
                viewPrepassCompute.SetBuffer(_prepassKernel, ID_Means, buffers.Means);
                viewPrepassCompute.SetBuffer(_prepassKernel, ID_Scales, buffers.Scales);
                viewPrepassCompute.SetBuffer(_prepassKernel, ID_Quats, buffers.Quats);
                viewPrepassCompute.SetBuffer(_prepassKernel, ID_FeaturesDC, buffers.FeaturesDC);
                viewPrepassCompute.SetBuffer(_prepassKernel, ID_FeaturesRest, buffers.FeaturesRest);
                viewPrepassCompute.SetBuffer(_prepassKernel, ID_Opacities, buffers.Opacities);

                int groups = (n + 255) / 256;
                viewPrepassCompute.Dispatch(_prepassKernel, groups, 1, 1);

                writeOffset += n;
            }

            // --- Phase 2: Sort by view-space depth ---
            if (_useRadixSort)
                DispatchRadixSort(totalCount);
            else if (_initSortBitonicKernel >= 0 && _bitonicStepKernel >= 0)
                DispatchBitonicSort(totalCount);

            // --- Phase 3: Bind buffers on material for the URP render pass ---
            splatMaterial.SetBuffer(ID_SplatViewData, _viewDataBuffer);
            if (_useRadixSort)
                splatMaterial.SetBuffer(ID_OrderBuffer, _sortPayloadBuffer);
            else
                splatMaterial.SetBuffer(ID_SortBuffer, _bitonicSortBuffer);
            splatMaterial.SetInt(ID_SplatCount, totalCount);

            _preparedTotalCount = totalCount;
            _preparedIsStereo = isStereo;
        }

        /// <summary>
        /// Called by <see cref="GSplatRenderFeature"/> from the URP render pass.
        /// Both VP and view matrices are captured at render time (after XR late-latching)
        /// so both splat positions AND covariance axes are world-locked.
        /// </summary>
        public void DrawSplats(CommandBuffer cmd, Matrix4x4 viewProjMatrix,
                               Matrix4x4 viewMatrix, float fx, float fy,
                               float screenW, float screenH)
        {
            if (_preparedTotalCount <= 0 || splatMaterial == null) return;

            _props.SetMatrix(ID_SplatVP, viewProjMatrix);
            _props.SetMatrix(ID_SplatView, viewMatrix);
            _props.SetVector(ID_SplatFocal, new Vector4(fx, fy, 0, 0));
            _props.SetVector(ID_SplatScreen, new Vector4(screenW, screenH, 0, 0));

            cmd.DrawProcedural(
                Matrix4x4.identity, splatMaterial, 0,
                MeshTopology.Quads, _preparedTotalCount * 4, 1, _props);
        }

        void DispatchRadixSort(int totalCount)
        {
            int threadGroups = (totalCount + 255) / 256;

            sortCompute.SetInt(ID_Count, totalCount);
            sortCompute.SetInt(ID_PaddedCount, totalCount);
            sortCompute.SetBuffer(_initSortKernel, ID_ViewData, _viewDataBuffer);
            sortCompute.SetBuffer(_initSortKernel, ID_SortKeys, _sortKeysBuffer);
            sortCompute.SetBuffer(_initSortKernel, ID_SortPayload, _sortPayloadBuffer);
            sortCompute.Dispatch(_initSortKernel, threadGroups, 1, 1);

            _radixSort.Dispatch(totalCount, _sortKeysBuffer, _sortPayloadBuffer, _radixResources);
        }

        void DispatchBitonicSort(int totalCount)
        {
            int padded = Mathf.NextPowerOfTwo(totalCount);
            int threadGroups = (padded + 255) / 256;

            sortCompute.SetInt(ID_Count, totalCount);
            sortCompute.SetInt(ID_PaddedCount, padded);
            sortCompute.SetBuffer(_initSortBitonicKernel, ID_ViewData, _viewDataBuffer);
            sortCompute.SetBuffer(_initSortBitonicKernel, ID_SortBuffer, _bitonicSortBuffer);
            sortCompute.Dispatch(_initSortBitonicKernel, threadGroups, 1, 1);

            sortCompute.SetBuffer(_bitonicStepKernel, ID_SortBuffer, _bitonicSortBuffer);
            for (int k = 2; k <= padded; k <<= 1)
            {
                for (int j = k >> 1; j >= 1; j >>= 1)
                {
                    sortCompute.SetInt(ID_StepSize, j);
                    sortCompute.SetInt(ID_GroupSize, k);
                    sortCompute.Dispatch(_bitonicStepKernel, threadGroups, 1, 1);
                }
            }
        }

        void OnDisable()
        {
            if (ActiveInstance == this) ActiveInstance = null;
            _preparedTotalCount = 0;
            _ready = false;
            _viewDataBuffer?.Release(); _viewDataBuffer = null; _viewDataCapacity = 0;
            _sortKeysBuffer?.Release(); _sortKeysBuffer = null;
            _sortPayloadBuffer?.Release(); _sortPayloadBuffer = null;
            _radixResources.Dispose();
            _sortCapacity = 0;
            _bitonicSortBuffer?.Release(); _bitonicSortBuffer = null; _bitonicCapacity = 0;
        }
    }
}
