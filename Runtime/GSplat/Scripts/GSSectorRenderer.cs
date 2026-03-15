using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders Gaussian splats via compute prepass -> GPU sort -> draw.
    /// All compute work (prepass + sort) executes inside the URP render pass
    /// command buffer (like UGS) so camera matrices are from the late-latched
    /// pose. Covariance uses mono MV/P; clip positions are per-eye.
    /// For stereo, writes 2 SplatViewData entries per splat (L+R).
    /// Front-to-back sort + Blend OneMinusDstAlpha One.
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

        GSplatBuffers _serverTrainedBuffers;

        int _preparedTotalCount;

        public bool HasSplatsReady => _preparedTotalCount > 0;

        GraphicsBuffer _viewDataBuffer;
        int _viewDataCapacity;

        GraphicsBuffer _sortKeysBuffer;
        GraphicsBuffer _sortPayloadBuffer;
        int _sortCapacity;
        GpuRadixSort _radixSort;
        GpuRadixSort.Resources _radixResources;
        bool _useRadixSort;

        GraphicsBuffer _bitonicSortBuffer;
        int _bitonicCapacity;

        int _prepassKernel = -1;
        int _initSortKernel = -1;
        int _initSortBitonicKernel = -1;
        int _bitonicStepKernel = -1;

        static readonly int ID_NumPoints       = Shader.PropertyToID("_NumPoints");
        static readonly int ID_WriteOffset     = Shader.PropertyToID("_WriteOffset");
        static readonly int ID_SplatSize       = Shader.PropertyToID("_SplatSize");
        static readonly int ID_SHDegree        = Shader.PropertyToID("_SHDegree");
        static readonly int ID_IsStereo        = Shader.PropertyToID("_IsStereo");
        static readonly int ID_ViewMatrix      = Shader.PropertyToID("_ViewMatrix");
        static readonly int ID_ProjMatrix      = Shader.PropertyToID("_ProjMatrix");
        static readonly int ID_VPMatrixLeft    = Shader.PropertyToID("_VPMatrixLeft");
        static readonly int ID_VPMatrixRight   = Shader.PropertyToID("_VPMatrixRight");
        static readonly int ID_VecScreenParams = Shader.PropertyToID("_VecScreenParams");
        static readonly int ID_CamPos          = Shader.PropertyToID("_CamPos");
        static readonly int ID_Means           = Shader.PropertyToID("_Means");
        static readonly int ID_Scales          = Shader.PropertyToID("_Scales");
        static readonly int ID_Quats           = Shader.PropertyToID("_Quats");
        static readonly int ID_FeaturesDC      = Shader.PropertyToID("_FeaturesDC");
        static readonly int ID_FeaturesRest    = Shader.PropertyToID("_FeaturesRest");
        static readonly int ID_Opacities       = Shader.PropertyToID("_Opacities");
        static readonly int ID_ViewData        = Shader.PropertyToID("_ViewData");
        static readonly int ID_SplatViewData   = Shader.PropertyToID("_SplatViewData");
        static readonly int ID_SplatCount      = Shader.PropertyToID("_SplatCount");
        static readonly int ID_EyeIndex        = Shader.PropertyToID("_EyeIndex");
        static readonly int ID_OrderBuffer     = Shader.PropertyToID("_OrderBuffer");
        static readonly int ID_SortKeys        = Shader.PropertyToID("_SortKeys");
        static readonly int ID_SortPayload     = Shader.PropertyToID("_SortPayload");
        static readonly int ID_Count           = Shader.PropertyToID("_Count");
        static readonly int ID_PaddedCount     = Shader.PropertyToID("_PaddedCount");
        static readonly int ID_Stride          = Shader.PropertyToID("_Stride");
        static readonly int ID_StepSize        = Shader.PropertyToID("_StepSize");
        static readonly int ID_GroupSize       = Shader.PropertyToID("_GroupSize");
        static readonly int ID_SortBuffer      = Shader.PropertyToID("_SortBuffer");

        public Material SplatMaterial
        {
            get => splatMaterial;
            set => splatMaterial = value;
        }

        /// <summary>
        /// Sets server-trained whole-room Gaussian buffers for rendering.
        /// When set, these take priority over per-sector training buffers.
        /// </summary>
        public void SetServerTrainedBuffers(GSplatBuffers buffers)
        {
            _serverTrainedBuffers = buffers;
            Debug.Log($"[GSSectorRenderer] Server-trained buffers set: {buffers?.CurrentCount ?? 0} Gaussians");
        }

        void OnEnable() => ActiveInstance = this;

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
            Debug.Log($"[GSSectorRenderer] Initialized, sort={(_useRadixSort ? "radix" : "bitonic")}");
        }

        void EnsureBuffers(int totalCount, bool stereo)
        {
            int viewEntries = stereo ? totalCount * 2 : totalCount;
            if (_viewDataBuffer == null || _viewDataCapacity < viewEntries)
            {
                _viewDataBuffer?.Release();
                _viewDataCapacity = Mathf.Max(viewEntries, 1024);
                _viewDataBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _viewDataCapacity, 40);
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

        /// <summary>
        /// Called each frame from LateUpdate to count ready splats.
        /// Prefers server-trained whole-room buffers when available.
        /// Buffer allocation is done here; actual compute is deferred to PrepareAndSort.
        /// </summary>
        void LateUpdate()
        {
            _preparedTotalCount = 0;
            if (splatMaterial == null || _prepassKernel < 0)
                return;

            var cam = Camera.main;
            if (cam == null) return;

            int totalCount;

            // Prefer server-trained whole-room buffers
            if (_serverTrainedBuffers != null && _serverTrainedBuffers.CurrentCount > 0)
            {
                _readySectors.Clear();
                _readySectors.Add((-1, _serverTrainedBuffers));
                totalCount = _serverTrainedBuffers.CurrentCount;
            }
            else
            {
                if (!_ready || _scheduler == null) return;

                _scheduler.GetSplatReadySectors(_readySectors);
                if (_readySectors.Count == 0) return;

                totalCount = 0;
                foreach (var (_, buf) in _readySectors)
                    totalCount += buf.CurrentCount;
                if (totalCount <= 0) return;
            }

            bool isStereo = XRSettings.enabled && cam.stereoEnabled;
            EnsureBuffers(totalCount, isStereo);

            _preparedTotalCount = totalCount;
        }

        /// <summary>
        /// Records prepass + sort dispatches into the command buffer.
        /// Called from <see cref="GSplatRenderFeature"/> at render time so
        /// all camera matrices are from the late-latched pose.
        /// </summary>
        public void PrepareAndSort(CommandBuffer cmd, Camera cam, bool isStereo,
                                   Matrix4x4 vpLeft, Matrix4x4 vpRight)
        {
            if (_preparedTotalCount <= 0 || _prepassKernel < 0) return;

            int totalCount = _preparedTotalCount;
            int stride = isStereo ? 2 : 1;

            if (!_loggedFirstDraw)
            {
                _loggedFirstDraw = true;
                Debug.Log($"[GSSectorRenderer] First draw: {_readySectors.Count} sector(s), " +
                          $"{totalCount} gaussians, stereo={isStereo}");
            }

            // Mono view matrix (Unity convention: -Z forward)
            Matrix4x4 monoView = cam.worldToCameraMatrix;
            Matrix4x4 monoProj = cam.projectionMatrix;

            float screenW = isStereo ? XRSettings.eyeTextureWidth : cam.pixelWidth;
            float screenH = isStereo ? XRSettings.eyeTextureHeight : cam.pixelHeight;

            // --- Phase 1: Prepass (command buffer dispatch) ---
            cmd.SetComputeFloatParam(viewPrepassCompute, ID_SplatSize, splatSizeMultiplier);
            cmd.SetComputeIntParam(viewPrepassCompute, ID_SHDegree, shDegree);
            cmd.SetComputeIntParam(viewPrepassCompute, ID_IsStereo, isStereo ? 1 : 0);
            cmd.SetComputeMatrixParam(viewPrepassCompute, ID_ViewMatrix, monoView);
            cmd.SetComputeMatrixParam(viewPrepassCompute, ID_ProjMatrix, monoProj);
            cmd.SetComputeMatrixParam(viewPrepassCompute, ID_VPMatrixLeft, vpLeft);
            cmd.SetComputeMatrixParam(viewPrepassCompute, ID_VPMatrixRight, isStereo ? vpRight : vpLeft);
            cmd.SetComputeVectorParam(viewPrepassCompute, ID_VecScreenParams,
                new Vector4(screenW, screenH, 0, 0));
            cmd.SetComputeVectorParam(viewPrepassCompute, ID_CamPos,
                (Vector4)cam.transform.position);
            cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel,
                ID_ViewData, _viewDataBuffer);

            int writeOffset = 0;
            foreach (var (_, buffers) in _readySectors)
            {
                int n = buffers.CurrentCount;
                if (n <= 0) continue;

                cmd.SetComputeIntParam(viewPrepassCompute, ID_NumPoints, n);
                cmd.SetComputeIntParam(viewPrepassCompute, ID_WriteOffset, writeOffset);
                cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel, ID_Means, buffers.Means);
                cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel, ID_Scales, buffers.Scales);
                cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel, ID_Quats, buffers.Quats);
                cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel, ID_FeaturesDC, buffers.FeaturesDC);
                cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel, ID_FeaturesRest, buffers.FeaturesRest);
                cmd.SetComputeBufferParam(viewPrepassCompute, _prepassKernel, ID_Opacities, buffers.Opacities);

                int groups = (n + 255) / 256;
                cmd.DispatchCompute(viewPrepassCompute, _prepassKernel, groups, 1, 1);

                writeOffset += n * stride;
            }

            // --- Phase 2: Sort ---
            if (_useRadixSort)
                DispatchRadixSort(cmd, totalCount, stride);
            else if (_initSortBitonicKernel >= 0 && _bitonicStepKernel >= 0)
                DispatchBitonicSort(cmd, totalCount, stride);

            // --- Phase 3: Bind on material ---
            splatMaterial.SetBuffer(ID_SplatViewData, _viewDataBuffer);
            if (_useRadixSort)
                splatMaterial.SetBuffer(ID_OrderBuffer, _sortPayloadBuffer);
            else
                splatMaterial.SetBuffer(ID_SortBuffer, _bitonicSortBuffer);
            splatMaterial.SetInt(ID_SplatCount, totalCount);
        }

        /// <summary>
        /// Draws the splats for one eye. Called per-eye from the render pass.
        /// </summary>
        public void DrawSplats(CommandBuffer cmd, int eyeIndex, bool isStereo)
        {
            if (_preparedTotalCount <= 0 || splatMaterial == null) return;

            _props.SetInteger(ID_EyeIndex, eyeIndex);
            _props.SetInteger(ID_IsStereo, isStereo ? 1 : 0);

            cmd.DrawProcedural(
                Matrix4x4.identity, splatMaterial, 0,
                MeshTopology.Triangles, 6, _preparedTotalCount, _props);
        }

        void DispatchRadixSort(CommandBuffer cmd, int totalCount, int stride)
        {
            int threadGroups = (totalCount + 255) / 256;

            cmd.SetComputeIntParam(sortCompute, ID_Count, totalCount);
            cmd.SetComputeIntParam(sortCompute, ID_PaddedCount, totalCount);
            cmd.SetComputeIntParam(sortCompute, ID_Stride, stride);
            cmd.SetComputeBufferParam(sortCompute, _initSortKernel, ID_ViewData, _viewDataBuffer);
            cmd.SetComputeBufferParam(sortCompute, _initSortKernel, ID_SortKeys, _sortKeysBuffer);
            cmd.SetComputeBufferParam(sortCompute, _initSortKernel, ID_SortPayload, _sortPayloadBuffer);
            cmd.DispatchCompute(sortCompute, _initSortKernel, threadGroups, 1, 1);

            _radixSort.Dispatch(cmd, totalCount, _sortKeysBuffer, _sortPayloadBuffer, _radixResources);
        }

        void DispatchBitonicSort(CommandBuffer cmd, int totalCount, int stride)
        {
            int padded = Mathf.NextPowerOfTwo(totalCount);
            int threadGroups = (padded + 255) / 256;

            cmd.SetComputeIntParam(sortCompute, ID_Count, totalCount);
            cmd.SetComputeIntParam(sortCompute, ID_PaddedCount, padded);
            cmd.SetComputeIntParam(sortCompute, ID_Stride, stride);
            cmd.SetComputeBufferParam(sortCompute, _initSortBitonicKernel, ID_ViewData, _viewDataBuffer);
            cmd.SetComputeBufferParam(sortCompute, _initSortBitonicKernel, ID_SortBuffer, _bitonicSortBuffer);
            cmd.DispatchCompute(sortCompute, _initSortBitonicKernel, threadGroups, 1, 1);

            cmd.SetComputeBufferParam(sortCompute, _bitonicStepKernel, ID_SortBuffer, _bitonicSortBuffer);
            for (int k = 2; k <= padded; k <<= 1)
            {
                for (int j = k >> 1; j >= 1; j >>= 1)
                {
                    cmd.SetComputeIntParam(sortCompute, ID_StepSize, j);
                    cmd.SetComputeIntParam(sortCompute, ID_GroupSize, k);
                    cmd.DispatchCompute(sortCompute, _bitonicStepKernel, threadGroups, 1, 1);
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
