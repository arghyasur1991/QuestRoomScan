using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders Gaussian splats via compute prepass → GPU sort → lightweight render.
    /// Zero-copy from training buffers. All sectors concatenated into one draw call.
    /// Bitonic sort provides back-to-front ordering for correct alpha compositing.
    /// </summary>
    public class GSSectorRenderer : MonoBehaviour
    {
        [SerializeField] Material splatMaterial;
        [SerializeField] ComputeShader viewPrepassCompute;
        [SerializeField] ComputeShader sortCompute;
        [SerializeField, Range(0.1f, 4f)] float splatSizeMultiplier = 1f;
        [SerializeField] int shDegree = 2;

        SectorScheduler _scheduler;
        MaterialPropertyBlock _props;
        readonly List<(int id, GSplatBuffers buffers)> _readySectors = new();
        bool _ready;
        bool _loggedFirstDraw;

        GraphicsBuffer _viewDataBuffer;
        int _viewDataCapacity;
        GraphicsBuffer _sortBuffer;
        int _sortBufferCapacity;

        int _prepassKernel = -1;
        int _initSortKernel = -1;
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
        static readonly int ID_SortBuffer    = Shader.PropertyToID("_SortBuffer");
        static readonly int ID_Count         = Shader.PropertyToID("_Count");
        static readonly int ID_PaddedCount   = Shader.PropertyToID("_PaddedCount");
        static readonly int ID_StepSize      = Shader.PropertyToID("_StepSize");
        static readonly int ID_GroupSize     = Shader.PropertyToID("_GroupSize");

        public Material SplatMaterial
        {
            get => splatMaterial;
            set => splatMaterial = value;
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
                _bitonicStepKernel = sortCompute.FindKernel("CSBitonicStep");
            }
            else
                Debug.LogWarning("[GSSectorRenderer] No sortCompute shader assigned!");

            _ready = true;
            Debug.Log($"[GSSectorRenderer] Initialized, material={splatMaterial?.name ?? "NULL"}, " +
                      $"prepass={viewPrepassCompute?.name ?? "NULL"}, sort={sortCompute?.name ?? "NULL"}");
        }

        void EnsureBuffers(int totalCount)
        {
            if (_viewDataBuffer == null || _viewDataCapacity < totalCount)
            {
                _viewDataBuffer?.Release();
                _viewDataCapacity = Mathf.Max(totalCount, 1024);
                _viewDataBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _viewDataCapacity, 40);
            }

            int padded = Mathf.NextPowerOfTwo(totalCount);
            if (_sortBuffer == null || _sortBufferCapacity < padded)
            {
                _sortBuffer?.Release();
                _sortBufferCapacity = padded;
                _sortBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, _sortBufferCapacity, 8);
            }
        }

        void LateUpdate()
        {
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
                          $"{totalCount} total gaussians, splatSize={splatSizeMultiplier}");
            }

            EnsureBuffers(totalCount);

            var camT = cam.transform;
            Matrix4x4 viewPosZ = Matrix4x4.TRS(camT.position, camT.rotation, Vector3.one).inverse;

            float screenW = cam.pixelWidth;
            float screenH = cam.pixelHeight;
            float fx = Mathf.Abs(cam.projectionMatrix[0, 0]) * screenW / 2f;
            float fy = Mathf.Abs(cam.projectionMatrix[1, 1]) * screenH / 2f;

            // --- Phase 1: Compute prepass (per-sector dispatch into shared buffer) ---
            viewPrepassCompute.SetFloat(ID_SplatSize, splatSizeMultiplier);
            viewPrepassCompute.SetInt(ID_SHDegree, shDegree);
            viewPrepassCompute.SetMatrix(ID_ViewMatrix, viewPosZ);
            viewPrepassCompute.SetVector(ID_Focal, new Vector4(fx, fy, 0, 0));
            viewPrepassCompute.SetVector(ID_ScreenSize, new Vector4(screenW, screenH, 0, 0));
            viewPrepassCompute.SetVector(ID_CamPos, (Vector4)camT.position);
            viewPrepassCompute.SetBuffer(_prepassKernel, ID_ViewData, _viewDataBuffer);

            int writeOffset = 0;
            Bounds combinedBounds = default;
            bool first = true;

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

                var bounds = _scheduler.GetSectorBounds(id);
                if (first) { combinedBounds = bounds; first = false; }
                else combinedBounds.Encapsulate(bounds);

                writeOffset += n;
            }

            // --- Phase 2: GPU bitonic sort (back-to-front) ---
            if (_initSortKernel >= 0 && _bitonicStepKernel >= 0)
                DispatchSort(totalCount);

            // --- Phase 3: Render ---
            _props.SetBuffer(ID_SplatViewData, _viewDataBuffer);
            _props.SetBuffer(ID_SortBuffer, _sortBuffer);
            _props.SetInt(ID_SplatCount, totalCount);

            var rp = new RenderParams(splatMaterial)
            {
                worldBounds = combinedBounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            Graphics.RenderPrimitives(rp, MeshTopology.Quads, totalCount * 4);
        }

        void DispatchSort(int totalCount)
        {
            int padded = Mathf.NextPowerOfTwo(totalCount);
            int threadGroups = (padded + 255) / 256;

            // Init sort keys from view data depths
            sortCompute.SetInt(ID_Count, totalCount);
            sortCompute.SetInt(ID_PaddedCount, padded);
            sortCompute.SetBuffer(_initSortKernel, ID_ViewData, _viewDataBuffer);
            sortCompute.SetBuffer(_initSortKernel, ID_SortBuffer, _sortBuffer);
            sortCompute.Dispatch(_initSortKernel, threadGroups, 1, 1);

            // Bitonic sort: O(log²N) passes
            sortCompute.SetBuffer(_bitonicStepKernel, ID_SortBuffer, _sortBuffer);
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
            _ready = false;
            _viewDataBuffer?.Release();
            _viewDataBuffer = null;
            _viewDataCapacity = 0;
            _sortBuffer?.Release();
            _sortBuffer = null;
            _sortBufferCapacity = 0;
        }
    }
}
