using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders Gaussian splats via compute prepass → GPU sort → lightweight render.
    /// Zero-copy from training buffers. All sectors concatenated into one draw call.
    /// Uses radix sort (13 dispatches) with bitonic fallback for Quest compatibility.
    /// </summary>
    public class GSSectorRenderer : MonoBehaviour
    {
        [SerializeField] Material splatMaterial;
        [SerializeField] ComputeShader viewPrepassCompute;
        [SerializeField] ComputeShader sortCompute;
        [SerializeField] ComputeShader radixSortCompute;
        [SerializeField, Range(0.1f, 4f)] float splatSizeMultiplier = 1f;
        [SerializeField] int shDegree = 2;

        SectorScheduler _scheduler;
        MaterialPropertyBlock _props;
        readonly List<(int id, GSplatBuffers buffers)> _readySectors = new();
        bool _ready;
        bool _loggedFirstDraw;

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
        int _initSortKernel = -1;       // radix: writes to _SortKeys/_SortPayload
        int _initSortBitonicKernel = -1; // bitonic: writes to _SortBuffer (uint2)
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
        static readonly int ID_ViewProjMatrix = Shader.PropertyToID("_ViewProjMatrix");
        static readonly int ID_ViewProjMatrixRight = Shader.PropertyToID("_ViewProjMatrixRight");
        static readonly int ID_IsStereo   = Shader.PropertyToID("_IsStereo");
        static readonly int ID_EyeIndex   = Shader.PropertyToID("_EyeIndex");

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

        void EnsureBuffers(int totalCount, bool stereo)
        {
            // View data buffer: 2x for stereo (left + right per Gaussian)
            int viewDataNeeded = stereo ? totalCount * 2 : totalCount;
            if (_viewDataBuffer == null || _viewDataCapacity < viewDataNeeded)
            {
                _viewDataBuffer?.Release();
                _viewDataCapacity = Mathf.Max(viewDataNeeded, 1024);
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
                          $"{totalCount} total gaussians, splatSize={splatSizeMultiplier}, " +
                          $"sort={(_useRadixSort ? "radix" : "bitonic")}");
            }

            // Detect stereo early (needed for buffer sizing)
            bool isStereoEarly = XRSettings.enabled && cam.stereoEnabled;
            EnsureBuffers(totalCount, isStereoEarly);

            var camT = cam.transform;
            Matrix4x4 viewPosZ = Matrix4x4.TRS(camT.position, camT.rotation, Vector3.one).inverse;

            // Detect stereo XR rendering
            bool isStereo = XRSettings.enabled && cam.stereoEnabled;

            float screenW, screenH;
            if (isStereo)
            {
                screenW = XRSettings.eyeTextureWidth;
                screenH = XRSettings.eyeTextureHeight;
            }
            else
            {
                screenW = cam.pixelWidth;
                screenH = cam.pixelHeight;
            }

            float fx = Mathf.Abs(cam.projectionMatrix[0, 0]) * screenW / 2f;
            float fy = Mathf.Abs(cam.projectionMatrix[1, 1]) * screenH / 2f;

            // VP matrices for clip-space output
            Matrix4x4 vpLeft, vpRight;
            if (isStereo)
            {
                var viewL = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
                var projL = cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
                vpLeft = GL.GetGPUProjectionMatrix(projL, true) * viewL;

                var viewR = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
                var projR = cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right);
                vpRight = GL.GetGPUProjectionMatrix(projR, true) * viewR;
            }
            else
            {
                Matrix4x4 gpuProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
                vpLeft = gpuProj * cam.worldToCameraMatrix;
                vpRight = vpLeft;
            }

            // --- Phase 1: Compute prepass ---
            viewPrepassCompute.SetFloat(ID_SplatSize, splatSizeMultiplier);
            viewPrepassCompute.SetInt(ID_SHDegree, shDegree);
            viewPrepassCompute.SetInt(ID_IsStereo, isStereo ? 1 : 0);
            viewPrepassCompute.SetMatrix(ID_ViewMatrix, viewPosZ);
            viewPrepassCompute.SetMatrix(ID_ViewProjMatrix, vpLeft);
            viewPrepassCompute.SetMatrix(ID_ViewProjMatrixRight, vpRight);
            viewPrepassCompute.SetVector(ID_Focal, new Vector4(fx, fy, 0, 0));
            viewPrepassCompute.SetVector(ID_ScreenSize, new Vector4(screenW, screenH, 0, 0));
            viewPrepassCompute.SetVector(ID_CamPos, (Vector4)camT.position);
            viewPrepassCompute.SetBuffer(_prepassKernel, ID_ViewData, _viewDataBuffer);

            // _WriteOffset is in view data entries. In stereo, each Gaussian writes
            // 2 entries (left + right), so the shader computes: _WriteOffset + idx*2 (stereo)
            // or _WriteOffset + idx (mono). C# advances by n*2 or n accordingly.
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

                writeOffset += isStereo ? n * 2 : n;
            }

            // --- Phase 2: Sort (by splat index, same order for both eyes) ---
            if (_useRadixSort)
                DispatchRadixSort(totalCount, isStereo);
            else if (_initSortBitonicKernel >= 0 && _bitonicStepKernel >= 0)
                DispatchBitonicSort(totalCount, isStereo);

            // --- Phase 3: Render ---
            splatMaterial.SetBuffer(ID_SplatViewData, _viewDataBuffer);
            if (_useRadixSort)
                splatMaterial.SetBuffer(ID_OrderBuffer, _sortPayloadBuffer);
            else
                splatMaterial.SetBuffer(ID_SortBuffer, _bitonicSortBuffer);
            splatMaterial.SetInt(ID_SplatCount, totalCount);
            splatMaterial.SetInt(ID_IsStereo, isStereo ? 1 : 0);

            var rp = new RenderParams(splatMaterial)
            {
                worldBounds = combinedBounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            if (isStereo)
            {
                splatMaterial.SetInt(ID_EyeIndex, 0);
                Graphics.RenderPrimitives(rp, MeshTopology.Quads, totalCount * 4);
                splatMaterial.SetInt(ID_EyeIndex, 1);
                Graphics.RenderPrimitives(rp, MeshTopology.Quads, totalCount * 4);
            }
            else
            {
                splatMaterial.SetInt(ID_EyeIndex, 0);
                Graphics.RenderPrimitives(rp, MeshTopology.Quads, totalCount * 4);
            }
        }

        void DispatchRadixSort(int totalCount, bool stereo = false)
        {
            int threadGroups = (totalCount + 255) / 256;

            sortCompute.SetInt(ID_Count, totalCount);
            sortCompute.SetInt(ID_PaddedCount, totalCount);
            sortCompute.SetInt(ID_IsStereo, stereo ? 1 : 0);
            sortCompute.SetBuffer(_initSortKernel, ID_ViewData, _viewDataBuffer);
            sortCompute.SetBuffer(_initSortKernel, ID_SortKeys, _sortKeysBuffer);
            sortCompute.SetBuffer(_initSortKernel, ID_SortPayload, _sortPayloadBuffer);
            sortCompute.Dispatch(_initSortKernel, threadGroups, 1, 1);

            // Radix sort: 13 dispatches total
            _radixSort.Dispatch(totalCount, _sortKeysBuffer, _sortPayloadBuffer, _radixResources);
        }

        void DispatchBitonicSort(int totalCount, bool stereo = false)
        {
            int padded = Mathf.NextPowerOfTwo(totalCount);
            int threadGroups = (padded + 255) / 256;

            sortCompute.SetInt(ID_Count, totalCount);
            sortCompute.SetInt(ID_PaddedCount, padded);
            sortCompute.SetInt(ID_IsStereo, stereo ? 1 : 0);
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
