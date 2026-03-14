using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders Gaussian splats via a compute prepass + lightweight render shader.
    /// Zero-copy from training buffers. Compute prepass handles EWA projection,
    /// covariance decomposition, and SH color evaluation. Render shader is trivial.
    /// All sectors are concatenated into a single draw call for performance.
    /// </summary>
    public class GSSectorRenderer : MonoBehaviour
    {
        [SerializeField] Material splatMaterial;
        [SerializeField] ComputeShader viewPrepassCompute;
        [SerializeField, Range(0.1f, 4f)] float splatSizeMultiplier = 1f;
        [SerializeField] int shDegree = 2;

        SectorScheduler _scheduler;
        MaterialPropertyBlock _props;
        readonly List<(int id, GSplatBuffers buffers)> _readySectors = new();
        bool _ready;
        bool _loggedFirstDraw;

        GraphicsBuffer _viewDataBuffer;
        int _viewDataCapacity;
        int _csKernel = -1;

        static readonly int ID_NumPoints    = Shader.PropertyToID("_NumPoints");
        static readonly int ID_WriteOffset  = Shader.PropertyToID("_WriteOffset");
        static readonly int ID_SplatSize    = Shader.PropertyToID("_SplatSize");
        static readonly int ID_SHDegree     = Shader.PropertyToID("_SHDegree");
        static readonly int ID_ViewMatrix   = Shader.PropertyToID("_ViewMatrix");
        static readonly int ID_Focal        = Shader.PropertyToID("_Focal");
        static readonly int ID_ScreenSize   = Shader.PropertyToID("_ScreenSize");
        static readonly int ID_CamPos       = Shader.PropertyToID("_CamPos");
        static readonly int ID_Means        = Shader.PropertyToID("_Means");
        static readonly int ID_Scales       = Shader.PropertyToID("_Scales");
        static readonly int ID_Quats        = Shader.PropertyToID("_Quats");
        static readonly int ID_FeaturesDC   = Shader.PropertyToID("_FeaturesDC");
        static readonly int ID_FeaturesRest = Shader.PropertyToID("_FeaturesRest");
        static readonly int ID_Opacities    = Shader.PropertyToID("_Opacities");
        static readonly int ID_ViewData     = Shader.PropertyToID("_ViewData");
        static readonly int ID_SplatViewData = Shader.PropertyToID("_SplatViewData");
        static readonly int ID_SplatCount   = Shader.PropertyToID("_SplatCount");

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
                _csKernel = viewPrepassCompute.FindKernel("CSCalcViewData");
            else
                Debug.LogWarning("[GSSectorRenderer] No viewPrepassCompute shader assigned!");

            _ready = true;
            Debug.Log($"[GSSectorRenderer] Initialized, material={splatMaterial?.name ?? "NULL"}, " +
                      $"compute={viewPrepassCompute?.name ?? "NULL"}");
        }

        void EnsureViewDataBuffer(int requiredCapacity)
        {
            if (_viewDataBuffer != null && _viewDataCapacity >= requiredCapacity) return;
            _viewDataBuffer?.Release();
            _viewDataCapacity = Mathf.Max(requiredCapacity, 1024);
            _viewDataBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, _viewDataCapacity, 40);
        }

        void LateUpdate()
        {
            if (!_ready || _scheduler == null || splatMaterial == null || _csKernel < 0) return;

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

            EnsureViewDataBuffer(totalCount);

            var camT = cam.transform;
            Matrix4x4 viewPosZ = Matrix4x4.TRS(camT.position, camT.rotation, Vector3.one).inverse;

            float screenW = cam.pixelWidth;
            float screenH = cam.pixelHeight;
            float fx = Mathf.Abs(cam.projectionMatrix[0, 0]) * screenW / 2f;
            float fy = Mathf.Abs(cam.projectionMatrix[1, 1]) * screenH / 2f;

            viewPrepassCompute.SetFloat(ID_SplatSize, splatSizeMultiplier);
            viewPrepassCompute.SetInt(ID_SHDegree, shDegree);
            viewPrepassCompute.SetMatrix(ID_ViewMatrix, viewPosZ);
            viewPrepassCompute.SetVector(ID_Focal, new Vector4(fx, fy, 0, 0));
            viewPrepassCompute.SetVector(ID_ScreenSize, new Vector4(screenW, screenH, 0, 0));
            viewPrepassCompute.SetVector(ID_CamPos, (Vector4)camT.position);
            viewPrepassCompute.SetBuffer(_csKernel, ID_ViewData, _viewDataBuffer);

            int writeOffset = 0;
            Bounds combinedBounds = default;
            bool first = true;

            foreach (var (id, buffers) in _readySectors)
            {
                int n = buffers.CurrentCount;
                if (n <= 0) continue;

                viewPrepassCompute.SetInt(ID_NumPoints, n);
                viewPrepassCompute.SetInt(ID_WriteOffset, writeOffset);
                viewPrepassCompute.SetBuffer(_csKernel, ID_Means, buffers.Means);
                viewPrepassCompute.SetBuffer(_csKernel, ID_Scales, buffers.Scales);
                viewPrepassCompute.SetBuffer(_csKernel, ID_Quats, buffers.Quats);
                viewPrepassCompute.SetBuffer(_csKernel, ID_FeaturesDC, buffers.FeaturesDC);
                viewPrepassCompute.SetBuffer(_csKernel, ID_FeaturesRest, buffers.FeaturesRest);
                viewPrepassCompute.SetBuffer(_csKernel, ID_Opacities, buffers.Opacities);

                int threadGroups = (n + 255) / 256;
                viewPrepassCompute.Dispatch(_csKernel, threadGroups, 1, 1);

                var bounds = _scheduler.GetSectorBounds(id);
                if (first) { combinedBounds = bounds; first = false; }
                else combinedBounds.Encapsulate(bounds);

                writeOffset += n;
            }

            _props.SetBuffer(ID_SplatViewData, _viewDataBuffer);
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

        void OnDisable()
        {
            _ready = false;
            _viewDataBuffer?.Release();
            _viewDataBuffer = null;
            _viewDataCapacity = 0;
        }
    }
}
