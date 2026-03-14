using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders trained Gaussian splat sectors directly from GPU training buffers.
    /// Zero-copy: no file I/O or CPU readback. Projects 3D covariance to 2D conic
    /// per-splat in the vertex shader for proper elliptical Gaussian rendering.
    /// </summary>
    public class GSSectorRenderer : MonoBehaviour
    {
        [SerializeField] Material splatMaterial;
        [SerializeField, Range(0.1f, 4f)] float splatSizeMultiplier = 1f;

        SectorScheduler _scheduler;
        MaterialPropertyBlock _props;
        readonly List<(int id, GSplatBuffers buffers)> _readySectors = new();
        bool _ready;
        bool _loggedFirstDraw;

        static readonly int ID_Means = Shader.PropertyToID("_Means");
        static readonly int ID_FeaturesDC = Shader.PropertyToID("_FeaturesDC");
        static readonly int ID_Opacities = Shader.PropertyToID("_Opacities");
        static readonly int ID_Scales = Shader.PropertyToID("_Scales");
        static readonly int ID_Quats = Shader.PropertyToID("_Quats");
        static readonly int ID_SplatCount = Shader.PropertyToID("_SplatCount");
        static readonly int ID_SplatSize = Shader.PropertyToID("_SplatSize");

        public Material SplatMaterial
        {
            get => splatMaterial;
            set => splatMaterial = value;
        }

        public void Initialize(SectorScheduler scheduler)
        {
            _scheduler = scheduler;
            _props = new MaterialPropertyBlock();
            _ready = true;
            Debug.Log($"[GSSectorRenderer] Initialized, material={splatMaterial?.name ?? "NULL"}");
        }

        void LateUpdate()
        {
            if (!_ready || _scheduler == null || splatMaterial == null) return;

            _scheduler.GetSplatReadySectors(_readySectors);
            if (_readySectors.Count == 0) return;

            if (!_loggedFirstDraw)
            {
                _loggedFirstDraw = true;
                int totalGaussians = 0;
                foreach (var (sid, buf) in _readySectors) totalGaussians += buf.CurrentCount;
                Debug.Log($"[GSSectorRenderer] First draw: {_readySectors.Count} sector(s), {totalGaussians} total gaussians, " +
                          $"splatSize={splatSizeMultiplier}");
            }

            foreach (var (id, buffers) in _readySectors)
            {
                if (buffers.CurrentCount <= 0) continue;

                _props.SetBuffer(ID_Means, buffers.Means);
                _props.SetBuffer(ID_FeaturesDC, buffers.FeaturesDC);
                _props.SetBuffer(ID_Opacities, buffers.Opacities);
                _props.SetBuffer(ID_Scales, buffers.Scales);
                _props.SetBuffer(ID_Quats, buffers.Quats);
                _props.SetInt(ID_SplatCount, buffers.CurrentCount);
                _props.SetFloat(ID_SplatSize, splatSizeMultiplier);

                var bounds = _scheduler.GetSectorBounds(id);
                var rp = new RenderParams(splatMaterial)
                {
                    worldBounds = bounds,
                    matProps = _props,
                    receiveShadows = false,
                    shadowCastingMode = ShadowCastingMode.Off,
                    layer = gameObject.layer
                };

                int vertexCount = buffers.CurrentCount * 4;
                Graphics.RenderPrimitives(rp, MeshTopology.Quads, vertexCount);
            }
        }

        void OnDisable()
        {
            _ready = false;
        }
    }
}
