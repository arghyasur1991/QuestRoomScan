using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Renders the GPU Surface Nets mesh via Graphics.RenderPrimitivesIndirect.
    /// Replaces per-chunk MeshFilter+MeshRenderer with a single indirect draw call.
    /// </summary>
    public class GPUMeshRenderer : MonoBehaviour
    {
        [SerializeField] private Material gpuMeshMaterial;

        private GPUSurfaceNets _surfaceNets;
        private MaterialPropertyBlock _props;
        private bool _ready;
        private Bounds _bounds;

        private static readonly int ID_SurfaceVerts = Shader.PropertyToID("_SurfaceVerts");
        private static readonly int ID_SurfaceIndices = Shader.PropertyToID("_SurfaceIndices");

        public Material GpuMeshMaterial
        {
            get => gpuMeshMaterial;
            set => gpuMeshMaterial = value;
        }

        public void Initialize(GPUSurfaceNets surfaceNets, Bounds volumeBounds)
        {
            _surfaceNets = surfaceNets;
            _bounds = volumeBounds;
            _props = new MaterialPropertyBlock();
            _ready = true;
        }

        public void UpdateBounds(Bounds bounds)
        {
            _bounds = bounds;
        }

        private void LateUpdate()
        {
            if (!_ready || _surfaceNets == null || gpuMeshMaterial == null)
                return;

            var vertBuf = _surfaceNets.VertexBuffer;
            var idxBuf = _surfaceNets.IndexBuffer;
            var argsBuf = _surfaceNets.DrawIndirectArgs;

            if (vertBuf == null || idxBuf == null || argsBuf == null)
                return;

            _props.SetBuffer(ID_SurfaceVerts, vertBuf);
            _props.SetBuffer(ID_SurfaceIndices, idxBuf);

            var rp = new RenderParams(gpuMeshMaterial)
            {
                worldBounds = _bounds,
                matProps = _props,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };

            Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles, argsBuf, 1);
        }

        private void OnDisable()
        {
            _ready = false;
        }
    }
}
