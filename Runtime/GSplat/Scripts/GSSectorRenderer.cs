using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Renders trained Gaussian splat sectors directly from GPU training buffers.
    /// Zero-copy: no file I/O or CPU readback. Uses RenderPrimitivesIndirect.
    /// </summary>
    public class GSSectorRenderer : MonoBehaviour
    {
        [SerializeField] Material splatMaterial;

        SectorScheduler _scheduler;
        MaterialPropertyBlock _props;
        readonly List<(int id, GSplatBuffers buffers)> _readySectors = new();
        bool _ready;

        static readonly int ID_SplatData = Shader.PropertyToID("_SplatData");
        static readonly int ID_SplatIndices = Shader.PropertyToID("_SplatIndices");
        static readonly int ID_SplatCount = Shader.PropertyToID("_SplatCount");

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
        }

        void LateUpdate()
        {
            if (!_ready || _scheduler == null || splatMaterial == null) return;

            _scheduler.GetSplatReadySectors(_readySectors);
            if (_readySectors.Count == 0) return;

            foreach (var (id, buffers) in _readySectors)
            {
                if (buffers.CurrentCount <= 0) continue;

                // For now, render each sector's Gaussians as a billboard quad pass.
                // The training buffers ARE the render buffers — zero copy.
                // A proper implementation would sort by depth per frame,
                // but for the initial version we skip sorting (sectors are small).
                _props.SetBuffer(ID_SplatData, buffers.Means);
                _props.SetInt(ID_SplatCount, buffers.CurrentCount);

                var bounds = _scheduler.GetSectorBounds(id);
                var rp = new RenderParams(splatMaterial)
                {
                    worldBounds = bounds,
                    matProps = _props,
                    receiveShadows = false,
                    shadowCastingMode = ShadowCastingMode.Off,
                    layer = gameObject.layer
                };

                // 4 vertices per splat (quad), 6 indices per quad
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
