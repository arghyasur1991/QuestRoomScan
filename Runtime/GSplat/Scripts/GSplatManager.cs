using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Top-level manager for progressive Gaussian splat training.
    /// Integrates with the scan pipeline: seeds from mesh, trains per-sector,
    /// masks mesh where splats are ready, renders splats.
    /// </summary>
    public class GSplatManager : MonoBehaviour
    {
        [Header("Compute Shaders")]
        [SerializeField] ComputeShader projectSHCompute;
        [SerializeField] ComputeShader tileSortCompute;
        [SerializeField] ComputeShader rasterizeCompute;
        [SerializeField] ComputeShader lossCompute;
        [SerializeField] ComputeShader rasterBwdCompute;
        [SerializeField] ComputeShader projBwdCompute;
        [SerializeField] ComputeShader adamCompute;
        [SerializeField] ComputeShader initGaussiansCompute;

        [Header("Rendering")]
        [SerializeField] Material meshMaterial;
        [SerializeField] GSSectorRenderer splatRenderer;

        [Header("Training")]
        [SerializeField] int maxGaussiansPerSector = 10000;
        [SerializeField] int targetItersPerSector = 300;
        [SerializeField] int itersPerFrame = 3;
        [SerializeField] int trainingResWidth = 640;
        [SerializeField] int trainingResHeight = 480;
        [SerializeField, Range(0f, 1f)] float ssimWeight = 0.2f;

        SectorScheduler _scheduler;
        GSplatTrainer _trainer;
        bool _initialized;

        // Keyframe ring buffer (textures + poses from scan pipeline)
        readonly List<KeyframeData> _keyframes = new();
        const int MaxKeyframes = 16;

        static readonly int ID_SectorSplatMask0 = Shader.PropertyToID("_SectorSplatMask0");
        static readonly int ID_SectorSplatMask1 = Shader.PropertyToID("_SectorSplatMask1");

        public struct KeyframeData
        {
            public Texture2D Texture;
            public Matrix4x4 ViewMatrix;
            public Matrix4x4 ProjMatrix;
            public float Fx, Fy, Cx, Cy;
            public Vector3 CamPos;
        }

        public SectorScheduler Scheduler => _scheduler;
        public bool IsTraining => _initialized && _scheduler?.CurrentSectorId >= 0;

        public void Initialize(int3 voxelCount, float voxelSize)
        {
            _scheduler = new SectorScheduler(voxelCount, voxelSize,
                                              maxGaussiansPerSector, targetItersPerSector);

            var config = GSplatTrainer.Config.Default;
            config.SSIMWeight = ssimWeight;

            _trainer = new GSplatTrainer(
                projectSHCompute, tileSortCompute, rasterizeCompute,
                lossCompute, rasterBwdCompute, projBwdCompute, adamCompute,
                maxGaussiansPerSector, trainingResWidth, trainingResHeight,
                config);

            if (splatRenderer != null)
                splatRenderer.Initialize(_scheduler);

            _initialized = true;
            Debug.Log("[GSplatManager] Initialized with sector grid 4x4x4");
        }

        /// <summary>
        /// Called by scan pipeline when a keyframe is captured.
        /// </summary>
        public void AddKeyframe(KeyframeData kf)
        {
            if (_keyframes.Count >= MaxKeyframes)
                _keyframes.RemoveAt(0);
            _keyframes.Add(kf);
        }

        /// <summary>
        /// Called by scan pipeline after mesh extraction to mark which sectors have geometry.
        /// </summary>
        public void NotifyMeshUpdated(GPUSurfaceNets surfaceNets, float voxelSize)
        {
            if (!_initialized || surfaceNets == null) return;

            // For now, mark all sectors as MeshOnly if volume has any data.
            // A proper implementation would check which sectors have vertices.
            for (int i = 0; i < SectorScheduler.TotalSectors; i++)
                _scheduler.MarkMeshReady(i);
        }

        /// <summary>
        /// Run N training iterations this frame. Called from the scan pipeline Update loop.
        /// </summary>
        public void TrainFrame(Vector3 headPos, Vector3 gazeDir, Plane[] frustumPlanes)
        {
            if (!_initialized || _keyframes.Count < 2) return;

            int sectorId = _scheduler.PickNextSector(headPos, gazeDir, frustumPlanes, Time.time);
            if (sectorId < 0) return;

            ref var sector = ref _scheduler.Sectors[sectorId];
            var buffers = _scheduler.GetOrCreateBuffers(sectorId);

            // Seed Gaussians from mesh if this is the first time
            if (buffers.CurrentCount == 0 && sector.State == SectorState.MeshOnly)
            {
                SeedFromMesh(sectorId, buffers);
                if (buffers.CurrentCount == 0) return;
            }

            // Pick keyframes visible to this sector
            var kf = PickBestKeyframe(sectorId);
            if (!kf.HasValue) return;

            // Run training iterations
            for (int i = 0; i < itersPerFrame; i++)
            {
                _trainer.TrainStep(buffers, kf.Value.Texture,
                                   kf.Value.ViewMatrix, kf.Value.ProjMatrix,
                                   kf.Value.Fx, kf.Value.Fy, kf.Value.Cx, kf.Value.Cy,
                                   kf.Value.CamPos);
            }

            _scheduler.AdvanceTraining(sectorId, itersPerFrame, 0f, buffers.CurrentCount);

            // Update mesh sector mask
            UpdateMeshMask();
        }

        void SeedFromMesh(int sectorId, GSplatBuffers buffers)
        {
            var vi = VolumeIntegrator.Instance;
            if (vi == null) return;

            // TODO: Read vertex positions from GPUSurfaceNets and dispatch InitGaussians compute
            // For now, this is a placeholder that would use the InitGaussians.compute shader
            // to extract mesh vertices within the sector AABB and create initial Gaussians.
            Debug.Log($"[GSplatManager] Seeding sector {sectorId} from mesh vertices");
        }

        KeyframeData? PickBestKeyframe(int sectorId)
        {
            if (_keyframes.Count == 0) return null;

            var bounds = _scheduler.GetSectorBounds(sectorId);
            float bestScore = -1;
            int bestIdx = -1;

            for (int i = 0; i < _keyframes.Count; i++)
            {
                var kf = _keyframes[i];
                Vector3 toSector = bounds.center - kf.CamPos;
                float dist = toSector.magnitude;
                float score = 1f / (1f + dist);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = i;
                }
            }

            return bestIdx >= 0 ? _keyframes[bestIdx] : null;
        }

        void UpdateMeshMask()
        {
            ulong mask = _scheduler.SplatMask;
            uint mask0 = (uint)(mask & 0xFFFFFFFF);
            uint mask1 = (uint)((mask >> 32) & 0xFFFFFFFF);

            if (meshMaterial != null)
            {
                if (mask != 0)
                {
                    meshMaterial.EnableKeyword("_GSPLAT_SECTOR_MASK");
                    meshMaterial.SetInt(ID_SectorSplatMask0, (int)mask0);
                    meshMaterial.SetInt(ID_SectorSplatMask1, (int)mask1);
                }
                else
                {
                    meshMaterial.DisableKeyword("_GSPLAT_SECTOR_MASK");
                }
            }
        }

        void OnDestroy()
        {
            _trainer?.Dispose();
            _scheduler?.Dispose();
        }
    }
}
