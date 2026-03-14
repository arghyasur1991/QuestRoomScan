using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Top-level manager for progressive Gaussian splat training.
    /// Integrates with the scan pipeline: seeds from mesh, trains per-sector,
    /// masks mesh where splats are ready, renders splats.
    ///
    /// RoomScanner calls OnCameraFrame / OnMeshExtracted directly (same assembly).
    /// TrainFrame runs in this component's own Update().
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

        [Header("Startup")]
        [SerializeField, Tooltip("Min integration count before GSplat training begins")]
        int minIntegrationsBeforeTraining = 60;

        SectorScheduler _scheduler;
        GSplatTrainer _trainer;
        bool _initialized;
        Camera _mainCam;
        Plane[] _frustumPlanes = new Plane[6];

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

        void Start()
        {
            _mainCam = Camera.main;
        }

        void TryInitialize()
        {
            if (_initialized) return;
            var vi = VolumeIntegrator.Instance;
            if (vi == null || vi.IntegrationCount < minIntegrationsBeforeTraining) return;

            _scheduler = new SectorScheduler(vi.VoxelCount, vi.VoxelSize,
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
            Debug.Log($"[GSplatManager] Initialized — voxels={vi.VoxelCount}, voxelSize={vi.VoxelSize}");
        }

        /// <summary>
        /// Called by RoomScanner each integration tick with the current camera frame.
        /// Converts to a keyframe for the training ring buffer.
        /// </summary>
        public void OnCameraFrame(Texture frame, Vector3 pos, Quaternion rot,
                                   Vector2 focalLen, Vector2 principalPt, Vector2 currentRes)
        {
            if (!_initialized || frame == null) return;

            Texture2D snap = ToTexture2D(frame);
            if (snap == null) return;

            Matrix4x4 view = Matrix4x4.TRS(pos, rot, Vector3.one).inverse;

            float fx = focalLen.x, fy = focalLen.y;
            float cx = principalPt.x, cy = principalPt.y;
            float w = currentRes.x, h = currentRes.y;

            float n = 0.1f, f = 100f;
            var proj = new Matrix4x4();
            proj[0, 0] = 2f * fx / w;
            proj[1, 1] = 2f * fy / h;
            proj[0, 2] = 1f - 2f * cx / w;
            proj[1, 2] = 2f * cy / h - 1f;
            proj[2, 2] = -(f + n) / (f - n);
            proj[2, 3] = -2f * f * n / (f - n);
            proj[3, 2] = -1f;

            AddKeyframe(new KeyframeData
            {
                Texture = snap,
                ViewMatrix = view,
                ProjMatrix = proj,
                Fx = fx, Fy = fy,
                Cx = cx, Cy = cy,
                CamPos = pos
            });
        }

        /// <summary>
        /// Called by RoomScanner after each mesh extraction cycle.
        /// </summary>
        public void OnMeshExtracted()
        {
            if (!_initialized) return;
            var me = MeshExtractor.Instance;
            if (me?.GpuSurfaceNets == null) return;
            var vi = VolumeIntegrator.Instance;
            if (vi == null) return;

            for (int i = 0; i < SectorScheduler.TotalSectors; i++)
                _scheduler.MarkMeshReady(i);
        }

        void Update()
        {
            TryInitialize();
            if (!_initialized || _mainCam == null) return;

            var camTr = _mainCam.transform;
            GeometryUtility.CalculateFrustumPlanes(_mainCam, _frustumPlanes);
            TrainFrame(camTr.position, camTr.forward, _frustumPlanes);
        }

        static Texture2D ToTexture2D(Texture src)
        {
            if (src is Texture2D t2d) return t2d;
            if (src is RenderTexture rt)
            {
                var prev = RenderTexture.active;
                RenderTexture.active = rt;
                var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
                tex.Apply(false, true);
                RenderTexture.active = prev;
                return tex;
            }
            return null;
        }

        void AddKeyframe(KeyframeData kf)
        {
            if (_keyframes.Count >= MaxKeyframes)
            {
                if (_keyframes[0].Texture != null)
                    Destroy(_keyframes[0].Texture);
                _keyframes.RemoveAt(0);
            }
            _keyframes.Add(kf);
        }

        void TrainFrame(Vector3 headPos, Vector3 gazeDir, Plane[] frustumPlanes)
        {
            if (_keyframes.Count < 2) return;

            const int maxSeedAttemptsPerFrame = 4;
            int seedAttempts = 0;

            while (seedAttempts < maxSeedAttemptsPerFrame)
            {
                int sectorId = _scheduler.PickNextSector(headPos, gazeDir, frustumPlanes, Time.time);
                if (sectorId < 0) return;

                ref var sector = ref _scheduler.Sectors[sectorId];
                var buffers = _scheduler.GetOrCreateBuffers(sectorId);

                if (buffers.CurrentCount == 0 && sector.State == SectorState.MeshOnly)
                {
                    seedAttempts++;
                    SeedFromMesh(sectorId, buffers);
                    if (buffers.CurrentCount == 0)
                    {
                        _scheduler.MarkEmpty(sectorId);
                        continue;
                    }
                }

                var kf = PickBestKeyframe(sectorId);
                if (!kf.HasValue) return;

                for (int i = 0; i < itersPerFrame; i++)
                {
                    _trainer.TrainStep(buffers, kf.Value.Texture,
                                       kf.Value.ViewMatrix, kf.Value.ProjMatrix,
                                       kf.Value.Fx, kf.Value.Fy, kf.Value.Cx, kf.Value.Cy,
                                       kf.Value.CamPos);
                }

                _scheduler.AdvanceTraining(sectorId, itersPerFrame, 0f, buffers.CurrentCount);
                UpdateMeshMask();
                return;
            }
        }

        static readonly int ID_NumVertices = Shader.PropertyToID("_NumVertices");
        static readonly int ID_MaxGaussians = Shader.PropertyToID("_MaxGaussians");
        static readonly int ID_InitScale = Shader.PropertyToID("_InitScale");
        static readonly int ID_SectorMin = Shader.PropertyToID("_SectorMin");
        static readonly int ID_SectorMax = Shader.PropertyToID("_SectorMax");
        static readonly int ID_MeshVertices = Shader.PropertyToID("_MeshVertices");
        static readonly int ID_OutMeans = Shader.PropertyToID("_OutMeans");
        static readonly int ID_OutScales = Shader.PropertyToID("_OutScales");
        static readonly int ID_OutQuats = Shader.PropertyToID("_OutQuats");
        static readonly int ID_OutOpacities = Shader.PropertyToID("_OutOpacities");
        static readonly int ID_OutFeaturesDC = Shader.PropertyToID("_OutFeaturesDC");
        static readonly int ID_OutFeaturesRest = Shader.PropertyToID("_OutFeaturesRest");
        static readonly int ID_OutCount = Shader.PropertyToID("_OutCount");

        readonly int[] _countReadback = new int[1];
        readonly int[] _meshCountReadback = new int[2];

        void SeedFromMesh(int sectorId, GSplatBuffers buffers)
        {
            if (initGaussiansCompute == null) return;

            var me = MeshExtractor.Instance;
            if (me?.GpuSurfaceNets == null) return;

            var vertBuf = me.GpuSurfaceNets.VertexBuffer;
            var countersBuf = me.GpuSurfaceNets.CountersBuffer;
            if (vertBuf == null || countersBuf == null) return;

            countersBuf.GetData(_meshCountReadback);
            int numVertices = _meshCountReadback[0];
            if (numVertices <= 0) return;

            var bounds = _scheduler.GetSectorBounds(sectorId);
            Vector3 sMin = bounds.center - bounds.extents;
            Vector3 sMax = bounds.center + bounds.extents;

            float vi_voxelSize = VolumeIntegrator.Instance?.VoxelSize ?? 0.05f;
            float initScale = Mathf.Log(vi_voxelSize);

            buffers.CountBuffer.SetData(new int[] { 0 });

            int kernel = initGaussiansCompute.FindKernel("InitFromMeshVertices");
            initGaussiansCompute.SetInt(ID_NumVertices, numVertices);
            initGaussiansCompute.SetInt(ID_MaxGaussians, buffers.MaxGaussians);
            initGaussiansCompute.SetFloat(ID_InitScale, initScale);
            initGaussiansCompute.SetVector(ID_SectorMin, new Vector4(sMin.x, sMin.y, sMin.z, 0));
            initGaussiansCompute.SetVector(ID_SectorMax, new Vector4(sMax.x, sMax.y, sMax.z, 0));

            initGaussiansCompute.SetBuffer(kernel, ID_MeshVertices, vertBuf);
            initGaussiansCompute.SetBuffer(kernel, ID_OutMeans, buffers.Means);
            initGaussiansCompute.SetBuffer(kernel, ID_OutScales, buffers.Scales);
            initGaussiansCompute.SetBuffer(kernel, ID_OutQuats, buffers.Quats);
            initGaussiansCompute.SetBuffer(kernel, ID_OutOpacities, buffers.Opacities);
            initGaussiansCompute.SetBuffer(kernel, ID_OutFeaturesDC, buffers.FeaturesDC);
            initGaussiansCompute.SetBuffer(kernel, ID_OutFeaturesRest, buffers.FeaturesRest);
            initGaussiansCompute.SetBuffer(kernel, ID_OutCount, buffers.CountBuffer);

            int threadGroups = (numVertices + 255) / 256;
            initGaussiansCompute.Dispatch(kernel, threadGroups, 1, 1);

            buffers.CountBuffer.GetData(_countReadback);
            buffers.CurrentCount = Mathf.Min(_countReadback[0], buffers.MaxGaussians);

            Debug.Log($"[GSplatManager] Seeded sector {sectorId}: {buffers.CurrentCount} Gaussians " +
                      $"from {numVertices} mesh verts (AABB {sMin} → {sMax})");
        }

        KeyframeData? PickBestKeyframe(int sectorId)
        {
            if (_keyframes.Count == 0) return null;

            var bounds = _scheduler.GetSectorBounds(sectorId);
            float bestScore = -1;
            int bestIdx = -1;

            for (int i = 0; i < _keyframes.Count; i++)
            {
                float dist = Vector3.Distance(bounds.center, _keyframes[i].CamPos);
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
            foreach (var kf in _keyframes)
            {
                if (kf.Texture != null) Destroy(kf.Texture);
            }
            _keyframes.Clear();

            _trainer?.Dispose();
            _scheduler?.Dispose();
        }
    }
}
