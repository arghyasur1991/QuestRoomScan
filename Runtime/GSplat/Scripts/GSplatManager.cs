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
    ///
    /// Self-wires at runtime by subscribing to RoomScanner events — no
    /// cross-assembly reference needed from RoomScanner to this assembly.
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
        RoomScanner _scanner;
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
            _scanner = RoomScanner.Instance;
            if (_scanner == null)
            {
                _scanner = FindFirstObjectByType<RoomScanner>();
                if (_scanner == null)
                {
                    Debug.LogWarning("[GSplatManager] RoomScanner not found — disabling");
                    enabled = false;
                    return;
                }
            }

            _scanner.ColorFrameCaptured += OnColorFrame;
            _scanner.MeshExtracted += OnMeshExtracted;
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

        void OnColorFrame(CameraFrameArgs args)
        {
            if (!_initialized) return;
            if (args.Frame == null) return;

            Texture2D snap = ToTexture2D(args.Frame);
            if (snap == null) return;

            Matrix4x4 view = Matrix4x4.TRS(args.Position, args.Rotation, Vector3.one).inverse;

            float fx = args.FocalLength.x;
            float fy = args.FocalLength.y;
            float cx = args.PrincipalPoint.x;
            float cy = args.PrincipalPoint.y;
            float w = args.CurrentResolution.x;
            float h = args.CurrentResolution.y;

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
                CamPos = args.Position
            });
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

        void OnMeshExtracted()
        {
            if (!_initialized) return;
            var me = MeshExtractor.Instance;
            if (me?.GpuSurfaceNets == null) return;
            var vi = VolumeIntegrator.Instance;
            if (vi == null) return;
            NotifyMeshUpdated(me.GpuSurfaceNets, vi.VoxelSize);
        }

        void Update()
        {
            TryInitialize();
            if (!_initialized || _mainCam == null) return;

            var camTr = _mainCam.transform;
            GeometryUtility.CalculateFrustumPlanes(_mainCam, _frustumPlanes);
            TrainFrame(camTr.position, camTr.forward, _frustumPlanes);
        }

        public void AddKeyframe(KeyframeData kf)
        {
            if (_keyframes.Count >= MaxKeyframes)
            {
                if (_keyframes[0].Texture != null)
                    Destroy(_keyframes[0].Texture);
                _keyframes.RemoveAt(0);
            }
            _keyframes.Add(kf);
        }

        public void NotifyMeshUpdated(GPUSurfaceNets surfaceNets, float voxelSize)
        {
            if (!_initialized || surfaceNets == null) return;
            for (int i = 0; i < SectorScheduler.TotalSectors; i++)
                _scheduler.MarkMeshReady(i);
        }

        void TrainFrame(Vector3 headPos, Vector3 gazeDir, Plane[] frustumPlanes)
        {
            if (!_initialized || _keyframes.Count < 2) return;

            int sectorId = _scheduler.PickNextSector(headPos, gazeDir, frustumPlanes, Time.time);
            if (sectorId < 0) return;

            ref var sector = ref _scheduler.Sectors[sectorId];
            var buffers = _scheduler.GetOrCreateBuffers(sectorId);

            if (buffers.CurrentCount == 0 && sector.State == SectorState.MeshOnly)
            {
                SeedFromMesh(sectorId, buffers);
                if (buffers.CurrentCount == 0) return;
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
        }

        void SeedFromMesh(int sectorId, GSplatBuffers buffers)
        {
            // TODO: dispatch InitGaussians compute shader to extract mesh vertices
            // within the sector AABB and create initial Gaussians.
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
                float dist = Vector3.Distance(bounds.center, kf.CamPos);
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
            if (_scanner != null)
            {
                _scanner.ColorFrameCaptured -= OnColorFrame;
                _scanner.MeshExtracted -= OnMeshExtracted;
            }

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
