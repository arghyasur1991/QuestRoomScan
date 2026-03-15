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
        [SerializeField] ComputeShader densifyCompute;
        [SerializeField] ComputeShader initGaussiansCompute;

        [Header("Rendering")]
        [SerializeField] Material meshMaterial;
        [SerializeField] GSSectorRenderer splatRenderer;

        [Header("Training")]
        [SerializeField] int maxGaussiansPerSector = 10000;
        [SerializeField] int targetItersPerSector = 1500;
        [SerializeField] int itersPerFrame = 1;
        [SerializeField] int trainingResWidth = 320;
        [SerializeField] int trainingResHeight = 240;
        [SerializeField, Range(0f, 1f)] float ssimWeight = 0.2f;

        [Header("Startup")]
        [SerializeField, Tooltip("Min integration count before GSplat training begins")]
        int minIntegrationsBeforeTraining = 60;

        [Header("Debug")]
        [SerializeField, Tooltip("Skip training; render seeded Gaussians directly")]
        bool debugSkipTraining;
        [SerializeField, Tooltip("Disable mesh masking (show both mesh and splats)")]
        bool debugDisableMeshMask = true;

        SectorScheduler _scheduler;
        GSplatTrainer _trainer;
        TrainingPacer _pacer;
        bool _initialized;
        Camera _mainCam;
        Plane[] _frustumPlanes = new Plane[6];
        int _lastTrainedSectorId = -1;
        readonly List<KeyframeData> _keyframes = new();
        const int MaxKeyframes = 32;

        // Motion gating for keyframe diversity
        const float MinKeyframeDist = 0.10f;   // meters
        const float MinKeyframeAngle = 10f;    // degrees
        const int MinKeyframesForTraining = 5;
        Vector3 _lastKeyframeCamPos = Vector3.positiveInfinity;
        Quaternion _lastKeyframeCamRot = Quaternion.identity;
        bool _hasLastKeyframePose;

        static readonly int ID_SectorSplatMask0 = Shader.PropertyToID("_SectorSplatMask0");
        static readonly int ID_SectorSplatMask1 = Shader.PropertyToID("_SectorSplatMask1");

        public struct KeyframeData
        {
            public RenderTexture Texture;
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
                densifyCompute,
                maxGaussiansPerSector, trainingResWidth, trainingResHeight,
                config);

            var pacerCfg = TrainingPacer.Config.Default;
            pacerCfg.MaxIterations = targetItersPerSector;
            pacerCfg.DensifyStopIter = Mathf.Min(pacerCfg.DensifyStopIter, targetItersPerSector - 200);
            _pacer = new TrainingPacer(pacerCfg);

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
                                   Vector2 focalLen, Vector2 principalPt,
                                   Vector2 sensorRes, Vector2 currentRes)
        {
            if (!_initialized || frame == null) return;

            // Motion gating: only capture keyframes when the camera has moved
            // enough for diverse multi-view supervision
            if (_hasLastKeyframePose)
            {
                float dist = Vector3.Distance(pos, _lastKeyframeCamPos);
                float angle = Quaternion.Angle(rot, _lastKeyframeCamRot);
                if (dist < MinKeyframeDist && angle < MinKeyframeAngle)
                    return;
            }

            // Blit the camera frame directly to a training-resolution RenderTexture.
            // No Y-flip: the camera texture orientation already matches the compute
            // shader's RWTexture2D convention on Quest/Vulkan.
            var gtRT = new RenderTexture(trainingResWidth, trainingResHeight, 0,
                RenderTextureFormat.ARGB32) { filterMode = FilterMode.Bilinear };
            gtRT.Create();
            Graphics.Blit(frame, gtRT);

            Matrix4x4 view = Matrix4x4.TRS(pos, rot, Vector3.one).inverse;

            // Intrinsics from PassthroughCameraProvider are at sensor resolution.
            // The capture texture may be a centered crop of the sensor. Adjust the
            // principal point for the crop first, then scale to training resolution.
            // This matches the Python pipeline's unity_to_colmap_pose conversion.
            float cropX = (sensorRes.x - currentRes.x) * 0.5f;
            float cropY = (sensorRes.y - currentRes.y) * 0.5f;
            float cxCapture = principalPt.x - cropX;
            float cyCapture = principalPt.y - cropY;

            float scaleX = (float)trainingResWidth / currentRes.x;
            float scaleY = (float)trainingResHeight / currentRes.y;
            float fx = focalLen.x * scaleX;
            float fy = focalLen.y * scaleY;
            float cx = cxCapture * scaleX;
            float cy = cyCapture * scaleY;

            // Projection matrix: positive-Z-forward (matching our view matrix),
            // NO principal point offset (Ndc2Pix handles pp via cx/cy in _Intrinsics).
            // This matches the 3DGS convention where projmatrix = proj * view.
            float tw = (float)trainingResWidth, th = (float)trainingResHeight;
            float n = 0.1f, f = 100f;
            var proj = Matrix4x4.zero;
            proj[0, 0] = 2f * fx / tw;
            proj[1, 1] = 2f * fy / th;
            proj[2, 2] = (f + n) / (f - n);
            proj[2, 3] = -2f * f * n / (f - n);
            proj[3, 2] = 1f;

            // Combined world-to-clip matrix (3DGS convention: projmatrix includes view)
            Matrix4x4 fullProj = proj * view;

            AddKeyframe(new KeyframeData
            {
                Texture = gtRT,
                ViewMatrix = view,
                ProjMatrix = fullProj,
                Fx = fx, Fy = fy,
                Cx = cx, Cy = cy,
                CamPos = pos
            });

            if (_keyframes.Count <= 1)
            {
                Debug.Log($"[GSplatManager] Keyframe #{_keyframes.Count}: " +
                          $"sensorRes={sensorRes}, camRes={currentRes}, " +
                          $"trainRes={trainingResWidth}x{trainingResHeight}, " +
                          $"texSize={gtRT.width}x{gtRT.height}, " +
                          $"rawFocal=({focalLen.x:F1},{focalLen.y:F1}), " +
                          $"crop=({cropX:F1},{cropY:F1}), " +
                          $"scaledFocal=({fx:F1},{fy:F1}), " +
                          $"scaledPP=({cx:F1},{cy:F1})");
            }

            _lastKeyframeCamPos = pos;
            _lastKeyframeCamRot = rot;
            _hasLastKeyframePose = true;
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



        void AddKeyframe(KeyframeData kf)
        {
            if (_keyframes.Count >= MaxKeyframes)
            {
                var old = _keyframes[0].Texture;
                if (old != null) { old.Release(); Destroy(old); }
                _keyframes.RemoveAt(0);
            }
            _keyframes.Add(kf);
        }

        readonly HashSet<int> _diagnosticLogged = new();

        void TrainFrame(Vector3 headPos, Vector3 gazeDir, Plane[] frustumPlanes)
        {
            if (!debugSkipTraining && _keyframes.Count < MinKeyframesForTraining) return;

            // Phase 1: Opportunistically seed 1 MeshOnly sector per frame
            int seedId = _scheduler.PickNextMeshOnlySector(headPos, gazeDir, frustumPlanes);
            if (seedId >= 0)
            {
                var seedBuf = _scheduler.GetOrCreateBuffers(seedId);
                SeedFromMesh(seedId, seedBuf);
                if (seedBuf.CurrentCount == 0)
                {
                    _scheduler.MarkNoGeometry(seedId);
                }
                else if (debugSkipTraining)
                {
                    _scheduler.AdvanceTraining(seedId, targetItersPerSector, float.MaxValue, seedBuf.CurrentCount);
                    LogGaussianDiagnostics(seedId, seedBuf, "seed-only");
                }
                else
                {
                    _scheduler.AdvanceTraining(seedId, 0, float.MaxValue, seedBuf.CurrentCount);
                }
            }

            if (debugSkipTraining)
            {
                if (!debugDisableMeshMask)
                    UpdateMeshMask();
                return;
            }

            // Phase 2: Train the best Training-state sector
            int sectorId = _scheduler.PickNextTrainingSector(headPos, gazeDir, frustumPlanes);
            if (sectorId < 0) return;

            var buffers = _scheduler.GetOrCreateBuffers(sectorId);
            if (buffers.CurrentCount == 0) return;

            // Reset when switching to a new sector
            if (sectorId != _lastTrainedSectorId)
            {
                _trainer.ResetStep();
                _pacer.Reset();
                _lastTrainedSectorId = sectorId;
            }

            ref var sector = ref _scheduler.Sectors[sectorId];
            int recommendedIters = _pacer.UpdatePace(
                sector.TrainingIteration, 0f,
                Time.deltaTime * 1000f, 0f);

            int iters = Mathf.Max(1, Mathf.Min(itersPerFrame, recommendedIters));

            var kf = PickBestKeyframe(sectorId);
            if (!kf.HasValue) return;

            for (int i = 0; i < iters; i++)
            {
                _trainer.TrainStep(buffers, kf.Value.Texture,
                                   kf.Value.ViewMatrix, kf.Value.ProjMatrix,
                                   kf.Value.Fx, kf.Value.Fy, kf.Value.Cx, kf.Value.Cy,
                                   kf.Value.CamPos);
            }

            // Densify when the pacer says so
            if (_pacer.ShouldDensify)
            {
                int prevCount = buffers.CurrentCount;
                _trainer.Densify(buffers);
                Debug.Log($"[GSplatManager] Densified sector {sectorId}: {prevCount} → {buffers.CurrentCount}");
            }

            _scheduler.AdvanceTraining(sectorId, iters, 0f, buffers.CurrentCount);

            // Re-read sector (AdvanceTraining may have updated it)
            sector = ref _scheduler.Sectors[sectorId];
            bool shouldLog = sector.TrainingIteration % 30 == 0 ||
                             sector.State == SectorState.SplatReady;

            if (shouldLog)
            {
                Debug.Log($"[GSplatManager] Training sector {sectorId}: iter={sector.TrainingIteration}/{targetItersPerSector}, " +
                          $"gaussians={buffers.CurrentCount}, state={sector.State}");
            }

            if (sector.TrainingIteration <= 3 || sector.TrainingIteration == 30 ||
                sector.TrainingIteration % 300 == 0)
            {
                LogTrainingDiag(sectorId, sector.TrainingIteration, kf.Value, buffers);
            }

            if (sector.State == SectorState.SplatReady)
                LogGaussianDiagnostics(sectorId, buffers, "trained");

            if (!debugDisableMeshMask)
                UpdateMeshMask();
        }

        /// <summary>
        /// ReadPixels from a RenderTexture into a temporary Texture2D for CPU-side analysis.
        /// Both rendered and GT are RenderTextures now, so same Y convention — no flip needed.
        /// </summary>
        static Texture2D ReadbackRT(RenderTexture rt)
        {
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
            tex.Apply(false);
            RenderTexture.active = prev;
            return tex;
        }

        void LogTrainingDiag(int sectorId, int iter, KeyframeData kf, GSplatBuffers buffers)
        {
            var rendered = _trainer.RenderedImage;
            if (rendered == null) return;

            var readTex = ReadbackRT(rendered);
            var rendPixels = readTex.GetPixels32();
            int w = readTex.width, h = readTex.height;
            float rSum = 0, gSum = 0, bSum = 0;
            int nonBlack = 0;
            foreach (var px in rendPixels)
            {
                rSum += px.r; gSum += px.g; bSum += px.b;
                if (px.r > 5 || px.g > 5 || px.b > 5) nonBlack++;
            }
            int total = rendPixels.Length;
            float rAvg = rSum / total, gAvg = gSum / total, bAvg = bSum / total;

            var kfRead = ReadbackRT(kf.Texture);
            var kfPixels = kfRead.GetPixels32();
            int kTotal = kfPixels.Length;
            float krSum = 0, kgSum = 0, kbSum = 0;
            foreach (var px in kfPixels)
            {
                krSum += px.r; kgSum += px.g; kbSum += px.b;
            }
            float krAvg = krSum / kTotal, kgAvg = kgSum / kTotal, kbAvg = kbSum / kTotal;

            // Gradient magnitude: sample from beginning, middle, and end of the buffer
            int N = buffers.CurrentCount;
            int samplesPerRegion = 4;
            int[] offsets = { 0, Mathf.Max(0, N / 2 - samplesPerRegion / 2), Mathf.Max(0, N - samplesPerRegion) };
            float gradSumSq = 0; int gradCount = 0; int gradNonZero = 0;
            foreach (int off in offsets)
            {
                int cnt = Mathf.Min(samplesPerRegion, N - off);
                if (cnt <= 0) continue;
                var gSlice = new float[cnt * 3];
                buffers.GradMeans.GetData(gSlice, 0, off * 3, cnt * 3);
                for (int i = 0; i < cnt * 3; i++)
                {
                    gradSumSq += gSlice[i] * gSlice[i];
                    if (gSlice[i] != 0f) gradNonZero++;
                }
                gradCount += cnt * 3;
            }
            float gradMag = gradCount > 0 ? Mathf.Sqrt(gradSumSq / gradCount) : 0f;

            // Top-half vs bottom-half brightness comparison
            float rendTopR = 0, rendBotR = 0, kfTopR = 0, kfBotR = 0;
            int halfH = h / 2;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float r = rendPixels[y * w + x].r;
                    if (y < halfH) rendTopR += r; else rendBotR += r;
                }
            int kw = kf.Texture.width, kh = kf.Texture.height;
            int kHalfH = kh / 2;
            for (int y = 0; y < kh; y++)
                for (int x = 0; x < kw; x++)
                {
                    float r = kfPixels[y * kw + x].r;
                    if (y < kHalfH) kfTopR += r; else kfBotR += r;
                }
            int halfPixR = w * halfH, halfPixK = kw * kHalfH;
            float rendTopAvg = rendTopR / halfPixR, rendBotAvg = rendBotR / Mathf.Max(1, total - halfPixR);
            float kfTopAvg = kfTopR / halfPixK, kfBotAvg = kfBotR / Mathf.Max(1, kTotal - halfPixK);

            Debug.Log($"[GSplat-TrainDiag] sector={sectorId} iter={iter}: " +
                      $"rendered=({rAvg:F1},{gAvg:F1},{bAvg:F1}) nonBlack={nonBlack}/{total} " +
                      $"keyframe=({krAvg:F1},{kgAvg:F1},{kbAvg:F1}) size={kf.Texture.width}x{kf.Texture.height} " +
                      $"gradMeanRMS={gradMag:E2} gradNonZero={gradNonZero}/{gradCount} " +
                      $"rendTopR={rendTopAvg:F1} rendBotR={rendBotAvg:F1} " +
                      $"kfTopR={kfTopAvg:F1} kfBotR={kfBotAvg:F1}");

            // Save both images to disk on the first iteration for visual verification.
            // Both are RTs → same Y convention → ReadPixels gives matching orientation.
            if (iter <= 1)
            {
                string dir = System.IO.Path.Combine(Application.persistentDataPath, "GSExport", "train_diag");
                System.IO.Directory.CreateDirectory(dir);
                var rendJpg = readTex.EncodeToJPG(95);
                System.IO.File.WriteAllBytes(System.IO.Path.Combine(dir, $"rendered_s{sectorId}_i{iter}.jpg"), rendJpg);
                var kfJpg = kfRead.EncodeToJPG(95);
                System.IO.File.WriteAllBytes(System.IO.Path.Combine(dir, $"groundtruth_s{sectorId}_i{iter}.jpg"), kfJpg);
                Debug.Log($"[GSplat-TrainDiag] Saved diagnostic images to {dir}");
            }

            Destroy(readTex);
            Destroy(kfRead);
        }

        void LogGaussianDiagnostics(int sectorId, GSplatBuffers buffers, string tag)
        {
            if (_diagnosticLogged.Contains(sectorId)) return;
            _diagnosticLogged.Add(sectorId);

            int n = Mathf.Min(buffers.CurrentCount, 8);
            if (n == 0) return;

            var means = new float[n * 3];
            var dc = new float[n * 3];
            var opac = new float[n];
            var scales = new float[n * 3];

            buffers.Means.GetData(means, 0, 0, n * 3);
            buffers.FeaturesDC.GetData(dc, 0, 0, n * 3);
            buffers.Opacities.GetData(opac, 0, 0, n);
            buffers.Scales.GetData(scales, 0, 0, n * 3);

            for (int i = 0; i < n; i++)
            {
                float3 pos = new(means[i*3], means[i*3+1], means[i*3+2]);
                float3 shDC = new(dc[i*3], dc[i*3+1], dc[i*3+2]);
                float3 col = math.saturate(0.28209479f * shDC + 0.5f);
                float3 logS = new(scales[i*3], scales[i*3+1], scales[i*3+2]);
                float3 expS = new(math.exp(logS.x), math.exp(logS.y), math.exp(logS.z));
                float alpha = 1f / (1f + math.exp(-opac[i]));

                Debug.Log($"[GSplat-Diag] sector={sectorId} [{tag}] #{i}: " +
                          $"pos=({pos.x:F3},{pos.y:F3},{pos.z:F3}) " +
                          $"col=({col.x:F2},{col.y:F2},{col.z:F2}) " +
                          $"alpha={alpha:F3} " +
                          $"scale=({expS.x:F4},{expS.y:F4},{expS.z:F4}) " +
                          $"logS=({logS.x:F2},{logS.y:F2},{logS.z:F2})");
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
        bool _loggedVertexRange;

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

            if (!_loggedVertexRange)
            {
                _loggedVertexRange = true;
                int sampleCount = Mathf.Min(numVertices, 2048);
                var sampleData = new byte[sampleCount * 32];
                vertBuf.GetData(sampleData, 0, 0, sampleCount);
                Vector3 pMin = Vector3.one * float.MaxValue;
                Vector3 pMax = Vector3.one * float.MinValue;
                for (int i = 0; i < sampleCount; i++)
                {
                    float px = System.BitConverter.ToSingle(sampleData, i * 32 + 0);
                    float py = System.BitConverter.ToSingle(sampleData, i * 32 + 4);
                    float pz = System.BitConverter.ToSingle(sampleData, i * 32 + 8);
                    if (px < pMin.x) pMin.x = px;
                    if (py < pMin.y) pMin.y = py;
                    if (pz < pMin.z) pMin.z = pz;
                    if (px > pMax.x) pMax.x = px;
                    if (py > pMax.y) pMax.y = py;
                    if (pz > pMax.z) pMax.z = pz;
                }
                Debug.Log($"[GSplatManager] Vertex range diagnostic: {numVertices} verts, " +
                          $"sample={sampleCount}, min={pMin}, max={pMax}");
            }

            var bounds = _scheduler.GetSectorBounds(sectorId);
            Vector3 sMin = bounds.center - bounds.extents;
            Vector3 sMax = bounds.center + bounds.extents;

            float vi_voxelSize = VolumeIntegrator.Instance?.VoxelSize ?? 0.05f;
            float initScale = Mathf.Log(vi_voxelSize * 1.0f);

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

        /// <summary>
        /// Randomly sample a keyframe, weighted by inverse distance to the sector.
        /// Ensures multi-view supervision rather than always training from the same viewpoint.
        /// </summary>
        KeyframeData? PickBestKeyframe(int sectorId)
        {
            if (_keyframes.Count == 0) return null;

            var bounds = _scheduler.GetSectorBounds(sectorId);

            // Compute weights: inverse distance with a floor so distant views still participate
            float totalWeight = 0f;
            for (int i = 0; i < _keyframes.Count; i++)
            {
                float dist = Vector3.Distance(bounds.center, _keyframes[i].CamPos);
                float weight = 1f / (0.5f + dist);
                totalWeight += weight;
            }

            if (totalWeight <= 0f)
                return _keyframes[UnityEngine.Random.Range(0, _keyframes.Count)];

            // Weighted random selection
            float r = UnityEngine.Random.value * totalWeight;
            float cumulative = 0f;
            for (int i = 0; i < _keyframes.Count; i++)
            {
                float dist = Vector3.Distance(bounds.center, _keyframes[i].CamPos);
                cumulative += 1f / (0.5f + dist);
                if (r <= cumulative)
                    return _keyframes[i];
            }

            return _keyframes[_keyframes.Count - 1];
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
                if (kf.Texture != null) { kf.Texture.Release(); Destroy(kf.Texture); }
            }
            _keyframes.Clear();

            _trainer?.Dispose();
            _scheduler?.Dispose();
        }
    }
}
