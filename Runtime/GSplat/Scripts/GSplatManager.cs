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

            // Motion gating: only capture keyframes when the camera has moved
            // enough for diverse multi-view supervision
            if (_hasLastKeyframePose)
            {
                float dist = Vector3.Distance(pos, _lastKeyframeCamPos);
                float angle = Quaternion.Angle(rot, _lastKeyframeCamRot);
                if (dist < MinKeyframeDist && angle < MinKeyframeAngle)
                    return;
            }

            Texture2D snap = ToTexture2D(frame);
            if (snap == null) return;

            // Resize to training resolution so the loss compares matching pixel grids
            Texture2D resized = ResizeToTrainingRes(snap);
            Destroy(snap);

            Matrix4x4 view = Matrix4x4.TRS(pos, rot, Vector3.one).inverse;

            // Flip Y-axis of view matrix: Unity uses Y-up but camera intrinsics
            // follow OpenCV convention (Y-down, origin at top-left). Without this,
            // the rendered image is vertically flipped relative to the keyframe,
            // causing the loss to compare mismatched pixels and produce garbage gradients.
            view[1, 0] = -view[1, 0];
            view[1, 1] = -view[1, 1];
            view[1, 2] = -view[1, 2];
            view[1, 3] = -view[1, 3];

            // Scale intrinsics from camera resolution to training resolution
            float scaleX = (float)trainingResWidth / currentRes.x;
            float scaleY = (float)trainingResHeight / currentRes.y;
            float fx = focalLen.x * scaleX;
            float fy = focalLen.y * scaleY;
            float cx = principalPt.x * scaleX;
            float cy = principalPt.y * scaleY;

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
                Texture = resized,
                ViewMatrix = view,
                ProjMatrix = fullProj,
                Fx = fx, Fy = fy,
                Cx = cx, Cy = cy,
                CamPos = pos
            });

            if (_keyframes.Count <= 1)
            {
                Debug.Log($"[GSplatManager] Keyframe #{_keyframes.Count}: " +
                          $"camRes={currentRes}, trainRes={trainingResWidth}x{trainingResHeight}, " +
                          $"texSize={resized.width}x{resized.height}, " +
                          $"rawFocal=({focalLen.x:F1},{focalLen.y:F1}), " +
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

        Texture2D ResizeToTrainingRes(Texture2D src)
        {
            if (src.width == trainingResWidth && src.height == trainingResHeight)
                return src;

            var rt = RenderTexture.GetTemporary(trainingResWidth, trainingResHeight, 0,
                RenderTextureFormat.ARGB32);
            Graphics.Blit(src, rt);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var resized = new Texture2D(trainingResWidth, trainingResHeight,
                TextureFormat.RGBA32, false);
            resized.ReadPixels(new Rect(0, 0, trainingResWidth, trainingResHeight), 0, 0, false);
            resized.Apply(false, true);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
            return resized;
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

            // Reset Adam state when switching to a new sector
            if (sectorId != _lastTrainedSectorId)
            {
                _trainer.ResetStep();
                _lastTrainedSectorId = sectorId;
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

            ref var trained = ref _scheduler.Sectors[sectorId];
            bool shouldLog = trained.TrainingIteration % 30 == 0 ||
                             trained.State == SectorState.SplatReady;

            if (shouldLog)
            {
                Debug.Log($"[GSplatManager] Training sector {sectorId}: iter={trained.TrainingIteration}/{targetItersPerSector}, " +
                          $"gaussians={buffers.CurrentCount}, state={trained.State}");
            }

            // Diagnostic: log rendered image + keyframe stats on first few iterations per sector
            if (trained.TrainingIteration <= 3 || trained.TrainingIteration == 30)
            {
                LogTrainingDiag(sectorId, trained.TrainingIteration, kf.Value, buffers);
            }

            if (trained.State == SectorState.SplatReady)
                LogGaussianDiagnostics(sectorId, buffers, "trained");

            if (!debugDisableMeshMask)
                UpdateMeshMask();
        }

        void LogTrainingDiag(int sectorId, int iter, KeyframeData kf, GSplatBuffers buffers)
        {
            // Read back rendered image average pixel (GPU → CPU readback, only for diagnostics)
            var rendered = _trainer.RenderedImage;
            if (rendered == null) return;

            int w = rendered.width, h = rendered.height;
            var prevRT = RenderTexture.active;
            RenderTexture.active = rendered;
            var readTex = new Texture2D(w, h, TextureFormat.RGBA32, false);
            readTex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            readTex.Apply(false);
            RenderTexture.active = prevRT;

            var rendPixels = readTex.GetPixels32();
            float rSum = 0, gSum = 0, bSum = 0;
            int nonBlack = 0;
            foreach (var px in rendPixels)
            {
                rSum += px.r; gSum += px.g; bSum += px.b;
                if (px.r > 5 || px.g > 5 || px.b > 5) nonBlack++;
            }
            int total = rendPixels.Length;
            float rAvg = rSum / total, gAvg = gSum / total, bAvg = bSum / total;

            // Keyframe average (texture is non-readable, blit to temp RT first)
            var kfRT = RenderTexture.GetTemporary(kf.Texture.width, kf.Texture.height, 0);
            Graphics.Blit(kf.Texture, kfRT);
            RenderTexture.active = kfRT;
            var kfRead = new Texture2D(kfRT.width, kfRT.height, TextureFormat.RGBA32, false);
            kfRead.ReadPixels(new Rect(0, 0, kfRT.width, kfRT.height), 0, 0, false);
            kfRead.Apply(false);
            RenderTexture.active = prevRT;
            var kfPixels = kfRead.GetPixels32();
            float krSum = 0, kgSum = 0, kbSum = 0;
            foreach (var px in kfPixels)
            {
                krSum += px.r; kgSum += px.g; kbSum += px.b;
            }
            int kTotal = kfPixels.Length;
            float krAvg = krSum / kTotal, kgAvg = kgSum / kTotal, kbAvg = kbSum / kTotal;
            Destroy(kfRead);
            RenderTexture.ReleaseTemporary(kfRT);

            // Gradient magnitude sample
            int n = Mathf.Min(buffers.CurrentCount, 4);
            var gMeans = new float[n * 3];
            buffers.GradMeans.GetData(gMeans, 0, 0, n * 3);
            float gradMag = 0;
            for (int i = 0; i < n * 3; i++) gradMag += gMeans[i] * gMeans[i];
            gradMag = Mathf.Sqrt(gradMag / (n * 3));

            // Top-half vs bottom-half brightness (detects Y-flip: if rendered top is dark
            // but keyframe top is bright, the images are vertically misaligned)
            float rendTopR = 0, rendBotR = 0, kfTopR = 0, kfBotR = 0;
            int halfH = h / 2;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float r = rendPixels[y * w + x].r;
                    if (y < halfH) rendTopR += r; else rendBotR += r;
                }
            int kHalfH = kf.Texture.height / 2;
            int kw = kf.Texture.width, kh = kf.Texture.height;
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
                      $"gradMeanRMS={gradMag:E2} " +
                      $"rendTopR={rendTopAvg:F1} rendBotR={rendBotAvg:F1} " +
                      $"kfTopR={kfTopAvg:F1} kfBotR={kfBotAvg:F1}");

            Destroy(readTex);
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
                if (kf.Texture != null) Destroy(kf.Texture);
            }
            _keyframes.Clear();

            _trainer?.Dispose();
            _scheduler?.Dispose();
        }
    }
}
