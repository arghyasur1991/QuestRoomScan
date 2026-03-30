#if HAS_AI_INFERENCE
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.AIDetection
{
    /// <summary>
    /// Scan-time object detection orchestrator. Subscribes to RoomScanner.ColorFrameProvided
    /// to receive camera frames with the exact same world-space pose and intrinsics used by
    /// the TSDF pipeline. Projects 2D YOLO detections to 3D via DepthProjection.compute
    /// (reusing the same DepthKit.hlsl depth pipeline as VolumeIntegration) and feeds
    /// results into SceneObjectRegistry.
    /// </summary>
    public class ObjectDetectionModule : MonoBehaviour, IRoomScanModule
    {
        [Header("Model")]
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private TextAsset classLabels;
        [SerializeField] private ComputeShader nmsComputeShader;

        [Header("Depth Projection")]
        [SerializeField] internal ComputeShader depthProjectionShader;

        [Header("Detection Settings")]
        [SerializeField] private int detectEveryNFrames = 5;
        [SerializeField, Range(0f, 1f)] private float minConfidence = 0.5f;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;
        [SerializeField] private bool splitOverFrames = true;
        [SerializeField, Range(1, 100)] private int layersPerFrame = 22;
        [Tooltip("Max input resolution for YOLO. 640 = full quality, 320 = faster/less VRAM. 0 = use model default.")]
        [SerializeField] private int maxInputResolution = 640;

        [Header("Filtering")]
        [Tooltip("Skip detections whose bbox covers more than this fraction of frame (large objects = unreliable projection)")]
        [SerializeField, Range(0f, 1f)] private float maxBboxFrameFraction = 0.6f;
        [Tooltip("Max depth distance for a valid detection (meters). Beyond this, depth likely hit a wall behind the object.")]
        [SerializeField] private float maxDetectionDepthM = 5f;
        [Tooltip("Max unique AI objects per label. Prevents the same class spawning many times at different locations.")]
        [SerializeField] private int maxObjectsPerLabel = 2;
        [Tooltip("Skip AI detection for labels already present in MRUK (e.g. bed, couch)")]
        [SerializeField] private bool skipLabelsInMruk = true;

        [Header("Deduplication")]
        [Tooltip("Detections closer than this distance to an existing object of the same class are merged")]
        [SerializeField] private float mergeDistanceM = 0.8f;
        [Tooltip("How fast existing positions converge to new observations (0=ignore new, 1=snap)")]
        [SerializeField, Range(0f, 1f)] private float positionSmoothingAlpha = 0.3f;

        private IDetectionModel _model;
        private RoomScanner _scanner;
        private bool _running;
        private bool _busy;
        private int _detectionCount;
        private int _framesSinceLastDetect;

        // Latest camera frame snapshot from ColorFrameProvided
        private Texture _latestFrame;
        private CameraSnapshot _latestSnapshot;

        private readonly Dictionary<string, int> _observationCounts = new();
        private readonly Dictionary<string, int> _labelObjectCounts = new();
        private readonly HashSet<string> _mrukLabels = new(StringComparer.OrdinalIgnoreCase);

        // GPU depth projection buffers
        private const int MaxDetections = 64;
        private ComputeBuffer _raysBuffer;
        private ComputeBuffer _resultsBuffer;
        private int _depthProjectKernel = -1;
        private readonly Vector4[] _raysCpu = new Vector4[MaxDetections];
        private readonly Vector4[] _resultsCpu = new Vector4[MaxDetections];

        public string ModuleName => "Object Detection";
        public bool IsRunning => _running;
        public int DetectionCount => _detectionCount;

        public event Action<SceneObject> OnObjectDetected;

        /// <summary>
        /// All camera params frozen at frame capture time, matching the TSDF pipeline exactly.
        /// </summary>
        private struct CameraSnapshot
        {
            public Pose pose;          // world-space (already TrackingToWorld'd)
            public Vector2 focal;      // sensor-space focal length
            public Vector2 principal;  // sensor-space principal point
            public Vector2 sensorRes;  // native sensor resolution
            public Vector2 currentRes; // delivered frame resolution
        }

        // ── IRoomScanModule lifecycle ────────────────────────────────

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _scanner = scanner;

            Logger.Info($"[ObjectDetection] Init — model={(modelAsset != null ? "assigned" : "MISSING")}, " +
                        $"labels={(classLabels != null ? "assigned" : "MISSING")}");
            if (modelAsset == null)
                Logger.Warning("[ObjectDetection] No model asset assigned — AI detection will be inactive");
        }

        public async void OnScanStarted()
        {
            var registry = _scanner?.SceneObjectRegistry;
            if (registry == null || modelAsset == null)
            {
                Logger.Info("[ObjectDetection] Skipping — no registry or model asset");
                return;
            }

            _scanner.ColorFrameProvided += OnColorFrame;
            await StartDetection(registry);
        }

        public void OnScanStopped()
        {
            if (_scanner != null)
                _scanner.ColorFrameProvided -= OnColorFrame;
            StopDetection();
        }

        private void OnColorFrame(Texture frame, Pose worldPose,
            Vector2 focal, Vector2 principal, Vector2 sensorRes, Vector2 currentRes)
        {
            _latestFrame = frame;
            _latestSnapshot = new CameraSnapshot
            {
                pose = worldPose,
                focal = focal,
                principal = principal,
                sensorRes = sensorRes,
                currentRes = currentRes
            };
        }

        // ── Detection control ────────────────────────────────────────

        public async Task StartDetection(SceneObjectRegistry registry)
        {
            if (_model == null && modelAsset != null)
            {
                _model = new YoloDetectionModel(
                    modelAsset, classLabels, nmsComputeShader,
                    backend, splitOverFrames, layersPerFrame, minConfidence,
                    maxInputResolution: maxInputResolution);
                try
                {
                    await _model.LoadAsync();
                    Logger.Info($"[ObjectDetection] Model loaded: {_model.ModelName}, " +
                                $"{_model.ClassLabels?.Length ?? 0} classes");
                }
                catch (Exception e)
                {
                    Logger.Error($"[ObjectDetection] Model load failed: {e.Message}");
                    return;
                }
            }
            _running = _model is { IsLoaded: true };
            if (_running)
                Logger.Info("[ObjectDetection] Detection started");
        }

        public void StopDetection()
        {
            if (_running)
                Logger.Info($"[ObjectDetection] Stopped. Total detections: {_detectionCount}");
            _running = false;
        }

        public List<SceneObject> GetAccumulatedDetections()
        {
            var registry = _scanner?.SceneObjectRegistry;
            return registry?.FindBySource(SceneObjectSource.AIDetection) ?? new List<SceneObject>();
        }

        // ── Per-frame detection loop ─────────────────────────────────

        private void Update()
        {
            if (!_running || _busy || _latestFrame == null) return;
            if (++_framesSinceLastDetect < detectEveryNFrames) return;
            _framesSinceLastDetect = 0;

            // Freeze the frame and camera snapshot for this detection run
            var frame = _latestFrame;
            var snap = _latestSnapshot;
            _latestFrame = null;

            _ = RunDetection(frame, snap);
        }

        private async Task RunDetection(Texture frame, CameraSnapshot cam)
        {
            _busy = true;
            try
            {
                var detections = await _model.DetectAsync(frame);
                if (detections == null || detections.Length == 0) return;
                if (!DepthCapture.DepthAvailable) return;

                var registry = _scanner?.SceneObjectRegistry;
                if (registry == null) return;

                RefreshMrukLabels(registry);

                var cropData = ComputeCropData(cam);
                var validDetections = new List<Detection>();
                int rayCount = 0;

                foreach (var d in detections)
                {
                    if (d.confidence < minConfidence) continue;
                    if (rayCount >= MaxDetections) break;

                    // Skip labels already covered by MRUK
                    if (skipLabelsInMruk && _mrukLabels.Contains(d.label))
                        continue;

                    // Skip detections that fill too much of the frame (unreliable projection)
                    float bboxFractionW = d.boundingBox.width / cam.currentRes.x;
                    float bboxFractionH = d.boundingBox.height / cam.currentRes.y;
                    if (bboxFractionW > maxBboxFrameFraction || bboxFractionH > maxBboxFrameFraction)
                        continue;

                    var dir = BboxToWorldRay(d.boundingBox, cam, cropData);
                    if (!dir.HasValue) continue;

                    _raysCpu[rayCount] = new Vector4(dir.Value.x, dir.Value.y, dir.Value.z, 0);
                    validDetections.Add(d);
                    rayCount++;
                }

                if (rayCount == 0) return;

                var worldPositions = await ProjectRaysViaDepth(cam.pose.position, rayCount);
                if (worldPositions == null) return;

                // All AI positions are stored relative to the scan session anchor
                // so they stay world-locked across tracking corrections.
                var anchorT = _scanner.ScanAnchorTransform;
                if (anchorT == null) return;

                for (int i = 0; i < validDetections.Count; i++)
                {
                    var wp = worldPositions[i];
                    if (wp.w < 0.5f) continue;

                    var worldPos = new Vector3(wp.x, wp.y, wp.z);
                    var d = validDetections[i];

                    float depthDist = Vector3.Distance(worldPos, cam.pose.position);
                    if (depthDist > maxDetectionDepthM) continue;

                    var worldScale = EstimateWorldScale(d.boundingBox, worldPos, cam, cropData);

                    // Convert to anchor-local space for storage and dedup
                    var localPos = anchorT.InverseTransformPoint(worldPos);

                    var existing = FindExisting(registry, d.label, localPos);
                    if (existing != null)
                    {
                        UpdateExisting(registry, existing, localPos, d);
                        continue;
                    }

                    // Cap unique objects per label
                    _labelObjectCounts.TryGetValue(d.label, out int labelCount);
                    if (labelCount >= maxObjectsPerLabel) continue;

                    var obj = new SceneObject
                    {
                        id = $"ai_{d.label}_{_detectionCount}",
                        label = d.label,
                        source = SceneObjectSource.AIDetection,
                        surfaceType = SurfaceType.Unknown,
                        confidence = d.confidence,
                        position = localPos,
                        rotation = Quaternion.identity,
                        size = worldScale,
                        classId = d.classId,
                        imageBoundingBox = d.boundingBox
                    };

                    _detectionCount++;
                    _observationCounts[obj.id] = 1;
                    _labelObjectCounts[d.label] = labelCount + 1;
                    registry.Add(obj);
                    OnObjectDetected?.Invoke(obj);
                }
            }
            catch (Exception e)
            {
                Logger.Warning($"[ObjectDetection] Frame detection failed: {e.Message}");
            }
            finally
            {
                _busy = false;
            }
        }

        private void RefreshMrukLabels(SceneObjectRegistry registry)
        {
            _mrukLabels.Clear();
            foreach (var obj in registry.AllObjects)
            {
                if (obj.source == SceneObjectSource.MRUK)
                    _mrukLabels.Add(obj.label);
            }
        }

        // ── Crop correction data (shared between ray and scale calculations) ──

        private struct CropData
        {
            public Vector2 cropMin;
            public Vector2 cropSize;
        }

        private static CropData ComputeCropData(CameraSnapshot cam)
        {
            Vector2 scaleFactor = cam.currentRes / cam.sensorRes;
            float maxScale = Mathf.Max(scaleFactor.x, scaleFactor.y);
            scaleFactor /= maxScale;
            return new CropData
            {
                cropMin = cam.sensorRes * (Vector2.one - scaleFactor) * 0.5f,
                cropSize = cam.sensorRes * scaleFactor
            };
        }

        // ── 2D → world ray direction (CPU, same pinhole model as TSDF) ─────

        private static Vector3? BboxToWorldRay(Rect bbox, CameraSnapshot cam, CropData crop)
        {
            if (cam.sensorRes.x < 1 || cam.sensorRes.y < 1) return null;

            // YOLO bbox uses top-left origin; camera intrinsics use bottom-left origin.
            // Flip Y to match (confirmed by Meta's ObjectDetectionVisualizer).
            float u = bbox.center.x / cam.currentRes.x;
            float v = 1.0f - (bbox.center.y / cam.currentRes.y);

            float sensorX = u * crop.cropSize.x + crop.cropMin.x;
            float sensorY = v * crop.cropSize.y + crop.cropMin.y;

            var localDir = new Vector3(
                (sensorX - cam.principal.x) / cam.focal.x,
                (sensorY - cam.principal.y) / cam.focal.y,
                1f).normalized;

            return cam.pose.rotation * localDir;
        }

        // ── GPU depth projection (reuses DepthKit.hlsl, same as VolumeIntegration) ──

        private void EnsureDepthProjectionBuffers()
        {
            if (_raysBuffer != null) return;
            _raysBuffer = new ComputeBuffer(MaxDetections, sizeof(float) * 4);
            _resultsBuffer = new ComputeBuffer(MaxDetections, sizeof(float) * 4);
            _depthProjectKernel = depthProjectionShader.FindKernel("ProjectDetections");
        }

        private Task<Vector4[]> ProjectRaysViaDepth(Vector3 camOrigin, int count)
        {
            if (depthProjectionShader == null)
            {
                Logger.Warning("[ObjectDetection] No depthProjectionShader assigned");
                return Task.FromResult<Vector4[]>(null);
            }

            EnsureDepthProjectionBuffers();

            _raysBuffer.SetData(_raysCpu, 0, 0, count);

            depthProjectionShader.SetBuffer(_depthProjectKernel, "_Rays", _raysBuffer);
            depthProjectionShader.SetBuffer(_depthProjectKernel, "_Results", _resultsBuffer);
            depthProjectionShader.SetVector("_CamOrigin", camOrigin);
            depthProjectionShader.SetInt("_Count", count);

            int groups = (count + 63) / 64;
            depthProjectionShader.Dispatch(_depthProjectKernel, groups, 1, 1);

            var tcs = new TaskCompletionSource<Vector4[]>();
            AsyncGPUReadback.Request(_resultsBuffer, count * sizeof(float) * 4, 0, req =>
            {
                if (req.hasError)
                {
                    Logger.Warning("[ObjectDetection] GPU depth readback failed");
                    tcs.SetResult(null);
                    return;
                }
                var data = req.GetData<Vector4>();
                var result = new Vector4[count];
                data.CopyTo(result);
                tcs.SetResult(result);
            });

            return tcs.Task;
        }

        // ── Deduplication with position smoothing ────────────────────

        private SceneObject FindExisting(SceneObjectRegistry registry, string label, Vector3 pos)
        {
            var nearby = registry.FindInRadius(pos, mergeDistanceM);
            foreach (var existing in nearby)
            {
                if (existing.source != SceneObjectSource.AIDetection) continue;
                if (string.Equals(existing.label, label, StringComparison.OrdinalIgnoreCase))
                    return existing;
            }
            return null;
        }

        private void UpdateExisting(SceneObjectRegistry registry, SceneObject existing,
            Vector3 newPos, Detection d)
        {
            _observationCounts.TryGetValue(existing.id, out int count);
            count++;
            _observationCounts[existing.id] = count;

            float alpha = positionSmoothingAlpha / Mathf.Sqrt(count);
            existing.position = Vector3.Lerp(existing.position, newPos, alpha);
            existing.confidence = Mathf.Max(existing.confidence, d.confidence);
            registry.Update(existing);
        }

        private static Vector3 EstimateWorldScale(Rect bbox, Vector3 worldPos,
            CameraSnapshot cam, CropData crop)
        {
            float depth = (worldPos - cam.pose.position).magnitude;

            // Convert bbox dimensions from delivered-image pixels to sensor pixels via crop,
            // then use sensor-space focal length for angular size → world size.
            float cropScaleX = crop.cropSize.x / cam.currentRes.x;
            float cropScaleY = crop.cropSize.y / cam.currentRes.y;
            float widthM = (bbox.width * cropScaleX) / cam.focal.x * depth;
            float heightM = (bbox.height * cropScaleY) / cam.focal.y * depth;
            float depthM = (widthM + heightM) * 0.25f;

            return new Vector3(
                Mathf.Max(widthM, 0.05f),
                Mathf.Max(heightM, 0.05f),
                Mathf.Max(depthM, 0.05f));
        }

        private void OnDestroy()
        {
            if (_scanner != null)
                _scanner.ColorFrameProvided -= OnColorFrame;
            StopDetection();
            _model?.Dispose();
            _raysBuffer?.Release();
            _resultsBuffer?.Release();
        }
    }
}
#endif
