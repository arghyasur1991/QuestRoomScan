#if HAS_AI_INFERENCE
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.AIDetection
{
    /// <summary>
    /// Scan-time object detection orchestrator. Subscribes to RoomScanner.ColorFrameProvided
    /// to receive camera frames with the exact same world-space pose and intrinsics used by
    /// the TSDF pipeline. Projects 2D YOLO detections to 3D via the same pinhole model
    /// (including crop correction) and feeds results into SceneObjectRegistry.
    /// </summary>
    public class ObjectDetectionModule : MonoBehaviour, IRoomScanModule
    {
        [Header("Model")]
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private TextAsset classLabels;
        [SerializeField] private ComputeShader nmsComputeShader;

        [Header("Detection Settings")]
        [SerializeField] private int detectEveryNFrames = 5;
        [SerializeField, Range(0f, 1f)] private float minConfidence = 0.5f;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;
        [SerializeField] private bool splitOverFrames = true;
        [SerializeField, Range(1, 100)] private int layersPerFrame = 22;
        [Tooltip("Max input resolution for YOLO. 640 = full quality, 320 = faster/less VRAM. 0 = use model default.")]
        [SerializeField] private int maxInputResolution = 640;

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

                var registry = _scanner?.SceneObjectRegistry;
                if (registry == null) return;

                foreach (var d in detections)
                {
                    if (d.confidence < minConfidence) continue;

                    var worldPos = ProjectToWorld(d.boundingBox, cam);
                    if (!worldPos.HasValue) continue;

                    var existing = FindExisting(registry, d.label, worldPos.Value);
                    if (existing != null)
                    {
                        UpdateExisting(registry, existing, worldPos.Value, d);
                        continue;
                    }

                    var worldScale = EstimateWorldScale(d.boundingBox, worldPos.Value, cam);

                    var obj = new SceneObject
                    {
                        id = $"ai_{d.label}_{_detectionCount}",
                        label = d.label,
                        source = SceneObjectSource.AIDetection,
                        surfaceType = SurfaceType.Unknown,
                        confidence = d.confidence,
                        position = worldPos.Value,
                        rotation = Quaternion.identity,
                        size = worldScale,
                        classId = d.classId,
                        imageBoundingBox = d.boundingBox
                    };

                    _detectionCount++;
                    _observationCounts[obj.id] = 1;
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

        // ── 2D → 3D projection — exact inverse of the TSDF ProjectToCameraUV ──

        private static Vector3? ProjectToWorld(Rect bbox, CameraSnapshot cam)
        {
            if (cam.sensorRes.x < 1 || cam.sensorRes.y < 1) return null;

            // Step 1: Convert bbox center from delivered-image pixels to sensor-space pixels.
            // This is the inverse of the crop correction in VolumeIntegration.compute:
            //   scaleFactor = currentRes / sensorRes;
            //   scaleFactor /= max(scaleFactor.x, scaleFactor.y);
            //   cropMin = sensorRes * (1 - scaleFactor) * 0.5;
            //   cropSize = sensorRes * scaleFactor;
            //   uv = (sensorPt - cropMin) / cropSize;
            // Inverse: sensorPt = uv * cropSize + cropMin
            Vector2 scaleFactor = cam.currentRes / cam.sensorRes;
            float maxScale = Mathf.Max(scaleFactor.x, scaleFactor.y);
            scaleFactor /= maxScale;
            Vector2 cropMin = cam.sensorRes * (Vector2.one - scaleFactor) * 0.5f;
            Vector2 cropSize = cam.sensorRes * scaleFactor;

            // bbox is in delivered-image pixel coords → convert to UV [0,1]
            float u = bbox.center.x / cam.currentRes.x;
            float v = bbox.center.y / cam.currentRes.y;

            // UV → sensor pixel coords
            float sensorX = u * cropSize.x + cropMin.x;
            float sensorY = v * cropSize.y + cropMin.y;

            // Step 2: Pinhole unproject (same convention as the compute shader — no Y negation)
            var localDir = new Vector3(
                (sensorX - cam.principal.x) / cam.focal.x,
                (sensorY - cam.principal.y) / cam.focal.y,
                1f).normalized;

            // Step 3: Camera local → world (pose is already in world space via TrackingToWorld)
            var worldDir = cam.pose.rotation * localDir;

            // Step 4: Raycast for depth
            if (Physics.Raycast(new Ray(cam.pose.position, worldDir), out var hit, 20f))
                return hit.point;

            return cam.pose.position + worldDir * 2.5f;
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

        private static Vector3 EstimateWorldScale(Rect bbox, Vector3 worldPos, CameraSnapshot cam)
        {
            float depth = (worldPos - cam.pose.position).magnitude;

            float widthM = bbox.width / cam.focal.x * depth;
            float heightM = bbox.height / cam.focal.y * depth;
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
        }
    }
}
#endif
