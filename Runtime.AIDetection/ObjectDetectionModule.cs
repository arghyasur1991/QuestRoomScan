#if HAS_AI_INFERENCE
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.AIDetection
{
    /// <summary>
    /// Scan-time object detection orchestrator. Runs inference on passthrough camera
    /// frames, projects 2D detections to 3D world space using ICameraProvider + raycasting,
    /// and feeds results into SceneObjectRegistry.
    /// Discovered automatically by RoomScanner via <see cref="IRoomScanModule"/>.
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
        private ICameraProvider _camera;
        private bool _running;
        private bool _busy;
        private int _detectionCount;

        // Track observation counts per object for weighted averaging
        private readonly Dictionary<string, int> _observationCounts = new();

        public string ModuleName => "Object Detection";
        public bool IsRunning => _running;
        public int DetectionCount => _detectionCount;

        public event Action<SceneObject> OnObjectDetected;

        // ── IRoomScanModule lifecycle ────────────────────────────────

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _scanner = scanner;
            _camera = scanner.GetComponent<ICameraProvider>();
            if (_camera == null)
                _camera = scanner.GetComponentInChildren<ICameraProvider>();

            Logger.Info($"[ObjectDetection] Init — model={(modelAsset != null ? "assigned" : "MISSING")}, " +
                        $"camera={(_camera != null ? _camera.GetType().Name : "MISSING")}, " +
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
            await StartDetection(registry);
        }

        public void OnScanStopped()
        {
            StopDetection();
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
            if (!_running || _busy || _camera == null || !_camera.IsReady) return;
            if (Time.frameCount % detectEveryNFrames != 0) return;

            // Capture camera state NOW, at the same instant as the frame texture.
            // Inference is async and will complete many frames later — we need
            // the pose from when the image was actually taken.
            var framePose = _camera.CameraPose;
            var frameFocal = _camera.FocalLength;
            var framePP = _camera.PrincipalPoint;
            var frameRes = _camera.CurrentResolution;
            var frame = _camera.CurrentFrame;

            if (frame == null) return;
            _ = RunDetection(frame, framePose, frameFocal, framePP, frameRes);
        }

        private async Task RunDetection(Texture frame, Pose cameraPose,
            Vector2 focal, Vector2 principalPoint, Vector2 resolution)
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

                    var worldPos = ProjectToWorld(d.boundingBox, cameraPose,
                        focal, principalPoint, resolution);
                    if (!worldPos.HasValue) continue;

                    var existing = FindExisting(registry, d.label, worldPos.Value);
                    if (existing != null)
                    {
                        UpdateExisting(registry, existing, worldPos.Value, d);
                        continue;
                    }

                    var worldScale = EstimateWorldScale(d.boundingBox, worldPos.Value,
                        cameraPose, focal);

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

        // ── 2D → 3D projection using CAPTURED camera pose ───────────

        private Vector3? ProjectToWorld(Rect bbox, Pose cameraPose,
            Vector2 focal, Vector2 principalPoint, Vector2 resolution)
        {
            if (resolution.x < 1 || resolution.y < 1) return null;

            float cx = bbox.center.x;
            float cy = bbox.center.y;

            var localDir = new Vector3(
                (cx - principalPoint.x) / focal.x,
                -(cy - principalPoint.y) / focal.y,
                1f).normalized;

            var worldDir = cameraPose.rotation * localDir;

            if (Physics.Raycast(new Ray(cameraPose.position, worldDir), out var hit, 20f))
                return hit.point;

            return cameraPose.position + worldDir * 2.5f;
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

            // Exponential moving average — early observations have more weight,
            // converges to stable position as count grows
            float alpha = positionSmoothingAlpha / Mathf.Sqrt(count);
            existing.position = Vector3.Lerp(existing.position, newPos, alpha);
            existing.confidence = Mathf.Max(existing.confidence, d.confidence);
            registry.Update(existing);
        }

        private Vector3 EstimateWorldScale(Rect bbox, Vector3 worldPos,
            Pose cameraPose, Vector2 focal)
        {
            float depth = (worldPos - cameraPose.position).magnitude;

            float widthM = bbox.width / focal.x * depth;
            float heightM = bbox.height / focal.y * depth;
            float depthM = (widthM + heightM) * 0.25f;

            return new Vector3(
                Mathf.Max(widthM, 0.05f),
                Mathf.Max(heightM, 0.05f),
                Mathf.Max(depthM, 0.05f));
        }

        private void OnDestroy()
        {
            StopDetection();
            _model?.Dispose();
        }
    }
}
#endif
