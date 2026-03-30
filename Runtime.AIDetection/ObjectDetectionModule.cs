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

        [Header("Detection Settings")]
        [SerializeField] private int detectEveryNFrames = 5;
        [SerializeField, Range(0f, 1f)] private float minConfidence = 0.5f;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;
        [SerializeField] private bool splitOverFrames = true;
        [SerializeField, Range(1, 100)] private int layersPerFrame = 22;

        [Header("Deduplication")]
        [Tooltip("Detections closer than this distance to an existing object of the same class are merged")]
        [SerializeField] private float mergeDistanceM = 0.4f;

        private IDetectionModel _model;
        private RoomScanner _scanner;
        private ICameraProvider _camera;
        private bool _running;
        private bool _busy;
        private int _detectionCount;

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
                    modelAsset, classLabels, backend, splitOverFrames,
                    layersPerFrame, minConfidence);
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
            _ = RunDetection();
        }

        private async Task RunDetection()
        {
            _busy = true;
            try
            {
                var frame = _camera.CurrentFrame;
                if (frame == null) return;

                var detections = await _model.DetectAsync(frame);
                if (detections == null || detections.Length == 0) return;

                var registry = _scanner?.SceneObjectRegistry;
                if (registry == null) return;

                foreach (var d in detections)
                {
                    if (d.confidence < minConfidence) continue;

                    var worldPos = ProjectToWorld(d.boundingBox);
                    if (!worldPos.HasValue) continue;

                    if (IsDuplicate(registry, d.label, worldPos.Value))
                        continue;

                    var worldScale = EstimateWorldScale(d.boundingBox, worldPos.Value);

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

        // ── 2D → 3D projection using camera intrinsics + raycasting ─

        private Vector3? ProjectToWorld(Rect bbox)
        {
            if (_camera == null) return null;

            var res = _camera.CurrentResolution;
            if (res.x < 1 || res.y < 1) return null;

            float cx = bbox.center.x;
            float cy = bbox.center.y;
            var focal = _camera.FocalLength;
            var pp = _camera.PrincipalPoint;

            var localDir = new Vector3(
                (cx - pp.x) / focal.x,
                -(cy - pp.y) / focal.y,
                1f).normalized;

            var camPose = _camera.CameraPose;
            var worldDir = camPose.rotation * localDir;

            // Raycast against scene geometry for accurate depth
            if (Physics.Raycast(new Ray(camPose.position, worldDir), out var hit, 20f))
                return hit.point;

            // Fallback: project at estimated average room depth
            return camPose.position + worldDir * 2.5f;
        }

        private bool IsDuplicate(SceneObjectRegistry registry, string label, Vector3 pos)
        {
            var nearby = registry.FindInRadius(pos, mergeDistanceM);
            foreach (var existing in nearby)
            {
                if (string.Equals(existing.label, label, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        private Vector3 EstimateWorldScale(Rect bbox, Vector3 worldPos)
        {
            if (_camera == null) return Vector3.one * 0.3f;

            var focal = _camera.FocalLength;
            float depth = (worldPos - _camera.CameraPose.position).magnitude;

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
