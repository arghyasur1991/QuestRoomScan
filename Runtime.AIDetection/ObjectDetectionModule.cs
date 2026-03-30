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
    /// frames, projects 2D detections to 3D world space using our own ICameraProvider +
    /// DepthCapture, and feeds results into SceneObjectRegistry.
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

        private IDetectionModel _model;
        private ICameraProvider _camera;
        private DepthCapture _depth;
        private SceneObjectRegistry _registry;
        private bool _running;
        private bool _busy;
        private int _detectionCount;

        public string ModuleName => "Object Detection";
        public bool IsRunning => _running;
        public int DetectionCount => _detectionCount;

        public event Action<SceneObject> OnObjectDetected;

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _camera = scanner.GetComponent<ICameraProvider>();
            _depth = scanner.GetComponent<DepthCapture>();
        }

        public async void StartDetection(SceneObjectRegistry registry)
        {
            _registry = registry;
            if (_model == null && modelAsset != null)
            {
                _model = new YoloDetectionModel(
                    modelAsset, classLabels, backend, splitOverFrames,
                    layersPerFrame, minConfidence);
                try
                {
                    await _model.LoadAsync();
                    Logger.Info($"[ObjectDetection] Model loaded: {_model.ModelName}");
                }
                catch (Exception e)
                {
                    Logger.Error($"[ObjectDetection] Model load failed: {e.Message}");
                    return;
                }
            }
            _running = _model is { IsLoaded: true };
        }

        public void StopDetection()
        {
            _running = false;
        }

        public List<SceneObject> GetAccumulatedDetections()
        {
            return _registry?.FindBySource(SceneObjectSource.AIDetection) ?? new List<SceneObject>();
        }

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
                if (detections == null) return;

                foreach (var d in detections)
                {
                    if (d.confidence < minConfidence) continue;

                    var worldPos = ProjectToWorld(d.boundingBox);
                    if (!worldPos.HasValue) continue;

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
                    _registry?.Add(obj);
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

        /// <summary>
        /// 2D bbox center → 3D world position using camera intrinsics + depth.
        /// </summary>
        private Vector3? ProjectToWorld(Rect bbox)
        {
            if (_camera == null || _depth == null) return null;

            float cx = bbox.center.x;
            float cy = bbox.center.y;
            var res = _camera.CurrentResolution;
            if (res.x < 1 || res.y < 1) return null;

            var focal = _camera.FocalLength;
            var pp = _camera.PrincipalPoint;

            var dir = new Vector3(
                (cx - pp.x) / focal.x,
                -(cy - pp.y) / focal.y,
                1f).normalized;

            float uvx = cx / res.x;
            float uvy = 1f - cy / res.y;
            float depth = SampleDepth(new Vector2(uvx, uvy));
            if (depth <= 0.1f || depth > 20f) return null;

            var camPose = _camera.CameraPose;
            var worldDir = camPose.rotation * dir;
            return camPose.position + worldDir * depth;
        }

        private float SampleDepth(Vector2 uv)
        {
            // DepthCapture may expose depth sampling in the future; for now use
            // the environment depth texture approach from AROcclusionManager.
            // This is a placeholder that returns a fixed estimate.
            return 2.5f;
        }

        private Vector3 EstimateWorldScale(Rect bbox, Vector3 worldPos)
        {
            if (_camera == null) return Vector3.one * 0.3f;

            var res = _camera.CurrentResolution;
            var focal = _camera.FocalLength;
            float depth = (worldPos - _camera.CameraPose.position).magnitude;

            float widthM = bbox.width / focal.x * depth;
            float heightM = bbox.height / focal.y * depth;
            float depthM = (widthM + heightM) * 0.25f;

            return new Vector3(widthM, heightM, depthM);
        }

        private void OnDestroy()
        {
            _model?.Dispose();
        }
    }
}
#endif
