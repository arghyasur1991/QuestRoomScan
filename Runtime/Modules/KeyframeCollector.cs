using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Automatically saves camera keyframes (JPEG + pose + intrinsics) to disk during scanning.
    /// Uses motion-based selection to avoid redundant captures. The export folder is always
    /// ready for adb pull and subsequent Gaussian Splat training.
    /// </summary>
    public class KeyframeCollector : MonoBehaviour
    {
        [SerializeField, Tooltip("Min translation (m) from any saved keyframe to trigger a new capture")]
        private float moveThreshold = 0.15f;

        [SerializeField, Tooltip("Min rotation (deg) from any saved keyframe to trigger a new capture")]
        private float rotateThresholdDeg = 10f;

        [SerializeField, Range(50, 100)]
        private int jpegQuality = 95;

        [SerializeField, Tooltip("Max angular velocity (deg/s) to accept a frame (rejects motion blur)")]
        private float maxAngularVelocity = 120f;

        [SerializeField, Tooltip("Min seconds between captures to prevent burst saves")]
        private float minCaptureInterval = 0.25f;

        private string _exportDir;
        private string _imagesDir;
        private string _manifestPath;

        private readonly List<Vector3> _savedPositions = new();
        private readonly List<Quaternion> _savedRotations = new();
        private int _nextId;
        private int _pendingWrites;
        private Quaternion _prevRot;
        private float _prevRotTime;
        private float _lastCaptureTime;
        private bool _initialized;

        /// <summary>Number of keyframes saved so far in this session.</summary>
        public int SavedCount => _nextId;

        /// <summary>Absolute path to the keyframe export directory on device.</summary>
        public string ExportDirectory => _exportDir;

        private RoomScanner _scanner;

        private void Start()
        {
            _prevRot = Quaternion.identity;
            _prevRotTime = Time.time;
            _initialized = true;

            _scanner = GetComponent<RoomScanner>();
            if (_scanner != null)
                _scanner.ColorFrameProvided += OnColorFrame;
        }

        /// <summary>
        /// Sets the keyframe export directory. Creates the directory structure if needed.
        /// Pass null to disable keyframe capture.
        /// </summary>
        public void SetExportDirectory(string dir)
        {
            if (string.IsNullOrEmpty(dir))
            {
                _exportDir = null;
                _imagesDir = null;
                _manifestPath = null;
                return;
            }

            _exportDir = dir;
            _imagesDir = Path.Combine(dir, "images");
            _manifestPath = Path.Combine(dir, "frames.jsonl");
            Directory.CreateDirectory(_imagesDir);
            Logger.Info($"KeyframeCollector: export dir={_exportDir}");
        }

        private void OnColorFrame(Texture frame, Pose pose, Vector2 focal, Vector2 principal,
            Vector2 sensor, Vector2 current)
        {
            TrySaveKeyframe(frame, pose.position, pose.rotation, focal, principal, sensor, current);
        }

        private void OnDestroy()
        {
            if (_scanner != null)
                _scanner.ColorFrameProvided -= OnColorFrame;
        }

        /// <summary>
        /// Called by RoomScanner each integration tick with the current camera data.
        /// Determines whether to save a new keyframe based on motion thresholds.
        /// </summary>
        public void TrySaveKeyframe(Texture frame, Vector3 pos, Quaternion rot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (!_initialized || frame == null || _exportDir == null) return;

            if (Time.time - _lastCaptureTime < minCaptureInterval) return;

            float dt = Time.time - _prevRotTime;
            if (dt > 0.001f)
            {
                float angVel = Quaternion.Angle(_prevRot, rot) / dt;
                _prevRot = rot;
                _prevRotTime = Time.time;
                if (angVel > maxAngularVelocity) return;
            }

            if (!ShouldCapture(pos, rot)) return;

            int id = _nextId++;
            _savedPositions.Add(pos);
            _savedRotations.Add(rot);
            _lastCaptureTime = Time.time;

            float timestamp = Time.realtimeSinceStartup;

            if (frame is RenderTexture rt)
            {
                _pendingWrites++;
                AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req =>
                    OnReadbackComplete(req, id, timestamp, pos, rot,
                        focalLen, principalPt, sensorRes, currentRes));
            }
            else if (frame is Texture2D tex2d)
            {
                SaveKeyframeData(tex2d.EncodeToJPG(jpegQuality), id, timestamp,
                    pos, rot, focalLen, principalPt, sensorRes, currentRes);
            }
        }

        private bool ShouldCapture(Vector3 pos, Quaternion rot)
        {
            for (int i = 0; i < _savedPositions.Count; i++)
            {
                float dist = Vector3.Distance(pos, _savedPositions[i]);
                float angle = Quaternion.Angle(rot, _savedRotations[i]);
                if (dist < moveThreshold && angle < rotateThresholdDeg)
                    return false;
            }
            return true;
        }

        private void OnReadbackComplete(AsyncGPUReadbackRequest req, int id, float timestamp,
            Vector3 pos, Quaternion rot, Vector2 focalLen, Vector2 principalPt,
            Vector2 sensorRes, Vector2 currentRes)
        {
            _pendingWrites--;
            if (req.hasError)
            {
                Logger.Warning($"KeyframeCollector: readback error for frame {id}");
                return;
            }

            try
            {
                var data = req.GetData<byte>();
                var tex = new Texture2D(req.width, req.height, TextureFormat.RGBA32, false);
                tex.LoadRawTextureData(data);
                tex.Apply();
                byte[] jpg = tex.EncodeToJPG(jpegQuality);
                Destroy(tex);

                SaveKeyframeData(jpg, id, timestamp, pos, rot,
                    focalLen, principalPt, sensorRes, currentRes);
            }
            catch (Exception e)
            {
                Logger.Error($"KeyframeCollector: encode error frame {id}: {e.Message}");
            }
        }

        private void SaveKeyframeData(byte[] jpgBytes, int id, float timestamp,
            Vector3 pos, Quaternion rot, Vector2 focalLen, Vector2 principalPt,
            Vector2 sensorRes, Vector2 currentRes)
        {
            Task.Run(() =>
            {
                try
                {
                    string imgPath = Path.Combine(_imagesDir, $"{id:D6}.jpg");
                    File.WriteAllBytes(imgPath, jpgBytes);

                    var sb = new StringBuilder(256);
                    sb.Append("{\"id\":").Append(id);
                    sb.Append(",\"ts\":").Append(timestamp.ToString("F3"));
                    sb.Append(",\"px\":").Append(pos.x.ToString("F6"));
                    sb.Append(",\"py\":").Append(pos.y.ToString("F6"));
                    sb.Append(",\"pz\":").Append(pos.z.ToString("F6"));
                    sb.Append(",\"qx\":").Append(rot.x.ToString("F6"));
                    sb.Append(",\"qy\":").Append(rot.y.ToString("F6"));
                    sb.Append(",\"qz\":").Append(rot.z.ToString("F6"));
                    sb.Append(",\"qw\":").Append(rot.w.ToString("F6"));
                    sb.Append(",\"fx\":").Append(focalLen.x.ToString("F4"));
                    sb.Append(",\"fy\":").Append(focalLen.y.ToString("F4"));
                    sb.Append(",\"cx\":").Append(principalPt.x.ToString("F4"));
                    sb.Append(",\"cy\":").Append(principalPt.y.ToString("F4"));
                    sb.Append(",\"sw\":").Append((int)sensorRes.x);
                    sb.Append(",\"sh\":").Append((int)sensorRes.y);
                    sb.Append(",\"w\":").Append((int)currentRes.x);
                    sb.Append(",\"h\":").Append((int)currentRes.y);
                    sb.Append('}');

                    lock (_manifestPath)
                    {
                        File.AppendAllText(_manifestPath, sb.ToString() + "\n");
                    }

                    if (id < 5 || id % 50 == 0)
                        Logger.Info($"KeyframeCollector: saved frame {id} ({jpgBytes.Length / 1024}KB)");
                }
                catch (Exception e)
                {
                    Logger.Error($"KeyframeCollector: write error frame {id}: {e.Message}");
                }
            });
        }

        /// <summary>
        /// Clears in-memory state only. Call before background file deletion.
        /// </summary>
        public void ClearInMemory()
        {
            _savedPositions.Clear();
            _savedRotations.Clear();
            _nextId = 0;
        }

    }
}
