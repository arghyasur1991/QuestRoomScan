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
        private float moveThreshold = 0.3f;

        [SerializeField, Tooltip("Min rotation (deg) from any saved keyframe to trigger a new capture")]
        private float rotateThresholdDeg = 20f;

        [SerializeField, Range(50, 100)]
        private int jpegQuality = 90;

        [SerializeField, Tooltip("Max angular velocity (deg/s) to accept a frame (rejects motion blur)")]
        private float maxAngularVelocity = 120f;

        private string _exportDir;
        private string _imagesDir;
        private string _manifestPath;

        private readonly List<Vector3> _savedPositions = new();
        private readonly List<Quaternion> _savedRotations = new();
        private int _nextId;
        private int _pendingWrites;
        private Quaternion _prevRot;
        private float _prevRotTime;
        private bool _initialized;

        public int SavedCount => _nextId;
        public string ExportDirectory => _exportDir;

        private void Start()
        {
            _exportDir = Path.Combine(Application.persistentDataPath, "GSExport");
            _imagesDir = Path.Combine(_exportDir, "images");
            _manifestPath = Path.Combine(_exportDir, "frames.jsonl");

            Directory.CreateDirectory(_imagesDir);

            _prevRot = Quaternion.identity;
            _prevRotTime = Time.time;
            _initialized = true;

            Debug.Log($"[RoomScan] KeyframeCollector: export dir={_exportDir}");
        }

        /// <summary>
        /// Called by RoomScanner each integration tick with the current camera data.
        /// Determines whether to save a new keyframe based on motion thresholds.
        /// </summary>
        public void TrySaveKeyframe(Texture frame, Vector3 pos, Quaternion rot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (!_initialized || frame == null) return;

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
                Debug.LogWarning($"[RoomScan] KeyframeCollector: readback error for frame {id}");
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
                Debug.LogError($"[RoomScan] KeyframeCollector: encode error frame {id}: {e.Message}");
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
                        Debug.Log($"[RoomScan] KeyframeCollector: saved frame {id} ({jpgBytes.Length / 1024}KB)");
                }
                catch (Exception e)
                {
                    Debug.LogError($"[RoomScan] KeyframeCollector: write error frame {id}: {e.Message}");
                }
            });
        }

        /// <summary>
        /// Clears the export directory for a fresh capture session.
        /// </summary>
        public void ClearExport()
        {
            if (Directory.Exists(_exportDir))
                Directory.Delete(_exportDir, true);
            Directory.CreateDirectory(_imagesDir);
            _savedPositions.Clear();
            _savedRotations.Clear();
            _nextId = 0;
            Debug.Log("[RoomScan] KeyframeCollector: export cleared");
        }
    }
}
