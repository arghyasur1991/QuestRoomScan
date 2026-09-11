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
        [SerializeField, Tooltip("Min translation (m) from every saved keyframe to trigger a new capture. A frame is redundant only if it is close in BOTH position and rotation to a saved one.")]
        private float moveThreshold = 0.5f;

        [SerializeField, Tooltip("Min rotation (deg) from every saved keyframe to trigger a new capture")]
        private float rotateThresholdDeg = 25f;

        [SerializeField, Range(50, 100)]
        private int jpegQuality = 95;

        [SerializeField, Tooltip("Max angular velocity (deg/s) to accept a frame (rejects motion blur)")]
        private float maxAngularVelocity = 120f;

        [SerializeField, Tooltip("Min seconds between captures to prevent burst saves")]
        private float minCaptureInterval = 1f;

        [SerializeField, Tooltip("Skip a frame when the player's hands / forearms cover more than this fraction of the image. Smaller intrusions are kept and masked out of the texture bake per pixel.")]
        [Range(0.02f, 0.5f)] private float maxHandCoverage = 0.12f;

        /// <summary>Hand / forearm capsules recorded per frame (see <see cref="VolumeIntegrator.CopyHandCapsules"/>).</summary>
        public const int MaxBodyCapsules = 8;
        private readonly Vector4[] _capP0 = new Vector4[MaxBodyCapsules];
        private readonly Vector4[] _capP1 = new Vector4[MaxBodyCapsules];
        private int _skippedForHands;

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

            // Where the hands are right now, in this frame's own image. A
            // frame that is mostly hand is useless; a hand in the corner is
            // recorded so the bake can mask those pixels.
            int capCount = _scanner != null && _scanner.VolumeIntegrator != null
                ? _scanner.VolumeIntegrator.CopyHandCapsules(_capP0, _capP1) : 0;
            string caps = null;
            if (capCount > 0)
            {
                float coverage = HandCoverage(capCount, pos, rot, focalLen, currentRes);
                if (coverage > maxHandCoverage)
                {
                    if (++_skippedForHands <= 3 || _skippedForHands % 25 == 0)
                        Logger.Info($"KeyframeCollector: skipped frame, hands cover {coverage:P0} of the image ({_skippedForHands} so far)");
                    return;
                }
                caps = FormatCapsules(capCount);
            }

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
                        focalLen, principalPt, sensorRes, currentRes, caps));
            }
            else if (frame is Texture2D tex2d)
            {
                SaveKeyframeData(tex2d.EncodeToJPG(jpegQuality), id, timestamp,
                    pos, rot, focalLen, principalPt, sensorRes, currentRes, caps);
            }
        }

        /// <summary>
        /// Fraction of the image covered by the hand capsules, estimated as the
        /// union-free sum of each capsule's projected footprint: the segment
        /// between its end points swept by the projected radius, clipped to
        /// the image. Capsules behind the camera contribute nothing.
        /// </summary>
        float HandCoverage(int count, Vector3 camPos, Quaternion camRot, Vector2 focal, Vector2 res)
        {
            if (res.x <= 1f || res.y <= 1f || focal.x <= 1f) return 0f;
            Quaternion inv = Quaternion.Inverse(camRot);
            float imgArea = res.x * res.y;
            float total = 0f;
            for (int i = 0; i < count; i++)
            {
                Vector3 a = inv * ((Vector3)_capP0[i] - camPos);
                Vector3 b = inv * ((Vector3)_capP1[i] - camPos);
                float r = _capP0[i].w;
                // Camera looks down +Z in this convention (see the bake's projection).
                float za = Mathf.Max(a.z, 0.05f), zb = Mathf.Max(b.z, 0.05f);
                if (a.z < 0.05f && b.z < 0.05f) continue;
                Vector2 pa = new Vector2(focal.x * a.x / za, focal.y * a.y / za) + res * 0.5f;
                Vector2 pb = new Vector2(focal.x * b.x / zb, focal.y * b.y / zb) + res * 0.5f;
                float pr = focal.x * r / Mathf.Min(za, zb);
                // Footprint of the capsule's projection, clipped to the image rect.
                Vector2 min = Vector2.Min(pa, pb) - Vector2.one * pr;
                Vector2 max = Vector2.Max(pa, pb) + Vector2.one * pr;
                min = Vector2.Max(min, Vector2.zero);
                max = Vector2.Min(max, res);
                if (max.x <= min.x || max.y <= min.y) continue;
                // A capsule is thinner than its bounding box: length × 2r + disc ends.
                float len = Vector2.Distance(pa, pb);
                float footprint = len * 2f * pr + Mathf.PI * pr * pr;
                float box = (max.x - min.x) * (max.y - min.y);
                total += Mathf.Min(footprint, box);
            }
            return Mathf.Clamp01(total / imgArea);
        }

        string FormatCapsules(int count)
        {
            var ci = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new StringBuilder(count * 64);
            for (int i = 0; i < count; i++)
            {
                if (i > 0) sb.Append(';');
                var a = _capP0[i];
                var b = _capP1[i];
                sb.Append(a.x.ToString("F4", ci)).Append(' ').Append(a.y.ToString("F4", ci)).Append(' ').Append(a.z.ToString("F4", ci)).Append(' ')
                  .Append(b.x.ToString("F4", ci)).Append(' ').Append(b.y.ToString("F4", ci)).Append(' ').Append(b.z.ToString("F4", ci)).Append(' ')
                  .Append(a.w.ToString("F4", ci));
            }
            return sb.ToString();
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
            Vector2 sensorRes, Vector2 currentRes, string caps)
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
                    focalLen, principalPt, sensorRes, currentRes, caps);
            }
            catch (Exception e)
            {
                Logger.Error($"KeyframeCollector: encode error frame {id}: {e.Message}");
            }
        }

        private void SaveKeyframeData(byte[] jpgBytes, int id, float timestamp,
            Vector3 pos, Quaternion rot, Vector2 focalLen, Vector2 principalPt,
            Vector2 sensorRes, Vector2 currentRes, string caps = null)
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
                    // Hand / forearm capsules at capture: "x y z x y z r;..." (no
                    // commas — the manifest reader splits lines on them).
                    if (!string.IsNullOrEmpty(caps))
                        sb.Append(",\"cap\":\"").Append(caps).Append('"');
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
        /// Saves a pre-captured JPEG as a keyframe unconditionally (no motion/interval gates).
        /// Used by detection modules that need to capture the frame before async processing
        /// and only decide to save after results are known.
        /// Returns the assigned keyframe ID, or -1 if export is not configured.
        /// </summary>
        public int SaveCapturedKeyframe(byte[] jpgBytes, float timestamp,
            Vector3 pos, Quaternion rot, Vector2 focalLen, Vector2 principalPt,
            Vector2 sensorRes, Vector2 currentRes)
        {
            if (_exportDir == null || jpgBytes == null || jpgBytes.Length == 0) return -1;
            int id = _nextId++;
            _savedPositions.Add(pos);
            _savedRotations.Add(rot);
            SaveKeyframeData(jpgBytes, id, timestamp, pos, rot,
                focalLen, principalPt, sensorRes, currentRes);
            return id;
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
