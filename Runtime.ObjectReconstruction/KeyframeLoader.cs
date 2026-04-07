#if HAS_ONNXRUNTIME
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Parsed detection entry from detections.jsonl.
    /// </summary>
    public struct DetectionEntry
    {
        public int keyframeId;
        public string objectId;
        public string label;
        public float confidence;
        public int classId;
        public Rect bbox;
        public Vector3 worldPos;
        public Vector3 scale;
    }

    /// <summary>
    /// Parsed keyframe metadata from frames.jsonl.
    /// </summary>
    public struct KeyframeEntry
    {
        public int id;
        public Pose pose;
        public Vector2 focal;
        public Vector2 principal;
        public Vector2 sensorRes;
        public Vector2 currentRes;
    }

    /// <summary>
    /// Loads detection keyframes from a scan directory and crops object images for reconstruction.
    /// Handles the full pipeline: parse JSONL, load JPEG, crop bbox, pad to square, optional denoise.
    /// </summary>
    public static class KeyframeLoader
    {
        /// <summary>
        /// Parses detections.jsonl from the given keyframe directory.
        /// </summary>
        public static List<DetectionEntry> LoadDetections(string keyframeDir)
        {
            var results = new List<DetectionEntry>();
            string path = Path.Combine(keyframeDir, "detections.jsonl");
            if (!File.Exists(path)) return results;

            foreach (string line in File.ReadAllLines(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    results.Add(ParseDetection(line));
                }
                catch (Exception e)
                {
                    Logger.Warning($"[KeyframeLoader] Failed to parse detection: {e.Message}");
                }
            }
            return results;
        }

        /// <summary>
        /// Parses frames.jsonl from the given keyframe directory.
        /// </summary>
        public static Dictionary<int, KeyframeEntry> LoadFrames(string keyframeDir)
        {
            var results = new Dictionary<int, KeyframeEntry>();
            string path = Path.Combine(keyframeDir, "frames.jsonl");
            if (!File.Exists(path)) return results;

            foreach (string line in File.ReadAllLines(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var entry = ParseFrame(line);
                    results[entry.id] = entry;
                }
                catch (Exception e)
                {
                    Logger.Warning($"[KeyframeLoader] Failed to parse frame: {e.Message}");
                }
            }
            return results;
        }

        /// <summary>
        /// Loads a keyframe JPEG and crops to the detection bounding box, padded to square.
        /// The bbox is in delivered-frame pixel coordinates (same space as the saved JPEG).
        /// Applies a simple 3x3 median-approximation denoise if <paramref name="denoise"/> is true.
        /// </summary>
        public static async Task<Texture2D> LoadAndCropAsync(
            string keyframeDir, DetectionEntry detection, bool denoise = true)
        {
            string imgPath = Path.Combine(keyframeDir, "images",
                $"{detection.keyframeId:D6}.jpg");

            if (!File.Exists(imgPath))
            {
                Logger.Warning($"[KeyframeLoader] Keyframe image not found: {imgPath}");
                return null;
            }

            byte[] jpgBytes = await Task.Run(() => File.ReadAllBytes(imgPath));

            var fullFrame = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (!fullFrame.LoadImage(jpgBytes))
            {
                UnityEngine.Object.Destroy(fullFrame);
                Logger.Warning($"[KeyframeLoader] Failed to decode JPEG: {imgPath}");
                return null;
            }

            var cropped = CropAndPadSquare(fullFrame, detection.bbox);
            UnityEngine.Object.Destroy(fullFrame);

            if (cropped == null) return null;

            if (denoise)
                cropped = ApplySimpleDenoise(cropped);

            return cropped;
        }

        /// <summary>
        /// Crops the texture to the given bbox (in pixel coordinates) and pads to a square.
        /// Padding uses gray (128,128,128) to match TripoSR's expected background.
        /// </summary>
        private static Texture2D CropAndPadSquare(Texture2D source, Rect bbox)
        {
            int srcW = source.width;
            int srcH = source.height;

            // Clamp bbox to image bounds
            int x = Mathf.Clamp(Mathf.FloorToInt(bbox.x), 0, srcW - 1);
            int y = Mathf.Clamp(Mathf.FloorToInt(bbox.y), 0, srcH - 1);
            int w = Mathf.Clamp(Mathf.CeilToInt(bbox.width), 1, srcW - x);
            int h = Mathf.Clamp(Mathf.CeilToInt(bbox.height), 1, srcH - y);

            // Unity GetPixels uses bottom-left origin; YOLO bbox uses top-left.
            // Flip Y: Unity row 0 = bottom of image.
            int unityY = srcH - y - h;
            unityY = Mathf.Clamp(unityY, 0, srcH - h);

            Color[] cropPixels;
            try
            {
                cropPixels = source.GetPixels(x, unityY, w, h);
            }
            catch (Exception e)
            {
                Logger.Warning($"[KeyframeLoader] GetPixels failed: {e.Message}");
                return null;
            }

            int sqSize = Mathf.Max(w, h);
            var result = new Texture2D(sqSize, sqSize, TextureFormat.RGBA32, false);

            // Fill with gray background
            var grayPixels = new Color[sqSize * sqSize];
            var gray = new Color(0.5f, 0.5f, 0.5f, 1f);
            for (int i = 0; i < grayPixels.Length; i++)
                grayPixels[i] = gray;
            result.SetPixels(grayPixels);

            // Center the crop in the square
            int padX = (sqSize - w) / 2;
            int padY = (sqSize - h) / 2;
            result.SetPixels(padX, padY, w, h, cropPixels);
            result.Apply();

            return result;
        }

        /// <summary>
        /// Simple 3x3 box-blur denoise. Reduces high-frequency sensor noise from
        /// Quest passthrough cameras while preserving edges reasonably well.
        /// One pass is enough — aggressive smoothing loses detail for TripoSR.
        /// </summary>
        private static Texture2D ApplySimpleDenoise(Texture2D input)
        {
            int w = input.width;
            int h = input.height;
            var src = input.GetPixels();
            var dst = new Color[src.Length];

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float r = 0, g = 0, b = 0;
                int count = 0;
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    var c = src[ny * w + nx];
                    r += c.r; g += c.g; b += c.b;
                    count++;
                }
                float inv = 1f / count;
                dst[y * w + x] = new Color(r * inv, g * inv, b * inv, 1f);
            }

            var result = new Texture2D(w, h, TextureFormat.RGBA32, false);
            result.SetPixels(dst);
            result.Apply();
            UnityEngine.Object.Destroy(input);
            return result;
        }

        // ── JSONL Parsing (minimal, no external JSON dependency) ──

        private static DetectionEntry ParseDetection(string json)
        {
            var e = new DetectionEntry();
            e.keyframeId = ReadInt(json, "keyframe_id");
            e.objectId = ReadString(json, "obj_id");
            e.label = ReadString(json, "label");
            e.confidence = ReadFloat(json, "confidence");
            e.classId = ReadInt(json, "class_id");
            e.bbox = new Rect(
                ReadFloat(json, "bbox_x"), ReadFloat(json, "bbox_y"),
                ReadFloat(json, "bbox_w"), ReadFloat(json, "bbox_h"));
            e.worldPos = new Vector3(
                ReadFloat(json, "world_x"), ReadFloat(json, "world_y"),
                ReadFloat(json, "world_z"));
            e.scale = new Vector3(
                ReadFloat(json, "scale_x"), ReadFloat(json, "scale_y"),
                ReadFloat(json, "scale_z"));
            return e;
        }

        private static KeyframeEntry ParseFrame(string json)
        {
            var e = new KeyframeEntry();
            e.id = ReadInt(json, "id");
            e.pose = new Pose(
                new Vector3(ReadFloat(json, "px"), ReadFloat(json, "py"), ReadFloat(json, "pz")),
                new Quaternion(ReadFloat(json, "qx"), ReadFloat(json, "qy"),
                    ReadFloat(json, "qz"), ReadFloat(json, "qw")));
            e.focal = new Vector2(ReadFloat(json, "fx"), ReadFloat(json, "fy"));
            e.principal = new Vector2(ReadFloat(json, "cx"), ReadFloat(json, "cy"));
            e.sensorRes = new Vector2(ReadFloat(json, "sw"), ReadFloat(json, "sh"));
            e.currentRes = new Vector2(ReadFloat(json, "w"), ReadFloat(json, "h"));
            return e;
        }

        private static int ReadInt(string json, string key)
        {
            string val = ExtractValue(json, key);
            return int.TryParse(val, out int i) ? i : 0;
        }

        private static float ReadFloat(string json, string key)
        {
            string val = ExtractValue(json, key);
            return float.TryParse(val, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float f) ? f : 0f;
        }

        private static string ReadString(string json, string key)
        {
            string val = ExtractValue(json, key);
            if (val.Length >= 2 && val[0] == '"' && val[val.Length - 1] == '"')
                return val.Substring(1, val.Length - 2);
            return val;
        }

        private static string ExtractValue(string json, string key)
        {
            string pattern = $"\"{key}\":";
            int idx = json.IndexOf(pattern, StringComparison.Ordinal);
            if (idx < 0) return "";
            int start = idx + pattern.Length;
            while (start < json.Length && json[start] == ' ') start++;
            int end = start;
            if (end < json.Length && json[end] == '"')
            {
                end++;
                while (end < json.Length && json[end] != '"') end++;
                end++;
            }
            else
            {
                while (end < json.Length && json[end] != ',' && json[end] != '}') end++;
            }
            return json.Substring(start, end - start);
        }
    }
}
#endif
