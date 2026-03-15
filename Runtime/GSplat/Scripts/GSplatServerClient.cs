using System;
using System.IO;
using System.IO.Compression;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// HTTP client for communicating with the PC-side GS training server (gs_server.py).
    /// Handles ZIP packaging of keyframes + point cloud, upload, status polling,
    /// and PLY download of trained Gaussians.
    /// </summary>
    public class GSplatServerClient : MonoBehaviour
    {
        [SerializeField] string serverUrl = "http://192.168.1.100:8420";
        [SerializeField, Range(1f, 30f)] float pollIntervalSeconds = 3f;

        public string ServerUrl
        {
            get => serverUrl;
            set => serverUrl = value.TrimEnd('/');
        }

        public bool IsUploading { get; private set; }
        public bool IsPolling { get; private set; }
        public bool IsDownloading { get; private set; }
        public TrainingStatus LastStatus { get; private set; } = new() { state = "idle" };

        public event Action<TrainingStatus> StatusUpdated;
        public event Action<string> Error;

        [Serializable]
        public class TrainingStatus
        {
            public string state;
            public float progress;
            public string message;
        }

        /// <summary>
        /// Packages the GSExport directory into a ZIP and uploads to the training server.
        /// </summary>
        public async Task<bool> UploadTrainingData()
        {
            if (IsUploading)
            {
                Debug.LogWarning("[GSplatServerClient] Upload already in progress");
                return false;
            }

            IsUploading = true;
            try
            {
                string exportDir = Path.Combine(Application.persistentDataPath, "GSExport");
                if (!Directory.Exists(exportDir))
                {
                    Debug.LogError("[GSplatServerClient] GSExport directory not found");
                    Error?.Invoke("GSExport directory not found");
                    return false;
                }

                string framesFile = Path.Combine(exportDir, "frames.jsonl");
                if (!File.Exists(framesFile))
                {
                    Debug.LogError("[GSplatServerClient] frames.jsonl not found");
                    Error?.Invoke("frames.jsonl not found — no keyframes collected");
                    return false;
                }

                Debug.Log("[GSplatServerClient] Creating ZIP from GSExport...");
                byte[] zipData = await Task.Run(() => CreateZip(exportDir));
                Debug.Log($"[GSplatServerClient] ZIP created: {zipData.Length / (1024 * 1024)}MB");

                string url = $"{serverUrl}/upload";
                Debug.Log($"[GSplatServerClient] Uploading to {url}...");

                using var request = new UnityWebRequest(url, "POST");
                request.uploadHandler = new UploadHandlerRaw(zipData);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/zip");
                request.timeout = 300;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string err = $"Upload failed: {request.error} (HTTP {request.responseCode})";
                    Debug.LogError($"[GSplatServerClient] {err}");
                    Error?.Invoke(err);
                    return false;
                }

                Debug.Log("[GSplatServerClient] Upload successful, training started on server");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[GSplatServerClient] Upload error: {e.Message}");
                Error?.Invoke(e.Message);
                return false;
            }
            finally
            {
                IsUploading = false;
            }
        }

        /// <summary>
        /// Polls the server status once and returns the result.
        /// </summary>
        public async Task<TrainingStatus> PollStatus()
        {
            try
            {
                string url = $"{serverUrl}/status";
                using var request = UnityWebRequest.Get(url);
                request.timeout = 10;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[GSplatServerClient] Status poll failed: {request.error}");
                    return null;
                }

                var status = JsonUtility.FromJson<TrainingStatus>(request.downloadHandler.text);
                LastStatus = status;
                StatusUpdated?.Invoke(status);
                return status;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[GSplatServerClient] Status poll error: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// Continuously polls until training completes or errors.
        /// Returns true if training completed successfully.
        /// </summary>
        public async Task<bool> PollUntilDone()
        {
            if (IsPolling) return false;
            IsPolling = true;

            try
            {
                while (true)
                {
                    var status = await PollStatus();
                    if (status == null)
                    {
                        await Task.Delay((int)(pollIntervalSeconds * 1000));
                        continue;
                    }

                    Debug.Log($"[GSplatServerClient] Training: {status.state} ({status.progress:P0}) - {status.message}");

                    if (status.state == "done") return true;
                    if (status.state == "error")
                    {
                        Error?.Invoke($"Server training error: {status.message}");
                        return false;
                    }
                    if (status.state == "idle") return false;

                    await Task.Delay((int)(pollIntervalSeconds * 1000));
                }
            }
            finally
            {
                IsPolling = false;
            }
        }

        /// <summary>
        /// Downloads the trained PLY file from the server.
        /// Returns the raw PLY bytes, or null on failure.
        /// </summary>
        public async Task<byte[]> DownloadResult()
        {
            if (IsDownloading)
            {
                Debug.LogWarning("[GSplatServerClient] Download already in progress");
                return null;
            }

            IsDownloading = true;
            try
            {
                string url = $"{serverUrl}/download";
                Debug.Log($"[GSplatServerClient] Downloading from {url}...");

                using var request = UnityWebRequest.Get(url);
                request.timeout = 120;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string err = $"Download failed: {request.error} (HTTP {request.responseCode})";
                    Debug.LogError($"[GSplatServerClient] {err}");
                    Error?.Invoke(err);
                    return null;
                }

                byte[] data = request.downloadHandler.data;
                Debug.Log($"[GSplatServerClient] Downloaded {data.Length / (1024 * 1024f):F1}MB PLY");
                return data;
            }
            catch (Exception e)
            {
                Debug.LogError($"[GSplatServerClient] Download error: {e.Message}");
                Error?.Invoke(e.Message);
                return null;
            }
            finally
            {
                IsDownloading = false;
            }
        }

        /// <summary>
        /// Sends a cancel request to the training server.
        /// </summary>
        public async Task CancelTraining()
        {
            try
            {
                string url = $"{serverUrl}/cancel";
                using var request = new UnityWebRequest(url, "POST");
                request.downloadHandler = new DownloadHandlerBuffer();
                request.timeout = 10;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                Debug.Log("[GSplatServerClient] Cancel request sent");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[GSplatServerClient] Cancel error: {e.Message}");
            }
        }

        static byte[] CreateZip(string sourceDir)
        {
            using var ms = new MemoryStream();
            using (var archive = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
            {
                string framesPath = Path.Combine(sourceDir, "frames.jsonl");
                if (File.Exists(framesPath))
                    archive.CreateEntryFromFile(framesPath, "frames.jsonl");

                string plyPath = Path.Combine(sourceDir, "points3d.ply");
                if (File.Exists(plyPath))
                    archive.CreateEntryFromFile(plyPath, "points3d.ply");

                string imagesDir = Path.Combine(sourceDir, "images");
                if (Directory.Exists(imagesDir))
                {
                    foreach (string img in Directory.GetFiles(imagesDir, "*.jpg"))
                    {
                        string entryName = "images/" + Path.GetFileName(img);
                        archive.CreateEntryFromFile(img, entryName);
                    }
                }
            }

            return ms.ToArray();
        }
    }
}
