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
        [SerializeField, Tooltip("IP:port of your PC running the GS training server (gs-server)")]
        string serverUrl = "http://192.168.1.100:8420";
        [SerializeField, Tooltip("Number of training iterations on the server. Lower = faster but fewer/coarser splats.")]
        int trainingIterations = 7000;
        [SerializeField, Range(1f, 30f)] float pollIntervalSeconds = 3f;

        public string ServerUrl
        {
            get => serverUrl;
            set => serverUrl = value.TrimEnd('/');
        }

        public int TrainingIterations
        {
            get => trainingIterations;
            set => trainingIterations = Mathf.Max(value, 100);
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
            public string backend;
            public int current_iteration;
            public int total_iterations;
            public float elapsed_seconds;
        }

        /// <summary>
        /// Packages the GSExport directory into a ZIP and uploads to the training server.
        /// If keyframeRelocation is non-identity, camera poses in frames.jsonl are
        /// transformed to match the relocated mesh coordinate frame.
        /// </summary>
        public async Task<bool> UploadTrainingData(Matrix4x4 keyframeRelocation = default)
        {
            if (keyframeRelocation == default) keyframeRelocation = Matrix4x4.identity;

            if (IsUploading)
            {
                Logger.Warning("Upload already in progress");
                return false;
            }

            IsUploading = true;
            try
            {
                string exportDir = Path.Combine(Application.persistentDataPath, "GSExport");
                if (!Directory.Exists(exportDir))
                {
                    Logger.Error("GSExport directory not found");
                    Error?.Invoke("GSExport directory not found");
                    return false;
                }

                string framesFile = Path.Combine(exportDir, "frames.jsonl");
                if (!File.Exists(framesFile))
                {
                    Logger.Error("frames.jsonl not found");
                    Error?.Invoke("frames.jsonl not found — no keyframes collected");
                    return false;
                }

                var reloc = keyframeRelocation;
                Logger.Info($"Creating ZIP from GSExport...{(reloc != Matrix4x4.identity ? " (relocating poses)" : "")}");
                byte[] zipData = await Task.Run(() => CreateZip(exportDir, reloc));
                Logger.Info($"ZIP created: {zipData.Length / (1024 * 1024)}MB");

                string url = $"{serverUrl}/upload?iterations={trainingIterations}";
                Logger.Info($"Uploading to {url} ({trainingIterations} iters)...");

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
                    Logger.Error(err);
                    Error?.Invoke(err);
                    return false;
                }

                Logger.Info("Upload successful, training started on server");
                return true;
            }
            catch (Exception e)
            {
                Logger.Error($"Upload error: {e.Message}");
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
                string url = $"{serverUrl}/api/status";
                using var request = UnityWebRequest.Get(url);
                request.timeout = 10;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Logger.Warning($"Status poll failed: {request.error}");
                    return null;
                }

                var status = JsonUtility.FromJson<TrainingStatus>(request.downloadHandler.text);
                LastStatus = status;
                StatusUpdated?.Invoke(status);
                return status;
            }
            catch (Exception e)
            {
                Logger.Warning($"Status poll error: {e.Message}");
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

                    Logger.Info($"Training: {status.state} ({status.progress:P0}) - {status.message}");

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
                Logger.Warning("Download already in progress");
                return null;
            }

            IsDownloading = true;
            try
            {
                string url = $"{serverUrl}/download";
                Logger.Info($"Downloading from {url}...");

                using var request = UnityWebRequest.Get(url);
                request.timeout = 120;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string err = $"Download failed: {request.error} (HTTP {request.responseCode})";
                    Logger.Error(err);
                    Error?.Invoke(err);
                    return null;
                }

                byte[] data = request.downloadHandler.data;
                Logger.Info($"Downloaded {data.Length / (1024 * 1024f):F1}MB PLY");
                return data;
            }
            catch (Exception e)
            {
                Logger.Error($"Download error: {e.Message}");
                Error?.Invoke(e.Message);
                return null;
            }
            finally
            {
                IsDownloading = false;
            }
        }

        /// <summary>
        /// Uploads an atlas PNG for server-side enhancement (super-resolution + optional inpainting).
        /// Returns the enhanced atlas as raw PNG bytes, or null on failure.
        /// </summary>
        public async Task<byte[]> EnhanceAtlasAsync(byte[] atlasPng, int scale = 2, bool inpaint = true)
        {
            try
            {
                string url = $"{serverUrl}/enhance-atlas?scale={scale}&inpaint={inpaint.ToString().ToLower()}";
                Logger.Info($"Uploading atlas for enhancement ({atlasPng.Length / 1024}KB, x{scale}, inpaint={inpaint})...");

                using var request = new UnityWebRequest(url, "POST");
                request.uploadHandler = new UploadHandlerRaw(atlasPng);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "image/png");
                request.timeout = 600;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string err = $"Atlas enhance failed: {request.error} (HTTP {request.responseCode})";
                    Logger.Error(err);
                    Error?.Invoke(err);
                    return null;
                }

                byte[] result = request.downloadHandler.data;
                Logger.Info($"Enhanced atlas received: {result.Length / 1024}KB");
                return result;
            }
            catch (Exception e)
            {
                Logger.Error($"Atlas enhance error: {e.Message}");
                Error?.Invoke(e.Message);
                return null;
            }
        }

        /// <summary>
        /// Uploads a refined_mesh.bin to the server for geometry enhancement
        /// (bilateral smoothing + plane snapping). Returns the enhanced mesh bytes, or null on failure.
        /// </summary>
        public async Task<byte[]> EnhanceMeshAsync(byte[] meshBin,
            int smoothIterations = 5, string smoothMethod = "bilateral",
            bool enablePlaneSnap = true)
        {
            try
            {
                string url = $"{serverUrl}/enhance-mesh?smooth_iterations={smoothIterations}" +
                             $"&smooth_method={smoothMethod}&enable_plane_snap={enablePlaneSnap.ToString().ToLower()}";
                Logger.Info($"Uploading mesh for enhancement ({meshBin.Length / 1024}KB)...");

                using var request = new UnityWebRequest(url, "POST");
                request.uploadHandler = new UploadHandlerRaw(meshBin);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/octet-stream");
                request.timeout = 120;

                var op = request.SendWebRequest();
                while (!op.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    string err = $"Mesh enhance failed: {request.error} (HTTP {request.responseCode})";
                    Logger.Error(err);
                    Error?.Invoke(err);
                    return null;
                }

                byte[] result = request.downloadHandler.data;
                Logger.Info($"Enhanced mesh received: {result.Length / 1024}KB");
                return result;
            }
            catch (Exception e)
            {
                Logger.Error($"Mesh enhance error: {e.Message}");
                Error?.Invoke(e.Message);
                return null;
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

                Logger.Info("Cancel request sent");
            }
            catch (Exception e)
            {
                Logger.Warning($"Cancel error: {e.Message}");
            }
        }

        static byte[] CreateZip(string sourceDir, Matrix4x4 relocation)
        {
            using var ms = new MemoryStream();
            using (var archive = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
            {
                byte[] framesData = TextureRefinement.RelocateFramesJsonl(sourceDir, relocation);
                if (framesData != null)
                {
                    var entry = archive.CreateEntry("frames.jsonl");
                    using var es = entry.Open();
                    es.Write(framesData, 0, framesData.Length);
                }

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
