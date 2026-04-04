#if HAS_AI_INFERENCE
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Resolves .sentis model paths from StreamingAssets. On Android, copies files
    /// from the APK to persistentDataPath on first access since StreamingAssets
    /// are inside the compressed archive and can't be opened with File.OpenRead.
    /// </summary>
    internal static class ModelPathResolver
    {
        internal static async Task<string> ResolveAsync(string relativePath, CancellationToken ct)
        {
            string streamingPath = Path.Combine(Application.streamingAssetsPath, relativePath);

#if UNITY_ANDROID && !UNITY_EDITOR
            string persistentPath = Path.Combine(Application.persistentDataPath, relativePath);

            if (!File.Exists(persistentPath))
            {
                string dir = Path.GetDirectoryName(persistentPath);
                if (dir != null) Directory.CreateDirectory(dir);

                var request = UnityWebRequest.Get(streamingPath);
                var op = request.SendWebRequest();

                while (!op.isDone)
                {
                    ct.ThrowIfCancellationRequested();
                    await Task.Yield();
                }

                if (request.result != UnityWebRequest.Result.Success)
                    throw new FileNotFoundException(
                        $"Failed to load model from StreamingAssets: {request.error}", relativePath);

                File.WriteAllBytes(persistentPath, request.downloadHandler.data);
                request.Dispose();

                Logger.Info($"[ModelPathResolver] Copied {relativePath} to persistentDataPath");
            }

            return persistentPath;
#else
            if (!File.Exists(streamingPath))
                throw new FileNotFoundException(
                    $"Model not found at {streamingPath}. Run 'Convert Models' in the Setup Wizard.", relativePath);

            await Task.CompletedTask;
            return streamingPath;
#endif
        }
    }
}
#endif
