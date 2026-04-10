#if HAS_ONNXRUNTIME
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Resolves .onnx model paths from StreamingAssets. On Android, copies files
    /// from the APK to persistentDataPath on first access since StreamingAssets
    /// are inside the compressed archive and ORT's InferenceSession needs a real
    /// filesystem path. Re-copies when the app version changes to pick up updated models.
    /// </summary>
    internal static class ModelPathResolver
    {
        private const string VersionPrefKey = "ModelPathResolver_AppVersion";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void InvalidateCacheOnUpgrade()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            bool shouldClear = false;

            if (Debug.isDebugBuild)
            {
                // Dev builds: always clear to guarantee fresh models after every deploy
                shouldClear = true;
            }
            else
            {
                string currentVersion = Application.version + "|" + Application.buildGUID;
                string cachedVersion = PlayerPrefs.GetString(VersionPrefKey, "");
                shouldClear = (cachedVersion != currentVersion);
            }

            if (shouldClear)
            {
                ClearAllCaches();
                string currentVersion = Application.version + "|" + Application.buildGUID;
                PlayerPrefs.SetString(VersionPrefKey, currentVersion);
                PlayerPrefs.Save();
            }
#endif
        }

        private static void ClearAllCaches()
        {
            string cacheDir = Path.Combine(Application.persistentDataPath, "ObjectReconstruction");
            if (Directory.Exists(cacheDir))
            {
                Directory.Delete(cacheDir, true);
                Logger.Info("[ModelPathResolver] Cleared model cache");
            }

            string ortOptDir = Path.Combine(Application.persistentDataPath, "ort_opt_cache");
            if (Directory.Exists(ortOptDir))
            {
                Directory.Delete(ortOptDir, true);
                Logger.Info("[ModelPathResolver] Cleared ORT optimization cache");
            }
        }

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
                try
                {
                    var op = request.SendWebRequest();

                    while (!op.isDone)
                    {
                        ct.ThrowIfCancellationRequested();
                        await Task.Yield();
                    }

                    if (request.result != UnityWebRequest.Result.Success)
                        throw new FileNotFoundException(
                            $"Failed to load model from StreamingAssets: {request.error}", relativePath);

                    var bytes = request.downloadHandler.data;
                    await Task.Run(() => File.WriteAllBytes(persistentPath, bytes));
                }
                finally
                {
                    request.Dispose();
                }
            }

            return persistentPath;
#else
            if (!File.Exists(streamingPath))
                throw new FileNotFoundException(
                    $"Model not found at {streamingPath}. Run 'Deploy Models' in the Setup Wizard.", relativePath);

            await Task.CompletedTask;
            return streamingPath;
#endif
        }
    }
}
#endif
