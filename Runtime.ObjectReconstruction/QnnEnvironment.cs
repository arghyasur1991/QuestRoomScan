#if HAS_ONNXRUNTIME
using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Sets up ADSP_LIBRARY_PATH on Android before QNN HTP session creation.
    /// Idempotent -- safe to call multiple times.
    /// </summary>
    internal static class QnnEnvironment
    {
        private static bool _initialized;
        private static string _nativeLibDir;

        internal static string NativeLibDir => _nativeLibDir;

        [DllImport("c")]
        private static extern int setenv(string name, string value, int overwrite);

        internal static void Initialize()
        {
            if (_initialized) return;
            _initialized = true;

#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                _nativeLibDir = GetAndroidNativeLibDir();
                setenv("ADSP_LIBRARY_PATH", _nativeLibDir, 1);
                Logger.Info($"[QnnEnvironment] ADSP_LIBRARY_PATH={_nativeLibDir}");
            }
            catch (Exception e)
            {
                Logger.Error($"[QnnEnvironment] Failed to set ADSP_LIBRARY_PATH: {e.Message}");
            }
#else
            Logger.Info("[QnnEnvironment] QNN HTP only available on Android/Quest. Skipping env setup.");
#endif
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private static string GetAndroidNativeLibDir()
        {
            using var unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            using var activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
            using var appInfo = activity.Call<AndroidJavaObject>("getApplicationInfo");
            return appInfo.Get<string>("nativeLibraryDir");
        }
#endif
    }
}
#endif
