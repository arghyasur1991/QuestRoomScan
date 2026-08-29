using System;
using System.Threading.Tasks;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace Genesis.RoomScan
{
    /// <summary>
    /// Thin wrapper around Unity's Android runtime-permission dialog so
    /// <see cref="RoomScanSession"/> can expose scene / camera / anchor
    /// grants without each caller inventing callbacks. Always resolves
    /// granted outside Android device builds.
    /// </summary>
    internal static class AndroidRuntimePermission
    {
        public const string Scene = "com.oculus.permission.USE_SCENE";
        public const string Anchors = "com.oculus.permission.USE_ANCHOR_API";

        public static bool Has(string permissionId)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return Permission.HasUserAuthorizedPermission(permissionId);
#else
            return true;
#endif
        }

        public static Task<bool> RequestAsync(string permissionId)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (Permission.HasUserAuthorizedPermission(permissionId))
                return Task.FromResult(true);

            var tcs = new TaskCompletionSource<bool>();
            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += _ => tcs.TrySetResult(true);
            callbacks.PermissionDenied += _ => tcs.TrySetResult(false);
            callbacks.PermissionDeniedAndDontAskAgain += _ => tcs.TrySetResult(false);
            try
            {
                Permission.RequestUserPermission(permissionId, callbacks);
            }
            catch (Exception ex)
            {
                Logger.Error($"Permission request failed ({permissionId}): {ex.Message}");
                tcs.TrySetResult(false);
            }
            return tcs.Task;
#else
            return Task.FromResult(true);
#endif
        }
    }
}
