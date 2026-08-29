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

        // Unity requires the PermissionCallbacks instance to stay reachable
        // until the user answers. A local that goes out of scope can be
        // collected before the OS dialog appears, which presents as "no
        // dialog, immediately denied".
#if UNITY_ANDROID && !UNITY_EDITOR
        static PermissionCallbacks _heldCallbacks;
#endif

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

            var tcs = new TaskCompletionSource<bool>(
                TaskCreationOptions.RunContinuationsAsynchronously);
            var callbacks = new PermissionCallbacks();
            _heldCallbacks = callbacks;
            void Done(bool granted)
            {
                Logger.Info(
                    $"Permission {(granted ? "granted" : "denied")}: {permissionId}");
                tcs.TrySetResult(granted);
                if (ReferenceEquals(_heldCallbacks, callbacks))
                    _heldCallbacks = null;
            }
            callbacks.PermissionGranted += _ => Done(true);
            callbacks.PermissionDenied += _ => Done(false);
            // Do not subscribe to PermissionDeniedAndDontAskAgain: Unity
            // documents it as unreliable and then skips PermissionDenied.
            try
            {
                Logger.Info($"Requesting permission: {permissionId}");
                Permission.RequestUserPermission(permissionId, callbacks);
            }
            catch (Exception ex)
            {
                Logger.Error($"Permission request failed ({permissionId}): {ex.Message}");
                Done(false);
            }
            return tcs.Task;
#else
            return Task.FromResult(true);
#endif
        }
    }
}
