using System;
using System.Threading.Tasks;
using Meta.XR;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace Genesis.RoomScan
{
    /// <summary>
    /// Camera provider backed by Meta's PassthroughCameraAccess (Quest 3+).
    /// Provides intrinsics, extrinsics, and RGB frames from the headset cameras.
    ///
    /// <para>
    /// PCA discovery is <b>scene-wide</b> (not GameObject-local): if the Meta XR
    /// Building Block already dropped a <see cref="PassthroughCameraAccess"/>
    /// onto an OVRCameraRig, the provider re-uses it. Without scene-wide find
    /// we ended up with two PCA components fighting over the single native
    /// camera handle — first session worked, subsequent sessions stuck because
    /// PCA self-disables when <c>Play()</c> fails.
    /// </para>
    /// </summary>
    public class PassthroughCameraProvider : MonoBehaviour, ICameraProvider
    {
        /// <summary>The Horizon OS permission required by PCA on Quest 3+.</summary>
        public const string CameraPermissionId = "horizonos.permission.HEADSET_CAMERA";

        [SerializeField] private PassthroughCameraAccess.CameraPositionType cameraPosition =
            PassthroughCameraAccess.CameraPositionType.Left;
        [SerializeField] private Vector2Int requestedResolution = new(1280, 960);
        [SerializeField] private int maxFramerate = 30;

        private PassthroughCameraAccess _pca;

        /// <inheritdoc />
        public bool IsReady => _pca != null && _pca.IsPlaying && _pca.IsUpdatedThisFrame;

        /// <inheritdoc />
        public bool IsPlaying => _pca != null && _pca.IsPlaying;

        /// <inheritdoc />
        public Texture CurrentFrame => _pca != null && _pca.IsPlaying ? _pca.GetTexture() : null;

        /// <inheritdoc />
        public Pose CameraPose =>
            _pca != null && _pca.IsPlaying ? _pca.GetCameraPose() : Pose.identity;

        /// <inheritdoc />
        public Vector2 FocalLength =>
            _pca != null && _pca.IsPlaying ? _pca.Intrinsics.FocalLength : Vector2.one;

        /// <inheritdoc />
        public Vector2 PrincipalPoint =>
            _pca != null && _pca.IsPlaying ? _pca.Intrinsics.PrincipalPoint : Vector2.zero;

        /// <inheritdoc />
        public Vector2 SensorResolution =>
            _pca != null && _pca.IsPlaying ? _pca.Intrinsics.SensorResolution : new Vector2(1280, 960);

        /// <inheritdoc />
        public Vector2 CurrentResolution =>
            _pca != null && _pca.IsPlaying
                ? new Vector2(_pca.CurrentResolution.x, _pca.CurrentResolution.y)
                : new Vector2(1280, 960);

        /// <summary>
        /// True when the user has granted the Horizon OS HEADSET_CAMERA
        /// permission. Always true outside Android device builds.
        /// </summary>
        public static bool HasCameraPermission
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            get => Permission.HasUserAuthorizedPermission(CameraPermissionId);
#else
            get => true;
#endif
        }

        /// <summary>
        /// Requests the HEADSET_CAMERA permission and resolves once the user
        /// accepts, denies, or dismisses the system dialog. Always resolves
        /// <c>true</c> outside Android device builds (no permission to request).
        /// Resolves <c>true</c> immediately if already granted.
        /// </summary>
        public static Task<bool> RequestCameraPermissionAsync()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (Permission.HasUserAuthorizedPermission(CameraPermissionId))
                return Task.FromResult(true);

            var tcs = new TaskCompletionSource<bool>();
            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += _ => tcs.TrySetResult(true);
            callbacks.PermissionDenied  += _ => tcs.TrySetResult(false);
            callbacks.PermissionDeniedAndDontAskAgain += _ => tcs.TrySetResult(false);
            try
            {
                Permission.RequestUserPermission(CameraPermissionId, callbacks);
            }
            catch (Exception ex)
            {
                Logger.Error($"PassthroughCameraProvider: permission request failed: {ex.Message}");
                tcs.TrySetResult(false);
            }
            return tcs.Task;
#else
            return Task.FromResult(true);
#endif
        }

        /// <inheritdoc />
        public void StartCapture()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!Permission.HasUserAuthorizedPermission(CameraPermissionId))
            {
                // Defensive: if the caller hasn't already gone through
                // RequestCameraPermissionAsync, kick the dialog now. PCA's
                // own OnEnable also coroutine-polls until granted, so this
                // is layered safety, not a single source of truth.
                Logger.Info("Requesting HEADSET_CAMERA permission");
                Permission.RequestUserPermission(CameraPermissionId);
            }
#endif

            // Scene-wide find: re-use the PCA from the Meta XR Building Block
            // (typically attached to OVRCameraRig). Falling back to a
            // GameObject-local AddComponent here was the bug that caused two
            // PCAs to race for the camera handle — PCA's native side allows
            // exactly one instance per camera position, so the second one
            // self-disables and from then on neither one plays.
            if (_pca == null)
            {
                _pca = FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include);
                if (_pca == null)
                {
                    Logger.Warning("PassthroughCameraProvider: no PassthroughCameraAccess in scene — adding one to " +
                                   $"'{gameObject.name}'. Prefer letting Meta's Building Block place it on the OVRCameraRig.");
                    _pca = gameObject.AddComponent<PassthroughCameraAccess>();
                }
                else
                {
                    Logger.Info($"PassthroughCameraProvider: adopted existing PassthroughCameraAccess on '{_pca.gameObject.name}'.");
                }
            }

            // PCA forbids MaxFramerate changes while running. Drive it disabled
            // for the property writes, then re-enable so OnEnable runs cleanly.
            // No-op if it was already disabled.
            bool wasEnabled = _pca.enabled;
            if (wasEnabled) _pca.enabled = false;
            _pca.CameraPosition = cameraPosition;
            _pca.RequestedResolution = requestedResolution;
            _pca.MaxFramerate = maxFramerate;
            _pca.enabled = true;
        }

        /// <inheritdoc />
        public void StopCapture()
        {
            if (_pca != null)
                _pca.enabled = false;
        }

        private void OnDestroy()
        {
            StopCapture();
        }
    }
}
