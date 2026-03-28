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
    /// </summary>
    public class PassthroughCameraProvider : MonoBehaviour, ICameraProvider
    {
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

        /// <inheritdoc />
        public void StartCapture()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            const string headsetCameraPerm = "horizonos.permission.HEADSET_CAMERA";
            if (!Permission.HasUserAuthorizedPermission(headsetCameraPerm))
            {
                Logger.Info("Requesting HEADSET_CAMERA permission");
                Permission.RequestUserPermission(headsetCameraPerm);
            }
#endif
            if (_pca == null)
            {
                _pca = gameObject.GetComponent<PassthroughCameraAccess>();
                if (_pca == null)
                    _pca = gameObject.AddComponent<PassthroughCameraAccess>();
            }

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
