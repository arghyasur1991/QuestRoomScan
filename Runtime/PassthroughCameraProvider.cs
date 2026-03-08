using Meta.XR;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Camera provider using Meta's PassthroughCameraAccess (MRUK).
    /// Provides proper intrinsics, extrinsics, and timestamps from the Quest 3 cameras.
    /// Replaces the deprecated WebCamTexture approach.
    /// </summary>
    public class PassthroughCameraProvider : MonoBehaviour, ICameraProvider
    {
        [SerializeField] private PassthroughCameraAccess.CameraPositionType cameraPosition =
            PassthroughCameraAccess.CameraPositionType.Left;
        [SerializeField] private Vector2Int requestedResolution = new(1280, 960);
        [SerializeField] private int maxFramerate = 30;

        private PassthroughCameraAccess _pca;

        public bool IsReady => _pca != null && _pca.IsPlaying && _pca.IsUpdatedThisFrame;

        public Texture CurrentFrame => _pca != null && _pca.IsPlaying ? _pca.GetTexture() : null;

        public Matrix4x4 CameraToWorld
        {
            get
            {
                if (_pca == null || !_pca.IsPlaying) return Matrix4x4.identity;
                Pose pose = _pca.GetCameraPose();
                return Matrix4x4.TRS(pose.position, pose.rotation, Vector3.one);
            }
        }

        public Matrix4x4 ProjectionMatrix
        {
            get
            {
                if (_pca == null || !_pca.IsPlaying) return Matrix4x4.identity;

                var intrinsics = _pca.Intrinsics;
                float fx = intrinsics.FocalLength.x;
                float fy = intrinsics.FocalLength.y;
                float cx = intrinsics.PrincipalPoint.x;
                float cy = intrinsics.PrincipalPoint.y;
                float w = _pca.CurrentResolution.x;
                float h = _pca.CurrentResolution.y;

                const float near = 0.1f;
                const float far = 100f;

                // Pinhole intrinsics → OpenGL projection matrix
                Matrix4x4 proj = Matrix4x4.zero;
                proj.m00 = 2f * fx / w;
                proj.m11 = 2f * fy / h;
                proj.m02 = 1f - 2f * cx / w;
                proj.m12 = 2f * cy / h - 1f;
                proj.m22 = -(far + near) / (far - near);
                proj.m23 = -2f * far * near / (far - near);
                proj.m32 = -1f;
                return proj;
            }
        }

        public void StartCapture()
        {
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
