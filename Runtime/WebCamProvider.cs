using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Fallback camera provider using WebCamTexture.
    /// Deprecated on Quest (use PassthroughCameraProvider instead),
    /// but useful for editor testing or non-Quest XR platforms.
    /// </summary>
    public class WebCamProvider : MonoBehaviour, ICameraProvider
    {
        [SerializeField] private int requestedWidth = 1280;
        [SerializeField] private int requestedHeight = 960;
        [SerializeField] private int requestedFPS = 30;

        [Header("Approximate Intrinsics")]
        [SerializeField] private float focalLengthX = 600f;
        [SerializeField] private float focalLengthY = 600f;
        [SerializeField] private float principalPointX = 640f;
        [SerializeField] private float principalPointY = 480f;

        private WebCamTexture _webcamTex;
        private Transform _headTransform;

        public bool IsReady => _webcamTex != null && _webcamTex.isPlaying && _webcamTex.didUpdateThisFrame;
        public Texture CurrentFrame => _webcamTex;

        public Matrix4x4 CameraToWorld
        {
            get
            {
                if (_headTransform == null)
                {
                    var cam = Camera.main;
                    _headTransform = cam != null ? cam.transform : transform;
                }
                return _headTransform.localToWorldMatrix;
            }
        }

        public Matrix4x4 ProjectionMatrix
        {
            get
            {
                float w = _webcamTex != null ? _webcamTex.width : requestedWidth;
                float h = _webcamTex != null ? _webcamTex.height : requestedHeight;

                const float near = 0.1f;
                const float far = 100f;

                Matrix4x4 proj = Matrix4x4.zero;
                proj.m00 = 2f * focalLengthX / w;
                proj.m11 = 2f * focalLengthY / h;
                proj.m02 = 1f - 2f * principalPointX / w;
                proj.m12 = 2f * principalPointY / h - 1f;
                proj.m22 = -(far + near) / (far - near);
                proj.m23 = -2f * far * near / (far - near);
                proj.m32 = -1f;
                return proj;
            }
        }

        public void StartCapture()
        {
            if (_webcamTex != null && _webcamTex.isPlaying) return;

            WebCamDevice[] devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                Debug.LogWarning("[RoomScan] No webcam devices found");
                return;
            }

            _webcamTex = new WebCamTexture(devices[0].name, requestedWidth, requestedHeight, requestedFPS);
            _webcamTex.Play();
        }

        public void StopCapture()
        {
            if (_webcamTex != null && _webcamTex.isPlaying)
                _webcamTex.Stop();
        }

        private void OnDestroy()
        {
            StopCapture();
            if (_webcamTex != null)
                Destroy(_webcamTex);
        }
    }
}
