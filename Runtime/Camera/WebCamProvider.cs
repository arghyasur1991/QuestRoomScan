using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Fallback camera provider using WebCamTexture.
    /// Deprecated on Quest (use PassthroughCameraProvider instead),
    /// but useful for editor testing or non-Quest XR platforms.
    /// </summary>
    internal class WebCamProvider : MonoBehaviour, ICameraProvider
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
        public bool IsPlaying => _webcamTex != null && _webcamTex.isPlaying;
        public Texture CurrentFrame => _webcamTex;

        public Pose CameraPose
        {
            get
            {
                if (_headTransform == null)
                {
                    var cam = Camera.main;
                    _headTransform = cam != null ? cam.transform : transform;
                }
                return new Pose(_headTransform.position, _headTransform.rotation);
            }
        }

        public Vector2 FocalLength => new(focalLengthX, focalLengthY);
        public Vector2 PrincipalPoint => new(principalPointX, principalPointY);

        public Vector2 SensorResolution =>
            new(_webcamTex != null ? _webcamTex.width : requestedWidth,
                _webcamTex != null ? _webcamTex.height : requestedHeight);

        public Vector2 CurrentResolution => SensorResolution;

        public void StartCapture()
        {
            if (_webcamTex != null && _webcamTex.isPlaying) return;

            WebCamDevice[] devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                Logger.Warning("No webcam devices found");
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
