using UnityEngine;
using UnityEngine.UI;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Debug overlay that shows the passthrough camera feed and status on a world-space canvas.
    /// Attach to the RoomScanner GameObject or any GO with a PassthroughCameraProvider reference.
    /// </summary>
    public class CameraDebugOverlay : MonoBehaviour
    {
        [SerializeField] private float canvasDistance = 1.5f;
        [SerializeField] private float canvasScale = 0.001f;
        [SerializeField] private Vector2 previewSize = new(320, 240);

        private Canvas _canvas;
        private RawImage _preview;
        private Text _statusText;
        private ICameraProvider _provider;
        private Transform _headTransform;
        private int _frameCount;

        private void Start()
        {
            _headTransform = Camera.main != null ? Camera.main.transform : null;

            var scanner = RoomScanner.Instance;
            if (scanner != null)
            {
                var field = typeof(RoomScanner).GetField("cameraProvider",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (field != null)
                    _provider = field.GetValue(scanner) as ICameraProvider;
            }

            if (_provider == null)
                _provider = FindFirstObjectByType<PassthroughCameraProvider>();

            CreateCanvas();
            Debug.Log($"[RoomScan] CameraDebugOverlay: provider={(_provider != null ? _provider.GetType().Name : "NULL")}");
        }

        private void CreateCanvas()
        {
            var canvasGo = new GameObject("CameraDebugCanvas");
            canvasGo.transform.SetParent(transform);

            _canvas = canvasGo.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.WorldSpace;
            var rt = _canvas.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(previewSize.x, previewSize.y + 60);
            canvasGo.transform.localScale = Vector3.one * canvasScale;

            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            var bgGo = new GameObject("Background");
            bgGo.transform.SetParent(canvasGo.transform, false);
            var bgImg = bgGo.AddComponent<Image>();
            bgImg.color = new Color(0, 0, 0, 0.7f);
            var bgRt = bgImg.GetComponent<RectTransform>();
            bgRt.anchorMin = Vector2.zero;
            bgRt.anchorMax = Vector2.one;
            bgRt.offsetMin = Vector2.zero;
            bgRt.offsetMax = Vector2.zero;

            var imgGo = new GameObject("CameraPreview");
            imgGo.transform.SetParent(canvasGo.transform, false);
            _preview = imgGo.AddComponent<RawImage>();
            var imgRt = _preview.GetComponent<RectTransform>();
            imgRt.anchorMin = new Vector2(0, 0.2f);
            imgRt.anchorMax = Vector2.one;
            imgRt.offsetMin = new Vector2(4, 4);
            imgRt.offsetMax = new Vector2(-4, -4);
            _preview.color = Color.white;

            var textGo = new GameObject("StatusText");
            textGo.transform.SetParent(canvasGo.transform, false);
            _statusText = textGo.AddComponent<Text>();
            _statusText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _statusText.fontSize = 14;
            _statusText.color = Color.green;
            _statusText.alignment = TextAnchor.MiddleLeft;
            var textRt = _statusText.GetComponent<RectTransform>();
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = new Vector2(1, 0.2f);
            textRt.offsetMin = new Vector2(6, 2);
            textRt.offsetMax = new Vector2(-6, -2);
        }

        private void LateUpdate()
        {
            if (_headTransform != null && _canvas != null)
            {
                Vector3 pos = _headTransform.position + _headTransform.forward * canvasDistance
                    + _headTransform.up * -0.3f;
                _canvas.transform.position = pos;
                _canvas.transform.rotation = Quaternion.LookRotation(
                    pos - _headTransform.position, Vector3.up);
            }

            if (_provider == null)
            {
                if (_statusText != null)
                {
                    _statusText.text = "No camera provider";
                    _statusText.color = Color.red;
                }
                return;
            }

            bool ready = _provider.IsReady;
            Texture tex = _provider.CurrentFrame;

            if (ready && tex != null)
            {
                _frameCount++;
                if (_preview != null) _preview.texture = tex;
            }

            if (_statusText != null)
            {
                _statusText.color = ready ? Color.green : Color.yellow;
                _statusText.text = $"Ready:{ready} Tex:{(tex != null ? $"{tex.width}x{tex.height}" : "NULL")} Frames:{_frameCount}";
            }
        }

        private void OnDestroy()
        {
            if (_canvas != null) Destroy(_canvas.gameObject);
        }
    }
}
