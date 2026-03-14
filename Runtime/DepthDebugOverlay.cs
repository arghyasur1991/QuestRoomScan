using UnityEngine;
using UnityEngine.UI;

namespace Genesis.RoomScan
{
    public class DepthDebugOverlay : MonoBehaviour
    {
        [SerializeField] private float canvasDistance = 1.5f;
        [SerializeField] private float canvasScale = 0.001f;
        [SerializeField] private Vector2 previewSize = new(320, 240);

        [Header("Depth Visualization")]
        [SerializeField] private Shader depthVisualizeShader;
        [SerializeField] private Color nearColor = Color.red;
        [SerializeField] private Color farColor = Color.blue;
        [SerializeField] private float nearDist = 0.3f;
        [SerializeField] private float farDist = 5f;

        private Canvas _canvas;
        private RawImage _preview;
        private Text _statusText;
        private Transform _headTransform;
        private Material _depthMat;
        private int _frameCount;

        private static readonly int NearColorID = Shader.PropertyToID("_NearColor");
        private static readonly int FarColorID = Shader.PropertyToID("_FarColor");
        private static readonly int NearDistID = Shader.PropertyToID("_NearDist");
        private static readonly int FarDistID = Shader.PropertyToID("_FarDist");

        private void Start()
        {
            _headTransform = Camera.main != null ? Camera.main.transform : null;

            var shader = depthVisualizeShader != null
                ? depthVisualizeShader
                : Shader.Find("Genesis/DepthVisualize");
            if (shader != null)
                _depthMat = new Material(shader);
            else
                Debug.LogWarning("[RoomScan] DepthDebugOverlay: Genesis/DepthVisualize shader not found");

            CreateCanvas();
            Debug.Log($"[RoomScan] DepthDebugOverlay: depthCapture={DepthCapture.Instance != null}, mat={_depthMat != null}");
        }

        private void CreateCanvas()
        {
            var canvasGo = new GameObject("DepthDebugCanvas");
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

            var imgGo = new GameObject("DepthPreview");
            imgGo.transform.SetParent(canvasGo.transform, false);
            _preview = imgGo.AddComponent<RawImage>();
            if (_depthMat != null) _preview.material = _depthMat;
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
                    + _headTransform.up * -0.3f + _headTransform.right * -0.4f;
                _canvas.transform.position = pos;
                _canvas.transform.rotation = Quaternion.LookRotation(
                    pos - _headTransform.position, Vector3.up);
            }

            var dc = DepthCapture.Instance;
            bool available = dc != null && DepthCapture.DepthAvailable;
            Texture depthTex = available ? dc.DepthTex : null;

            if (available && depthTex != null)
            {
                _frameCount++;
                if (_preview != null) _preview.texture = depthTex;
            }

            if (_depthMat != null)
            {
                _depthMat.SetColor(NearColorID, nearColor);
                _depthMat.SetColor(FarColorID, farColor);
                _depthMat.SetFloat(NearDistID, nearDist);
                _depthMat.SetFloat(FarDistID, farDist);
            }

            if (_statusText != null)
            {
                _statusText.color = available ? Color.green : Color.yellow;
                _statusText.text = $"Depth:{available} " +
                    $"Tex:{(depthTex != null ? $"{depthTex.width}x{depthTex.height}" : "NULL")} " +
                    $"Frames:{_frameCount}";
            }
        }

        private void OnDestroy()
        {
            if (_canvas != null) Destroy(_canvas.gameObject);
            if (_depthMat != null) Destroy(_depthMat);
        }
    }
}
