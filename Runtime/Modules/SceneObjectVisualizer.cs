using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Debug overlay that renders wireframe bounding boxes + floating labels for
    /// all detected SceneObjects (MRUK + AI). Works on top of any render mode,
    /// toggled via <see cref="RoomScanner.ShowSceneObjects"/>.
    /// </summary>
    public class SceneObjectVisualizer : MonoBehaviour
    {
        private static readonly Color MrukColor = new Color(0f, 1f, 1f, 0.7f);
        private static readonly Color AiColor = new Color(1f, 1f, 0f, 0.7f);
        private const float LabelHeight = 0.08f;
        private const float LineWidth = 0.003f;

        private SceneObjectRegistry _registry;
        private readonly List<GameObject> _annotations = new();
        private Material _wireMaterial;
        private bool _visible;

        public void Show(SceneObjectRegistry registry)
        {
            _registry = registry;
            _visible = true;
            Rebuild();

            if (_registry != null)
                _registry.ObjectAdded += OnObjectAdded;
        }

        public void Hide()
        {
            if (_registry != null)
                _registry.ObjectAdded -= OnObjectAdded;

            _visible = false;
            foreach (var a in _annotations)
            {
                if (a != null) Destroy(a);
            }
            _annotations.Clear();
        }

        private void OnObjectAdded(SceneObject obj)
        {
            if (!_visible) return;
            SpawnAnnotation(obj);
        }

        private void Rebuild()
        {
            var wasVisible = _visible;
            Hide();
            _visible = wasVisible;

            if (_registry == null) return;
            if (_registry != null)
                _registry.ObjectAdded += OnObjectAdded;

            foreach (var obj in _registry.AllObjects)
                SpawnAnnotation(obj);
        }

        private void SpawnAnnotation(SceneObject obj)
        {
            var go = new GameObject($"SceneObj_{obj.id}");
            go.transform.SetParent(transform);
            go.transform.SetPositionAndRotation(obj.position, obj.rotation);

            var color = obj.source == SceneObjectSource.MRUK ? MrukColor : AiColor;
            color.a = Mathf.Clamp(obj.confidence, 0.3f, 0.9f);

            CreateWireBox(go, obj.size, color);
            CreateLabel(go, obj, color);

            _annotations.Add(go);
        }

        private void CreateWireBox(GameObject parent, Vector3 size, Color color)
        {
            var lr = parent.AddComponent<LineRenderer>();
            lr.useWorldSpace = false;
            lr.loop = true;
            lr.startWidth = LineWidth;
            lr.endWidth = LineWidth;
            lr.startColor = color;
            lr.endColor = color;
            lr.material = GetWireMaterial();
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;

            var hx = size.x * 0.5f;
            var hy = size.y * 0.5f;
            var hz = size.z * 0.5f;

            lr.positionCount = 16;
            lr.SetPositions(new Vector3[]
            {
                // Bottom face
                new(-hx, -hy, -hz), new(hx, -hy, -hz),
                new(hx, -hy, hz), new(-hx, -hy, hz),
                new(-hx, -hy, -hz),
                // Up to top
                new(-hx, hy, -hz),
                // Top face
                new(hx, hy, -hz), new(hx, -hy, -hz),
                new(hx, hy, -hz), new(hx, hy, hz),
                new(hx, -hy, hz), new(hx, hy, hz),
                new(-hx, hy, hz), new(-hx, -hy, hz),
                new(-hx, hy, hz), new(-hx, hy, -hz)
            });
        }

        private void CreateLabel(GameObject parent, SceneObject obj, Color color)
        {
            var labelGo = new GameObject("Label");
            labelGo.transform.SetParent(parent.transform, false);
            labelGo.transform.localPosition = Vector3.up * (obj.size.y * 0.5f + 0.05f);

            var billboard = labelGo.AddComponent<BillboardLabel>();
            var icon = obj.source == SceneObjectSource.MRUK ? "M" : "AI";
            billboard.Text = $"[{icon}] {obj.label}\n{obj.confidence:P0}";
            billboard.TextColor = color;
        }

        private Material GetWireMaterial()
        {
            if (_wireMaterial != null) return _wireMaterial;
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            _wireMaterial = new Material(shader);
            _wireMaterial.SetFloat("_Surface", 1);
            _wireMaterial.SetFloat("_Blend", 0);
            return _wireMaterial;
        }

        private void OnDestroy()
        {
            Hide();
            if (_wireMaterial != null)
                Destroy(_wireMaterial);
        }
    }

    /// <summary>
    /// Simple billboard that faces the camera. Uses TextMesh for broad compatibility
    /// (no TMP dependency required).
    /// </summary>
    public class BillboardLabel : MonoBehaviour
    {
        public string Text { get; set; }
        public Color TextColor { get; set; } = Color.white;

        private TextMesh _textMesh;

        private void Start()
        {
            _textMesh = gameObject.AddComponent<TextMesh>();
            _textMesh.text = Text;
            _textMesh.color = TextColor;
            _textMesh.characterSize = 0.02f;
            _textMesh.fontSize = 48;
            _textMesh.anchor = TextAnchor.LowerCenter;
            _textMesh.alignment = TextAlignment.Center;
        }

        private void LateUpdate()
        {
            var cam = Camera.main;
            if (cam == null) return;
            transform.rotation = Quaternion.LookRotation(
                transform.position - cam.transform.position);
        }
    }
}
