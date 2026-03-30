using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Debug overlay that renders wireframe bounding boxes + floating labels for
    /// all detected SceneObjects (MRUK + AI). Works on top of any render mode,
    /// toggled via <see cref="RoomScanner.ShowSceneObjects"/>.
    /// </summary>
    public class SceneObjectVisualizer : MonoBehaviour
    {
        private static readonly Color MrukColor = new Color(0f, 1f, 1f, 0.85f);
        private static readonly Color AiColor = new Color(1f, 1f, 0f, 0.85f);
        private const float LineWidth = 0.004f;

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

            Debug.Log($"[SceneObjectVisualizer] Show: {_registry?.Count ?? 0} objects, spawned {_annotations.Count} annotations");
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
            go.transform.SetParent(transform, false);
            go.transform.SetPositionAndRotation(obj.position, obj.rotation);

            var color = obj.source == SceneObjectSource.MRUK ? MrukColor : AiColor;

            CreateWireBox(go, obj.size, color);
            CreateLabel(go, obj, color);

            _annotations.Add(go);
        }

        private void CreateWireBox(GameObject parent, Vector3 size, Color color)
        {
            // Ensure minimum visible size for flat plane anchors
            var vizSize = new Vector3(
                Mathf.Max(size.x, 0.1f),
                Mathf.Max(size.y, 0.1f),
                Mathf.Max(size.z, 0.1f)
            );

            var hx = vizSize.x * 0.5f;
            var hy = vizSize.y * 0.5f;
            var hz = vizSize.z * 0.5f;

            // 12 edges of a box need 3 separate LineRenderers to draw cleanly
            // Use a single LR with a Hamiltonian path through all 8 vertices
            var lr = parent.AddComponent<LineRenderer>();
            lr.useWorldSpace = false;
            lr.loop = false;
            lr.startWidth = LineWidth;
            lr.endWidth = LineWidth;
            lr.material = GetWireMaterial();
            lr.material.color = color;
            lr.startColor = color;
            lr.endColor = color;
            lr.shadowCastingMode = ShadowCastingMode.Off;
            lr.receiveShadows = false;
            lr.allowOcclusionWhenDynamic = false;

            lr.positionCount = 16;
            lr.SetPositions(new Vector3[]
            {
                new(-hx, -hy, -hz), new(hx, -hy, -hz),
                new(hx, -hy, hz), new(-hx, -hy, hz),
                new(-hx, -hy, -hz),
                new(-hx, hy, -hz),
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
            float yOffset = Mathf.Max(obj.size.y, 0.1f) * 0.5f + 0.06f;
            labelGo.transform.localPosition = Vector3.up * yOffset;

            var billboard = labelGo.AddComponent<BillboardLabel>();
            var icon = obj.source == SceneObjectSource.MRUK ? "M" : "AI";
            billboard.Init($"[{icon}] {obj.label}\n{obj.confidence:P0}", color, GetWireMaterial());
        }

        private Material GetWireMaterial()
        {
            if (_wireMaterial != null) return _wireMaterial;

            // Sprites/Default supports vertex colors + transparency out of the box in URP
            var shader = Shader.Find("Sprites/Default");
            if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");

            _wireMaterial = new Material(shader);
            _wireMaterial.color = Color.white;
            _wireMaterial.renderQueue = (int)RenderQueue.Overlay;
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
    /// Billboard label using a world-space Canvas for reliable URP rendering.
    /// </summary>
    public class BillboardLabel : MonoBehaviour
    {
        private Canvas _canvas;
        private UnityEngine.UI.Text _text;

        public void Init(string text, Color color, Material wireMat)
        {
            _canvas = gameObject.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.WorldSpace;
            _canvas.sortingOrder = 30000;

            var rt = _canvas.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(0.4f, 0.12f);
            rt.localScale = Vector3.one * 0.001f;

            var bg = new GameObject("BG");
            bg.transform.SetParent(rt, false);
            var bgRt = bg.AddComponent<RectTransform>();
            bgRt.anchorMin = Vector2.zero;
            bgRt.anchorMax = Vector2.one;
            bgRt.sizeDelta = Vector2.zero;
            var bgImg = bg.AddComponent<UnityEngine.UI.Image>();
            bgImg.color = new Color(0, 0, 0, 0.6f);

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(rt, false);
            var textRt = textGo.AddComponent<RectTransform>();
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.sizeDelta = Vector2.zero;
            textRt.offsetMin = new Vector2(4, 2);
            textRt.offsetMax = new Vector2(-4, -2);

            _text = textGo.AddComponent<UnityEngine.UI.Text>();
            _text.text = text;
            _text.color = color;
            _text.fontSize = 24;
            _text.font = Font.CreateDynamicFontFromOSFont("Arial", 24);
            _text.alignment = TextAnchor.MiddleCenter;
            _text.horizontalOverflow = HorizontalWrapMode.Overflow;
            _text.verticalOverflow = VerticalWrapMode.Overflow;
            _text.raycastTarget = false;
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
