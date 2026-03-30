using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Contextual 3D prop placement using SceneObjectRegistry bounds.
    /// Props fade in as transformation progress passes their threshold.
    /// Placement rules determine where props appear relative to detected objects.
    /// </summary>
    public class RoomPropSpawner : MonoBehaviour
    {
        private ThemePack _theme;
        private float _progress;
        private readonly List<PropInstance> _props = new();

        private struct PropInstance
        {
            public GameObject go;
            public float threshold;
            public Vector3 targetScale;
        }

        public void Initialize(SceneObjectRegistry registry, ThemePack theme)
        {
            _theme = theme;
            if (theme?.propDefinitions == null || registry == null) return;

            foreach (var def in theme.propDefinitions)
            {
                if (def.prefab == null) continue;

                var positions = ResolvePositions(def, registry);
                int count = Mathf.Min(positions.Count, def.maxInstances);

                for (int i = 0; i < count; i++)
                {
                    var go = Instantiate(def.prefab, transform);
                    go.transform.position = positions[i];
                    var targetScale = go.transform.localScale;
                    go.transform.localScale = Vector3.zero;
                    go.SetActive(false);

                    _props.Add(new PropInstance
                    {
                        go = go,
                        threshold = def.progressThreshold,
                        targetScale = targetScale
                    });
                }
            }
        }

        public void SetProgress(float progress)
        {
            _progress = progress;
            if (_theme == null) return;

            float layerIntensity = _theme.propCurve.Evaluate(progress);

            for (int i = 0; i < _props.Count; i++)
            {
                var prop = _props[i];
                if (prop.go == null) continue;

                bool visible = progress >= prop.threshold && layerIntensity > 0.01f;
                prop.go.SetActive(visible);

                if (visible)
                {
                    float fadeIn = Mathf.Clamp01((progress - prop.threshold) / 0.1f) * layerIntensity;
                    prop.go.transform.localScale = prop.targetScale * fadeIn;
                }
            }
        }

        private static List<Vector3> ResolvePositions(PropDefinition def, SceneObjectRegistry registry)
        {
            var positions = new List<Vector3>();

            switch (def.placementRule)
            {
                case PropPlacementRule.NearObject:
                    var objects = registry.FindByLabel(def.targetLabel ?? "");
                    foreach (var obj in objects)
                    {
                        var offset = new Vector3(
                            Random.Range(-0.3f, 0.3f), 0,
                            Random.Range(-0.3f, 0.3f));
                        positions.Add(obj.position + offset);
                    }
                    break;

                case PropPlacementRule.OnSurface:
                    var surfaces = registry.FindBySurface(SurfaceType.Furniture);
                    foreach (var s in surfaces)
                        positions.Add(s.position + Vector3.up * s.size.y * 0.5f);
                    break;

                case PropPlacementRule.AtCorner:
                    var walls = registry.FindBySurface(SurfaceType.Wall);
                    for (int i = 0; i < walls.Count && i < def.maxInstances; i++)
                    {
                        var w = walls[i];
                        positions.Add(w.position + w.rotation * Vector3.right * w.size.x * 0.5f);
                    }
                    break;

                case PropPlacementRule.OnCeiling:
                    var ceilings = registry.FindBySurface(SurfaceType.Ceiling);
                    foreach (var c in ceilings)
                        positions.Add(c.position);
                    break;

                case PropPlacementRule.Random:
                    for (int i = 0; i < def.maxInstances; i++)
                        positions.Add(new Vector3(
                            Random.Range(-3f, 3f), 0, Random.Range(-3f, 3f)));
                    break;
            }

            return positions;
        }

        public void Cleanup()
        {
            foreach (var prop in _props)
            {
                if (prop.go != null) Destroy(prop.go);
            }
            _props.Clear();
        }

        private void OnDestroy() => Cleanup();
    }
}
