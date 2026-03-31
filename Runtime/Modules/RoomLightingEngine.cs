using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Manipulates virtual lights based on detected light sources from SceneObjectRegistry.
    /// Places lights at lamp/screen positions, controls global dimming, per-light flicker,
    /// and color temperature shift — all driven by transformation progress.
    /// </summary>
    public class RoomLightingEngine : MonoBehaviour
    {
        private readonly List<Light> _virtualLights = new();
        private readonly List<float> _flickerOffsets = new();
        private ThemePack _theme;
        private float _progress;
        private Light _globalLight;
        private float _originalAmbientIntensity;
        private Color _originalAmbientColor;

        private static readonly Color WarmColor = new Color(1f, 0.9f, 0.7f);
        private static readonly Color ColdColor = new Color(0.7f, 0.8f, 1f);

        public void Initialize(SceneObjectRegistry registry, ThemePack theme)
        {
            _theme = theme;
            _originalAmbientIntensity = RenderSettings.ambientIntensity;
            _originalAmbientColor = RenderSettings.ambientLight;

            if (registry == null) return;

            var lamps = registry.FindByLabel("lamp");
            var screens = registry.FindByLabel("screen");

            foreach (var lamp in lamps) CreateVirtualLight(lamp, 2f);
            foreach (var screen in screens) CreateVirtualLight(screen, 1f);
        }

        private void CreateVirtualLight(SceneObject obj, float baseIntensity)
        {
            var go = new GameObject($"VirtualLight_{obj.label}_{obj.id}");
            go.transform.SetParent(transform);
            go.transform.position = obj.position;

            var light = go.AddComponent<Light>();
            light.type = LightType.Point;
            light.intensity = baseIntensity;
            light.range = Mathf.Max(obj.size.magnitude * 2f, 3f);
            light.color = WarmColor;
            light.shadows = LightShadows.None;

            _virtualLights.Add(light);
            _flickerOffsets.Add(Random.Range(0f, 100f));
        }

        public void SetProgress(float progress)
        {
            _progress = progress;
            if (_theme == null) return;

            float layerIntensity = _theme.lightingCurve.Evaluate(progress);
            float dimFactor = _theme.lightDimCurve.Evaluate(progress);

            RenderSettings.ambientIntensity = _originalAmbientIntensity * Mathf.Lerp(1f, dimFactor, layerIntensity);

            float tempShift = _theme.colorTemperatureShift * layerIntensity;
            RenderSettings.ambientLight = Color.Lerp(_originalAmbientColor,
                Color.Lerp(WarmColor, ColdColor, tempShift), layerIntensity);

            for (int i = 0; i < _virtualLights.Count; i++)
            {
                if (_virtualLights[i] == null) continue;

                float baseDim = Mathf.Lerp(1f, dimFactor, layerIntensity);
                float flicker = 1f;

                if (_theme.flickerIntensity > 0.001f && layerIntensity > 0.1f)
                {
                    float t = Time.time * _theme.flickerSpeed + _flickerOffsets[i];
                    float noise = Mathf.PerlinNoise(t, _flickerOffsets[i] * 0.7f);
                    float spike = Mathf.Abs(Mathf.Sin(t * 7.3f + _flickerOffsets[i]));
                    float raw = noise * 0.7f + spike * 0.3f;
                    flicker = Mathf.Lerp(1f, raw, _theme.flickerIntensity * layerIntensity);
                }

                _virtualLights[i].intensity = baseDim * flicker * 2f;
                _virtualLights[i].color = Color.Lerp(WarmColor, ColdColor, tempShift);
            }
        }

        public void Cleanup()
        {
            foreach (var light in _virtualLights)
            {
                if (light != null) Destroy(light.gameObject);
            }
            _virtualLights.Clear();
            _flickerOffsets.Clear();

            RenderSettings.ambientIntensity = _originalAmbientIntensity;
            RenderSettings.ambientLight = _originalAmbientColor;
        }

        private void OnDestroy() => Cleanup();
    }
}
