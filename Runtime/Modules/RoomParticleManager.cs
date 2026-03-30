using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Volumetric and contextual particle effects driven by transformation progress.
    /// Spawns fog volumes, dust motes, and contextual emitters from SceneObjectRegistry.
    /// </summary>
    public class RoomParticleManager : MonoBehaviour
    {
        private ThemePack _theme;
        private float _progress;
        private GameObject _fogInstance;
        private GameObject _dustInstance;
        private readonly List<GameObject> _contextualEmitters = new();

        public void Initialize(SceneObjectRegistry registry, ThemePack theme, Bounds roomBounds)
        {
            _theme = theme;

            if (theme.fogPrefab != null)
            {
                _fogInstance = Instantiate(theme.fogPrefab, transform);
                _fogInstance.transform.position = roomBounds.center;
                _fogInstance.SetActive(false);
            }

            if (theme.dustPrefab != null)
            {
                _dustInstance = Instantiate(theme.dustPrefab, transform);
                _dustInstance.transform.position = roomBounds.center;
                _dustInstance.SetActive(false);
            }

            if (registry != null && theme.contextualVFXPrefabs != null)
            {
                foreach (var prefab in theme.contextualVFXPrefabs)
                {
                    if (prefab == null) continue;
                    var furnitureObjects = registry.FindBySurface(SurfaceType.Furniture);
                    foreach (var obj in furnitureObjects)
                    {
                        var emitter = Instantiate(prefab, transform);
                        emitter.transform.position = obj.position + Vector3.up * obj.size.y * 0.5f;
                        emitter.SetActive(false);
                        _contextualEmitters.Add(emitter);
                    }
                }
            }
        }

        public void SetProgress(float progress)
        {
            _progress = progress;
            if (_theme == null) return;

            float layerIntensity = _theme.particleCurve.Evaluate(progress);

            if (_fogInstance != null)
            {
                _fogInstance.SetActive(layerIntensity > 0.05f);
                ScaleParticleEmission(_fogInstance, layerIntensity);
            }

            if (_dustInstance != null)
            {
                _dustInstance.SetActive(layerIntensity > 0.1f);
                ScaleParticleEmission(_dustInstance, layerIntensity);
            }

            foreach (var emitter in _contextualEmitters)
            {
                if (emitter == null) continue;
                emitter.SetActive(layerIntensity > 0.2f);
                ScaleParticleEmission(emitter, layerIntensity);
            }
        }

        private static void ScaleParticleEmission(GameObject obj, float intensity)
        {
            var ps = obj.GetComponentInChildren<ParticleSystem>();
            if (ps == null) return;
            var emission = ps.emission;
            emission.rateOverTimeMultiplier = intensity;
        }

        public void Cleanup()
        {
            if (_fogInstance != null) Destroy(_fogInstance);
            if (_dustInstance != null) Destroy(_dustInstance);
            foreach (var e in _contextualEmitters)
            {
                if (e != null) Destroy(e);
            }
            _contextualEmitters.Clear();
        }

        private void OnDestroy() => Cleanup();
    }
}
