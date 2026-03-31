using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Position-aware audio system driven by transformation progress.
    /// Manages ambient loops, spatial sources anchored to detected objects,
    /// and one-shot triggers at progress thresholds.
    /// </summary>
    public class RoomAudioManager : MonoBehaviour
    {
        private ThemePack _theme;
        private float _progress;
        private AudioSource _ambientSource;
        private readonly List<AudioSource> _spatialSources = new();
        private readonly HashSet<int> _firedTriggers = new();

        public void Initialize(SceneObjectRegistry registry, ThemePack theme)
        {
            _theme = theme;

            if (theme.ambientAudio != null)
            {
                _ambientSource = gameObject.AddComponent<AudioSource>();
                _ambientSource.clip = theme.ambientAudio;
                _ambientSource.loop = true;
                _ambientSource.spatialBlend = 0f;
                _ambientSource.volume = 0f;
                _ambientSource.Play();
            }

            if (registry != null && theme.transformSound != null)
            {
                var furniture = registry.FindBySurface(SurfaceType.Furniture);
                foreach (var obj in furniture)
                {
                    var go = new GameObject($"SpatialAudio_{obj.label}");
                    go.transform.SetParent(transform);
                    go.transform.position = obj.position;

                    var src = go.AddComponent<AudioSource>();
                    src.clip = theme.transformSound;
                    src.loop = true;
                    src.spatialBlend = 1f;
                    src.minDistance = 0.5f;
                    src.maxDistance = 5f;
                    src.volume = 0f;
                    src.Play();

                    _spatialSources.Add(src);
                }
            }
        }

        public void SetProgress(float progress)
        {
            _progress = progress;
            if (_theme == null) return;

            float layerIntensity = _theme.audioCurve.Evaluate(progress);

            if (_ambientSource != null)
                _ambientSource.volume = layerIntensity * 0.6f;

            foreach (var src in _spatialSources)
            {
                if (src != null) src.volume = layerIntensity * 0.3f;
            }

            if (_theme.progressTriggers != null)
            {
                for (int i = 0; i < _theme.progressTriggers.Count; i++)
                {
                    if (_firedTriggers.Contains(i)) continue;
                    var trigger = _theme.progressTriggers[i];
                    if (progress >= trigger.progressThreshold && trigger.clip != null)
                    {
                        _firedTriggers.Add(i);
                        PlayOneShot(trigger);
                    }
                }
            }
        }

        private void PlayOneShot(ProgressAudioTrigger trigger)
        {
            if (trigger.spatial)
            {
                if (_spatialSources.Count > 0)
                {
                    var src = _spatialSources[Random.Range(0, _spatialSources.Count)];
                    src.PlayOneShot(trigger.clip);
                }
            }
            else
            {
                if (_ambientSource != null)
                    _ambientSource.PlayOneShot(trigger.clip);
                else
                    AudioSource.PlayClipAtPoint(trigger.clip, transform.position);
            }
        }

        public void ResetTriggers() => _firedTriggers.Clear();

        public void Cleanup()
        {
            if (_ambientSource != null)
            {
                _ambientSource.Stop();
                Destroy(_ambientSource);
            }

            foreach (var src in _spatialSources)
            {
                if (src != null)
                {
                    src.Stop();
                    Destroy(src.gameObject);
                }
            }
            _spatialSources.Clear();
            _firedTriggers.Clear();
        }

        private void OnDestroy() => Cleanup();
    }
}
