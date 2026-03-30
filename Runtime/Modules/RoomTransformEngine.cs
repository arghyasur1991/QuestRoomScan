using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Central orchestrator that owns all transformation layers and drives them
    /// from a single float progress (0-1). Game developers assign a ThemePack
    /// and call SetProgress — all layers update automatically.
    /// </summary>
    public class RoomTransformEngine : MonoBehaviour
    {
        private RoomLightingEngine _lighting;
        private RoomParticleManager _particles;
        private RoomPropSpawner _props;
        private RoomAudioManager _audio;

        private ThemePack _theme;
        private SceneObjectRegistry _registry;
        private bool _initialized;
        private float _progress;

        /// <summary>Current global transformation progress (0 = real room, 1 = fully themed).</summary>
        public float Progress => _progress;
        public bool IsInitialized => _initialized;

        /// <summary>
        /// Initialize all transformation layers with the given theme and scene data.
        /// Call this after the refined mesh is ready and the SceneObjectRegistry is populated.
        /// </summary>
        public void Initialize(ThemePack theme, SceneObjectRegistry registry, Bounds roomBounds)
        {
            if (_initialized) Cleanup();

            _theme = theme;
            _registry = registry;

            _lighting = gameObject.AddComponent<RoomLightingEngine>();
            _lighting.Initialize(registry, theme);

            _particles = gameObject.AddComponent<RoomParticleManager>();
            _particles.Initialize(registry, theme, roomBounds);

            _props = gameObject.AddComponent<RoomPropSpawner>();
            _props.Initialize(registry, theme);

            _audio = gameObject.AddComponent<RoomAudioManager>();
            _audio.Initialize(registry, theme);

            _initialized = true;
            Logger.Info("[RoomTransformEngine] Initialized with all layers");
        }

        /// <summary>
        /// Sets transformation progress across all layers. Each layer evaluates its
        /// own activation curve from the ThemePack.
        /// </summary>
        public void SetProgress(float progress)
        {
            _progress = Mathf.Clamp01(progress);
            if (!_initialized) return;

            _lighting?.SetProgress(_progress);
            _particles?.SetProgress(_progress);
            _props?.SetProgress(_progress);
            _audio?.SetProgress(_progress);
        }

        /// <summary>Tears down all layers and cleans up spawned objects.</summary>
        public void Cleanup()
        {
            if (_lighting != null) { _lighting.Cleanup(); Destroy(_lighting); }
            if (_particles != null) { _particles.Cleanup(); Destroy(_particles); }
            if (_props != null) { _props.Cleanup(); Destroy(_props); }
            if (_audio != null) { _audio.Cleanup(); Destroy(_audio); }

            _lighting = null;
            _particles = null;
            _props = null;
            _audio = null;
            _initialized = false;
        }

        private void OnDestroy() => Cleanup();
    }
}
