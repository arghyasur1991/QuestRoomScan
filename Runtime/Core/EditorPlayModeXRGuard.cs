// Editor play-mode safety net for Quest-only components.
//
// When you hit Play in the Unity Editor without an active XR loader (no
// Quest, no Quest Link), Meta's PassthroughCameraAccess and Unity's
// AROcclusionManager / ARCameraManager / ARSession throw a wall of
// errors in OnEnable / Update because no AR subsystem ever materialised.
// That makes the Editor unusable for non-XR work (UI, AI, gameplay
// scaffolding, anything else).
//
// We can't catch this from a [RuntimeInitializeOnLoadMethod] hook —
// SceneManager.sceneLoaded fires AFTER every Awake/OnEnable in that
// scene, so by the time we'd disable the components their OnEnable has
// already run (and the subsequent disable triggers a secondary NRE in
// AROcclusionManager.OnDisable → DestroyTextures).
//
// Instead, this is a MonoBehaviour wedged into the scene by the
// RoomScanSetupWizard. With [DefaultExecutionOrder(-32000)] its Awake
// runs before any AR component's Awake. Unity completes the Awake phase
// for every component before running OnEnable on any of them, so
// disabling the AR pieces in our Awake makes Unity skip their OnEnable
// entirely — no NRE chain ever fires.
//
// Why -32000 and not int.MinValue: Unity's documented execution-order
// range is [-32000, 32000] (clamped by MonoImporter). int.MinValue is
// silently treated as 0, which competes with AR/PCA's default 0 and
// loses to scene/component declaration order — exactly the bug this
// file was originally trying to fix.
//
// On Quest device the XR loader is active, so the whole guard early-outs
// in Awake. The component is harmless in builds and stays in the saved
// scene asset.

using UnityEngine;

#if UNITY_EDITOR
using Meta.XR;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
#endif

namespace Genesis.RoomScan
{
    /// <summary>
    /// Disables Quest-only components inside the Editor play-mode scene
    /// when no XR loader is active. No-op on device. Wizard adds this to
    /// the RoomScan root automatically.
    /// </summary>
    [DefaultExecutionOrder(-32000)]
    [DisallowMultipleComponent]
    [AddComponentMenu("Genesis/Room Scan/Editor Play Mode XR Guard")]
    public sealed class EditorPlayModeXRGuard : MonoBehaviour
    {
#if UNITY_EDITOR
        // Per-play-session marker: skip the redundant sceneLoaded pass for
        // any scene the Awake-time pass already handled.
        static readonly System.Collections.Generic.HashSet<int> s_handledScenes = new();

        void Awake()
        {
            // Quest device, Quest Link, anything with a real OpenXR loader →
            // these components work normally; nothing to guard against.
            if (XRRuntimeGuard.IsXRActive) return;

            DisableInScene(gameObject.scene, fromAwake: true);

            // Cover scenes loaded additively after this one.
            SceneManager.sceneLoaded -= OnSceneLoaded;
            SceneManager.sceneLoaded += OnSceneLoaded;
        }

        void OnDestroy()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
        }

        static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            // Awake already cleaned this scene; skip — running again post-OnEnable
            // would only trigger AROcclusionManager.OnDisable → DestroyTextures NRE.
            if (s_handledScenes.Contains(scene.handle)) return;
            DisableInScene(scene, fromAwake: false);
        }

        static void DisableInScene(Scene scene, bool fromAwake)
        {
            if (!scene.isLoaded) return;

            int flipped = 0;
            foreach (var go in scene.GetRootGameObjects())
            {
                flipped += DisableInHierarchy<PassthroughCameraAccess>(go);
                flipped += DisableInHierarchy<ARSession>(go);
                flipped += DisableInHierarchy<AROcclusionManager>(go);
                flipped += DisableInHierarchy<ARCameraManager>(go);
            }

            s_handledScenes.Add(scene.handle);

            if (flipped > 0)
            {
                Debug.Log($"[RoomScan] EditorPlayModeXRGuard: disabled {flipped} Quest-runtime " +
                          $"component(s) in scene '{scene.name}' " +
                          $"(phase: {(fromAwake ? "Awake (pre-OnEnable)" : "sceneLoaded (post-OnEnable)")}). " +
                          "Scene asset is NOT modified — this only applies to the play-mode clone.");
            }
        }

        static int DisableInHierarchy<T>(GameObject root) where T : Behaviour
        {
            int n = 0;
            foreach (var c in root.GetComponentsInChildren<T>(true))
            {
                if (c == null || !c.enabled) continue;
                try
                {
                    // AR components whose OnEnable half-failed (no XR subsystem)
                    // can NRE inside their own OnDisable → DestroyTextures the
                    // moment we flip enabled. Swallow it; the component still
                    // ends up disabled and the rest of the play session is clean.
                    c.enabled = false;
                    n++;
                }
                catch (System.Exception)
                {
                    n++;
                }
            }
            return n;
        }
#endif
    }
}
