// Editor play-mode safety net for Quest-only components.
//
// When you hit Play in the Unity Editor without an active XR loader (no
// Quest, no Quest Link), Meta's PassthroughCameraAccess and Unity's
// AROcclusionManager / ARCameraManager throw NREs in Update because no AR
// subsystem ever materialised. The error wall makes the Editor unusable
// for non-XR work (UI, AI, gameplay scaffolding, anything else).
//
// This guard hooks RuntimeInitializeOnLoadMethod (which fires only when
// the player runtime spins up — i.e. on play-mode entry in Editor or on
// device start in a build) and disables those components inside the
// play-mode scene clone the moment each scene finishes loading.
//
// Critically: edits to the play-mode scene clone do NOT propagate back to
// the saved scene asset, so the next time you stop and re-enter play mode
// the components come back enabled. On device, XRRuntimeGuard.IsXRActive
// is true and this guard is a complete no-op. The whole class is also
// wrapped in UNITY_EDITOR so it never reaches built players.

#if UNITY_EDITOR

using Meta.XR;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;

namespace Genesis.RoomScan
{
    internal static class EditorPlayModeXRGuard
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Hook()
        {
            // Quest device, Quest Link, anything with a real OpenXR loader →
            // these components work normally; nothing to guard against.
            if (XRRuntimeGuard.IsXRActive) return;

            SceneManager.sceneLoaded -= OnSceneLoaded;
            SceneManager.sceneLoaded += OnSceneLoaded;

            // Cover the active scene that may have been loaded before
            // BeforeSceneLoad fired (single-scene play sessions).
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                var s = SceneManager.GetSceneAt(i);
                if (s.isLoaded) DisableInScene(s);
            }
        }

        static void OnSceneLoaded(Scene scene, LoadSceneMode mode) => DisableInScene(scene);

        static void DisableInScene(Scene scene)
        {
            int flipped = 0;
            foreach (var go in scene.GetRootGameObjects())
            {
                flipped += DisableInHierarchy<PassthroughCameraAccess>(go);
                flipped += DisableInHierarchy<AROcclusionManager>(go);
                flipped += DisableInHierarchy<ARCameraManager>(go);
            }

            if (flipped > 0)
            {
                Debug.Log($"[RoomScan] EditorPlayModeXRGuard: disabled {flipped} Quest-runtime " +
                          "component(s) for this play session (no XR loader active). " +
                          "Scene asset is NOT modified — this only applies to the play-mode clone.");
            }
        }

        static int DisableInHierarchy<T>(GameObject root) where T : Behaviour
        {
            int n = 0;
            foreach (var c in root.GetComponentsInChildren<T>(true))
            {
                if (!c.enabled) continue;
                c.enabled = false;
                n++;
            }
            return n;
        }
    }
}

#endif
