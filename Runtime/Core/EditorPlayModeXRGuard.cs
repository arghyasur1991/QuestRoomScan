// Editor play-mode safety net for Quest-only components.
//
// When you hit Play in the Unity Editor without an active XR loader (no
// Quest, no Quest Link), Meta's PassthroughCameraAccess and Unity's
// AROcclusionManager / ARCameraManager / ARSession throw NREs in
// OnEnable / Update because no AR subsystem ever materialised. The error
// wall makes the Editor unusable for non-XR work (UI, AI, gameplay
// scaffolding, anything else).
//
// We can't catch this from a [RuntimeInitializeOnLoadMethod] hook —
// SceneManager.sceneLoaded fires AFTER every Awake/OnEnable in that
// scene, so by the time we'd disable the components their OnEnable has
// already crashed (and the subsequent disable triggers a secondary NRE
// in AROcclusionManager.OnDisable → DestroyTextures).
//
// Instead, this is a MonoBehaviour wedged into the scene by the
// RoomScanSetupWizard. With [DefaultExecutionOrder(int.MinValue)] its
// Awake runs before any AR component's Awake. Unity then runs OnEnable
// on every still-enabled component — so disabling the AR pieces during
// our Awake skips their OnEnable entirely, and no NRE chain ever fires.
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
    [DefaultExecutionOrder(int.MinValue)]
    [DisallowMultipleComponent]
    [AddComponentMenu("Genesis/Room Scan/Editor Play Mode XR Guard")]
    public sealed class EditorPlayModeXRGuard : MonoBehaviour
    {
#if UNITY_EDITOR
        void Awake()
        {
            // Quest device, Quest Link, anything with a real OpenXR loader →
            // these components work normally; nothing to guard against.
            if (XRRuntimeGuard.IsXRActive) return;

            DisableInScene(gameObject.scene);

            // Cover scenes loaded additively after this one.
            SceneManager.sceneLoaded -= OnSceneLoaded;
            SceneManager.sceneLoaded += OnSceneLoaded;
        }

        void OnDestroy()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
        }

        static void OnSceneLoaded(Scene scene, LoadSceneMode mode) => DisableInScene(scene);

        static void DisableInScene(Scene scene)
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

            if (flipped > 0)
            {
                Debug.Log($"[RoomScan] EditorPlayModeXRGuard: disabled {flipped} Quest-runtime " +
                          $"component(s) in scene '{scene.name}' for this play session " +
                          "(no XR loader active). Scene asset is NOT modified — this only " +
                          "applies to the play-mode clone.");
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
#endif
    }
}
