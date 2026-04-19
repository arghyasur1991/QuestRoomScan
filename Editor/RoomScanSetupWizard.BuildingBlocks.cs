// Programmatically install Meta XR Building Blocks (OVRCameraRig,
// Passthrough Underlay, PassthroughCameraAccess) so users get the same
// "Add to Scene" wiring the Building Blocks window provides — without
// having to open it.
//
// The actual install API (Meta.XR.BuildingBlocks.Editor.Utils +
// BlockData.InstallWithDependencies) is internal to its assembly and
// `Genesis.RoomScan.Editor` is not on its InternalsVisibleTo list, so
// we drive it via reflection. The runtime BuildingBlock component is
// public, which is enough to detect already-installed blocks.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Meta.XR.BuildingBlocks;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        // ── Block IDs (mirrored from
        //    com.meta.xr.sdk.core/Editor/BuildingBlocks/BlockDataIds.cs) ──
        const string BB_CAMERA_RIG                = "e47682b9-c270-40b1-b16d-90b627a5ce1b";
        const string BB_PASSTHROUGH               = "f0540b20-dfd6-420e-b20d-c270f88dc77e";
        const string BB_PASSTHROUGH_CAMERA_ACCESS = "0792d3af-c7d9-4f9c-a6f0-fd580a051e48";

        struct BlockSpec
        {
            public string Id;
            public string Label;
        }

        static readonly BlockSpec[] REQUIRED_BLOCKS =
        {
            new BlockSpec { Id = BB_CAMERA_RIG,                Label = "OVRCameraRig (Building Block)" },
            new BlockSpec { Id = BB_PASSTHROUGH,               Label = "Passthrough Layer (Building Block)" },
            new BlockSpec { Id = BB_PASSTHROUGH_CAMERA_ACCESS, Label = "Passthrough Camera Access (Building Block)" },
        };

        readonly Dictionary<string, bool> _bbPresent = new Dictionary<string, bool>();
        bool _bbAllPresent;

        // Passthrough scene-config state (separate from block presence so the
        // wizard reports both pieces independently).
        bool _ovrPassthroughEnabled;     // OVRManager.isInsightPassthroughEnabled
        bool _ovrCameraBackgroundClear;  // CenterEyeAnchor camera clears to transparent
        bool _ovrCameraPermissionOnStartup; // OVRManager.requestPassthroughCameraAccessPermissionOnStartup
        bool _ovrPassthroughReady;       // all of the above (only when passthrough block present)

        void RefreshBuildingBlocksState()
        {
            _bbPresent.Clear();
            var inScene = Object.FindObjectsByType<BuildingBlock>(FindObjectsSortMode.None);

            foreach (var spec in REQUIRED_BLOCKS)
            {
                _bbPresent[spec.Id] = inScene.Any(b => b != null && b.BlockId == spec.Id);
            }
            _bbAllPresent = _bbPresent.Values.All(v => v);

            RefreshPassthroughSceneState();
        }

        void RefreshPassthroughSceneState()
        {
            // OVRManager.isInsightPassthroughEnabled must be true whenever any
            // OVRPassthroughLayer (including the Building-Block one) lives in
            // the scene. We mirror Meta's check here so the row stays green
            // exactly when their setup tool would.
            var ovrManager = FindAny<OVRManager>();
            _ovrPassthroughEnabled = ovrManager != null && ovrManager.isInsightPassthroughEnabled;
            _ovrCameraPermissionOnStartup = ovrManager != null
                && ReadOvrManagerStartupPermFlag(ovrManager);

            _ovrCameraBackgroundClear = false;
            var rig = FindAny<OVRCameraRig>();
            if (rig != null && rig.centerEyeAnchor != null)
            {
                var cam = rig.centerEyeAnchor.GetComponent<Camera>();
                if (cam != null)
                {
                    _ovrCameraBackgroundClear =
                        cam.clearFlags == CameraClearFlags.SolidColor &&
                        cam.backgroundColor.a < 1f;
                }
            }

            // Only flag as "not ready" when the Passthrough block is present —
            // the warning is gated on that. The startup-permission flag is
            // separately gated on the PassthroughCameraAccess block (it lives
            // on OVRManager but is only meaningful when PCA is in use).
            bool passthroughBlockPresent = _bbPresent.TryGetValue(BB_PASSTHROUGH, out var p) && p;
            bool pcaBlockPresent = _bbPresent.TryGetValue(BB_PASSTHROUGH_CAMERA_ACCESS, out var c) && c;
            _ovrPassthroughReady = (!passthroughBlockPresent || (_ovrPassthroughEnabled && _ovrCameraBackgroundClear))
                                && (!pcaBlockPresent       || _ovrCameraPermissionOnStartup);
        }

        // OVRManager.requestPassthroughCameraAccessPermissionOnStartup is
        // declared `internal` in com.meta.xr.sdk.core, so we can't read or
        // write it directly from this assembly. SerializedObject lets us
        // poke it through Unity's serialization layer either way.
        static bool ReadOvrManagerStartupPermFlag(OVRManager m)
        {
            using var so = new UnityEditor.SerializedObject(m);
            var prop = so.FindProperty("requestPassthroughCameraAccessPermissionOnStartup");
            return prop != null && prop.boolValue;
        }

        static bool WriteOvrManagerStartupPermFlag(OVRManager m, bool value)
        {
            using var so = new UnityEditor.SerializedObject(m);
            var prop = so.FindProperty("requestPassthroughCameraAccessPermissionOnStartup");
            if (prop == null) return false;
            if (prop.boolValue == value) return false;
            prop.boolValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
            return true;
        }

        // ────────────────────────────────────────────────────────────────
        //  Reflection wrappers around Meta.XR.BuildingBlocks.Editor.Utils
        // ────────────────────────────────────────────────────────────────

        static Type _bbUtilsType;
        static MethodInfo _bbGetBlockData;
        static Type _bbBlockDataType;
        static MethodInfo _bbInstallWithDeps;
        static PropertyInfo _bbIsSingletonAlreadyPresent;

        static bool ResolveBuildingBlocksApi()
        {
            if (_bbUtilsType != null && _bbGetBlockData != null
                && _bbInstallWithDeps != null) return true;

            var asm = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Meta.XR.BuildingBlocks.Editor");
            if (asm == null)
            {
                Debug.LogError("[RoomScan Setup] Meta.XR.BuildingBlocks.Editor assembly not loaded — cannot install blocks.");
                return false;
            }

            _bbUtilsType = asm.GetType("Meta.XR.BuildingBlocks.Editor.Utils");
            if (_bbUtilsType == null)
            {
                Debug.LogError("[RoomScan Setup] Type Meta.XR.BuildingBlocks.Editor.Utils not found.");
                return false;
            }

            _bbGetBlockData = _bbUtilsType.GetMethod(
                "GetBlockData",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
                null, new[] { typeof(string) }, null);

            _bbBlockDataType = asm.GetType("Meta.XR.BuildingBlocks.Editor.BlockData");
            if (_bbBlockDataType == null)
            {
                Debug.LogError("[RoomScan Setup] Type Meta.XR.BuildingBlocks.Editor.BlockData not found.");
                return false;
            }

            // InstallWithDependencies has two overloads — pick the
            // single-GameObject one which we can pass null into.
            _bbInstallWithDeps = _bbBlockDataType.GetMethods(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "InstallWithDependencies"
                                     && m.GetParameters().Length == 1
                                     && m.GetParameters()[0].ParameterType == typeof(GameObject));

            _bbIsSingletonAlreadyPresent = _bbBlockDataType.GetProperty(
                "IsSingletonAndAlreadyPresent",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            return _bbGetBlockData != null && _bbInstallWithDeps != null;
        }

        static object GetBlockDataReflective(string blockId)
        {
            return _bbGetBlockData?.Invoke(null, new object[] { blockId });
        }

        static async Task InstallBlockReflective(object blockData)
        {
            if (blockData == null) return;
            var task = _bbInstallWithDeps.Invoke(blockData, new object[] { null }) as Task;
            if (task != null) await task;
        }

        /// <summary>
        /// Idempotently installs OVRCameraRig + Passthrough Underlay +
        /// PassthroughCameraAccess via the Meta XR Building Blocks pipeline.
        /// Skips any block already in the scene.
        /// </summary>
        async Task EnsureRequiredBuildingBlocksAsync()
        {
            if (!ResolveBuildingBlocksApi()) return;

            RefreshBuildingBlocksState();

            int installed = 0;
            foreach (var spec in REQUIRED_BLOCKS)
            {
                if (_bbPresent.TryGetValue(spec.Id, out var present) && present) continue;

                var data = GetBlockDataReflective(spec.Id);
                if (data == null)
                {
                    Debug.LogWarning($"[RoomScan Setup] Building Block '{spec.Label}' (id {spec.Id}) not found in registry — skipping.");
                    continue;
                }

                if (_bbIsSingletonAlreadyPresent != null
                    && (bool)_bbIsSingletonAlreadyPresent.GetValue(data))
                {
                    continue;
                }

                try
                {
                    await InstallBlockReflective(data);
                    installed++;
                    Debug.Log($"[RoomScan Setup] Installed Building Block: {spec.Label}");
                }
                catch (TargetInvocationException tex) when (tex.InnerException != null
                    && tex.InnerException.GetType().Name == "InstallationCancelledException")
                {
                    // Singleton already present at install-time, or another
                    // benign cancel — treat as success and move on.
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[RoomScan Setup] Failed to install Building Block '{spec.Label}': {ex.Message}");
                }
            }

            if (installed > 0)
            {
                AssetDatabase.SaveAssets();
                RefreshBuildingBlocksState();
            }

            // Building Blocks install puts the parts in the scene but doesn't
            // wire OVRManager.isInsightPassthroughEnabled or clear the
            // CenterEyeAnchor's background. Without these two follow-ups, Meta's
            // own "Outstanding Issues" panel shows two red rows and Passthrough
            // simply doesn't render.
            EnsurePassthroughSceneConfig();
        }

        /// <summary>
        /// Mirrors Meta's own `OVRProjectSetupPassthrough` and
        /// `PassthroughBuildingBlockRules` fixes:
        ///   * `OVRManager.isInsightPassthroughEnabled = true`
        ///   * Center eye camera `clearFlags = SolidColor`, `backgroundColor = (0,0,0,0)`
        /// Both are required for the Passthrough Building Block to actually
        /// render an underlay; Meta installs the BB but doesn't apply these
        /// fixes itself, so we bake them into the wizard.
        /// </summary>
        void EnsurePassthroughSceneConfig()
        {
            // No Passthrough block in the scene → nothing to do (and nothing
            // is "wrong"; Meta only gates these warnings on the block being
            // present).
            bool passthroughBlockPresent = _bbPresent.TryGetValue(BB_PASSTHROUGH, out var p) && p;
            if (!passthroughBlockPresent) return;

            int changed = 0;

            var ovrManager = FindAny<OVRManager>();
            if (ovrManager != null && !ovrManager.isInsightPassthroughEnabled)
            {
                Undo.RecordObject(ovrManager, "Enable Insight Passthrough");
                ovrManager.isInsightPassthroughEnabled = true;
                EditorUtility.SetDirty(ovrManager);
                changed++;
                Debug.Log("[RoomScan Setup] Enabled OVRManager.isInsightPassthroughEnabled.");
            }

            // Have OVRManager request HEADSET_CAMERA at app startup. Without
            // this, PCA's permission dialog only appears once the user
            // triggers a scan — by which point the scanner has already kicked
            // off in degraded depth-only mode. Game code that wants a
            // deterministic "asking for permission" UI state should still
            // call RoomScanSession.RequestCameraPermissionAsync() before
            // StartScan() as defense-in-depth (covers the user dismissing
            // the startup dialog). Mirrors Meta's
            // PassthroughCameraAccessProjectSetup Optional task.
            bool pcaBlockPresent = _bbPresent.TryGetValue(BB_PASSTHROUGH_CAMERA_ACCESS, out var pp) && pp;
            if (ovrManager != null && pcaBlockPresent
                && WriteOvrManagerStartupPermFlag(ovrManager, true))
            {
                changed++;
                Debug.Log("[RoomScan Setup] Enabled OVRManager.requestPassthroughCameraAccessPermissionOnStartup.");
            }

            var rig = FindAny<OVRCameraRig>();
            if (rig != null && rig.centerEyeAnchor != null)
            {
                var cam = rig.centerEyeAnchor.GetComponent<Camera>();
                if (cam != null)
                {
                    bool needsClear =
                        cam.clearFlags != CameraClearFlags.SolidColor ||
                        cam.backgroundColor.a >= 1f;

                    if (needsClear)
                    {
                        Undo.RecordObject(cam, "Clear Camera Background for Passthrough");
                        cam.clearFlags = CameraClearFlags.SolidColor;
                        cam.backgroundColor = Color.clear;
                        EditorUtility.SetDirty(cam);
                        changed++;
                        Debug.Log("[RoomScan Setup] Set CenterEyeAnchor camera background to transparent.");
                    }
                }
                else
                {
                    Debug.LogWarning("[RoomScan Setup] OVRCameraRig has no Camera under CenterEyeAnchor — skipping background clear.");
                }
            }

            if (changed > 0)
            {
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                RefreshPassthroughSceneState();
            }
        }
    }
}
