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
using UnityEngine;

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

        void RefreshBuildingBlocksState()
        {
            _bbPresent.Clear();
            var inScene = UnityEngine.Object
                .FindObjectsByType<BuildingBlock>(FindObjectsSortMode.None);

            foreach (var spec in REQUIRED_BLOCKS)
            {
                _bbPresent[spec.Id] = inScene.Any(b => b != null && b.BlockId == spec.Id);
            }
            _bbAllPresent = _bbPresent.Values.All(v => v);
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
        }
    }
}
