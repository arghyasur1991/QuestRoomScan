using System;
using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// All actions that can be bound to controller buttons.
    /// </summary>
    public enum ScanAction
    {
        None,
        ToggleScanning,
        FreezeInView,
        UnfreezeInView,
        CycleRenderMode,
        ClearAllData,
        ExportPointCloud,
        StartServerTraining,
        ToggleDebugMenu,
    }

    /// <summary>
    /// Maps a single OVRInput button to a <see cref="ScanAction"/>.
    /// Disable individual bindings by setting <see cref="enabled"/> to false,
    /// or remove/replace the entire <see cref="RoomScanInputHandler"/> component.
    /// </summary>
    [Serializable]
    public class ScanInputBinding
    {
        public ScanAction action = ScanAction.None;
        public OVRInput.Button button = OVRInput.Button.None;
        public bool enabled = true;
    }

    /// <summary>
    /// Optional component that polls OVRInput each frame and calls the
    /// corresponding <see cref="RoomScanner"/> public API methods.
    ///
    /// Clients can:
    ///   - Edit bindings in the Inspector or at runtime via <see cref="Bindings"/>.
    ///   - Disable individual bindings or the entire component.
    ///   - Remove this component entirely and call RoomScanner APIs directly.
    ///   - Add/remove bindings at runtime via <see cref="AddBinding"/>/<see cref="RemoveBinding"/>.
    /// </summary>
    public class RoomScanInputHandler : MonoBehaviour
    {
        [SerializeField, Tooltip("Controller button → action mappings. Editable at runtime.")]
        private List<ScanInputBinding> bindings = new()
        {
            new() { action = ScanAction.ToggleDebugMenu,    button = OVRInput.Button.Start, enabled = true },
            new() { action = ScanAction.FreezeInView,       button = OVRInput.Button.Three, enabled = true },
            new() { action = ScanAction.UnfreezeInView,     button = OVRInput.Button.Four,  enabled = true },
            new() { action = ScanAction.CycleRenderMode,    button = OVRInput.Button.Two,   enabled = true },
            new() { action = ScanAction.StartServerTraining, button = OVRInput.Button.One,  enabled = true },
        };

        /// <summary>
        /// The live bindings list. Mutate freely at runtime.
        /// </summary>
        public List<ScanInputBinding> Bindings => bindings;

        /// <summary>
        /// Convenience: add a new binding at runtime.
        /// </summary>
        public void AddBinding(ScanAction action, OVRInput.Button button)
        {
            bindings.Add(new ScanInputBinding { action = action, button = button, enabled = true });
        }

        /// <summary>
        /// Remove all bindings for a given action.
        /// </summary>
        public void RemoveBindingsForAction(ScanAction action)
        {
            bindings.RemoveAll(b => b.action == action);
        }

        /// <summary>
        /// Remove all bindings for a given button.
        /// </summary>
        public void RemoveBindingsForButton(OVRInput.Button button)
        {
            bindings.RemoveAll(b => b.button == button);
        }

        /// <summary>
        /// Clear all bindings. After this, no controller input will trigger any action.
        /// </summary>
        public void ClearAllBindings()
        {
            bindings.Clear();
        }

        private void Update()
        {
            var scanner = RoomScanner.Instance;
            if (scanner == null) return;

            for (int i = 0; i < bindings.Count; i++)
            {
                var b = bindings[i];
                if (!b.enabled || b.action == ScanAction.None || b.button == OVRInput.Button.None)
                    continue;

                if (!OVRInput.GetDown(b.button))
                    continue;

                ExecuteAction(scanner, b.action);
            }
        }

        private static void ExecuteAction(RoomScanner scanner, ScanAction action)
        {
            switch (action)
            {
                case ScanAction.ToggleScanning:
                    scanner.ToggleScanning();
                    break;
                case ScanAction.FreezeInView:
                    scanner.FreezeInView();
                    break;
                case ScanAction.UnfreezeInView:
                    scanner.UnfreezeInView();
                    break;
                case ScanAction.CycleRenderMode:
                    scanner.CycleRenderMode();
                    break;
                case ScanAction.ClearAllData:
                    scanner.ClearAllData();
                    break;
                case ScanAction.ExportPointCloud:
                    _ = scanner.ExportPointCloudAsync();
                    break;
                case ScanAction.StartServerTraining:
                    scanner.StartServerTraining();
                    break;
                case ScanAction.ToggleDebugMenu:
                    scanner.ToggleDebugMenu();
                    break;
            }
        }
    }
}
