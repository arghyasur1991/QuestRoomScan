using System;
using System.IO;
using System.Linq;
using Genesis.RoomScan.UI;
using Meta.XR;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// Custom inspector for <see cref="RoomScanner"/> that shows attached modules
    /// and provides an "Add Module" dropdown for optional features.
    /// </summary>
    [CustomEditor(typeof(RoomScanner))]
    public class RoomScannerEditor : UnityEditor.Editor
    {
        static readonly (string label, Type type, Type[] extraDeps, bool triggerXAtlasBuild)[] ModuleOptions =
        {
            ("Passthrough Camera", typeof(PassthroughCameraProvider), new[] { typeof(PassthroughCameraAccess) }, false),
            ("Triplanar Cache", typeof(TriplanarCache), null, false),
            ("Texture Refinement", typeof(TextureRefinement), null, true),
            ("Point Cloud Export", typeof(PointCloudExporter), null, false),
            ("Input Handler", typeof(RoomScanInputHandler), null, false),
            ("Debug Overlays", typeof(CameraDebugOverlay), new[] { typeof(DepthDebugOverlay) }, false),
        };

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Modules", EditorStyles.boldLabel);

            var scanner = (RoomScanner)target;
            var modules = scanner.GetComponents<IRoomScanModule>();

            if (modules.Length == 0)
            {
                EditorGUILayout.HelpBox("No optional modules attached.", MessageType.Info);
            }
            else
            {
                foreach (var m in modules)
                {
                    if (m is RoomScanPersistence || m is RoomAnchorManager)
                        continue;
                    EditorGUILayout.LabelField($"  \u2022 {m.ModuleName}", EditorStyles.miniLabel);
                }
            }

            EditorGUILayout.Space(4);
            if (EditorGUILayout.DropdownButton(new GUIContent("Add Module\u2026"), FocusType.Keyboard))
                ShowModuleMenu(scanner);
        }

        void ShowModuleMenu(RoomScanner scanner)
        {
            var menu = new GenericMenu();

            foreach (var (label, type, extraDeps, triggerBuild) in ModuleOptions)
            {
                bool alreadyAttached = scanner.GetComponent(type) != null;
                if (alreadyAttached)
                {
                    menu.AddDisabledItem(new GUIContent($"{label} (attached)"));
                }
                else
                {
                    menu.AddItem(new GUIContent(label), false, () =>
                    {
                        Undo.RegisterCompleteObjectUndo(scanner.gameObject, $"Add {label}");
                        var added = Undo.AddComponent(scanner.gameObject, type);
                        if (extraDeps != null)
                        {
                            foreach (var dep in extraDeps)
                            {
                                if (scanner.GetComponent(dep) == null)
                                {
                                    var depComp = Undo.AddComponent(scanner.gameObject, dep);
                                    RoomScanSetupWizard.WireComponent(depComp);
                                }
                            }
                        }

                        RoomScanSetupWizard.WireComponent(added);

                        if (triggerBuild)
                            EnsureXAtlasPlugins();

                        EditorUtility.SetDirty(scanner.gameObject);
                    });
                }
            }

            // GSplat module — full setup via the wizard helper
            bool hasGSplat = scanner.GetComponent<IGSplatProvider>() != null;
            var gsplatType = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(a => { try { return a.GetTypes(); } catch { return Type.EmptyTypes; } })
                .FirstOrDefault(t => t.Name == "GSplatManager" && typeof(IRoomScanModule).IsAssignableFrom(t));

            if (gsplatType != null)
            {
                if (hasGSplat)
                    menu.AddDisabledItem(new GUIContent("Gaussian Splat (attached)"));
                else
                    menu.AddItem(new GUIContent("Gaussian Splat"), false, () =>
                    {
                        Undo.RegisterCompleteObjectUndo(scanner.gameObject, "Add Gaussian Splat");
                        RoomScanSetupWizard.SetupGSplatModule(scanner.gameObject);
                        EditorUtility.SetDirty(scanner.gameObject);
                    });
            }

            // Debug Menu — lives on a child GameObject with UIDocument
            bool hasDebugMenu = scanner.GetComponentInChildren<DebugMenuController>(true) != null;
            if (hasDebugMenu)
                menu.AddDisabledItem(new GUIContent("Debug Menu (attached)"));
            else
                menu.AddItem(new GUIContent("Debug Menu"), false, () =>
                {
                    Undo.RegisterCompleteObjectUndo(scanner.gameObject, "Add Debug Menu");

                    var debugGo = new GameObject("DebugMenu");
                    debugGo.transform.SetParent(scanner.transform);
                    Undo.RegisterCreatedObjectUndo(debugGo, "Create DebugMenu");

                    Undo.AddComponent<UIDocument>(debugGo);
                    Undo.AddComponent<DebugMenuController>(debugGo);

                    RoomScanSetupWizard.EnsureDebugMenuAssets();
                    RoomScanSetupWizard.EnsureVRInput();

                    EditorUtility.SetDirty(scanner.gameObject);
                });

            menu.ShowAsContext();
        }

        static void EnsureXAtlasPlugins()
        {
            string pkgRoot = Path.GetFullPath("Packages/com.genesis.roomscan/Runtime");
            string androidPlugin = Path.Combine(pkgRoot, "Plugins/Android/libxatlas.so");
            string macPlugin = Path.Combine(pkgRoot, "Plugins/macOS/libxatlas.bundle");

            if (File.Exists(androidPlugin) && File.Exists(macPlugin))
                return;

            RoomScanSetupWizard.BuildXAtlasPlugin();
        }
    }
}
