using System.Linq;
using Meta.XR;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using Unity.XR.CoreUtils;

namespace Genesis.RoomScan.Editor
{
    public class RoomScanSetupWizard : EditorWindow
    {
        double _lastRefresh;
        const double REFRESH_SEC = 0.8;
        Vector2 _scroll;

        // Cached scene state
        ARSession _arSession;
        AROcclusionManager _arOcclusion;
        GameObject _cameraRig;

        DepthCapture _depthCapture;
        VolumeIntegrator _volumeIntegrator;
        ChunkManager _chunkManager;
        TextureProjector _textureProjector;
        RoomScanner _roomScanner;
        PassthroughCameraProvider _cameraProvider;
        PassthroughCameraAccess _pcaComponent;

        bool _depthCaptureWired, _volumeWired, _projectorWired, _chunkMatWired;

        // Style
        static readonly Color COL_OK   = new(0.25f, 0.82f, 0.35f);
        static readonly Color COL_WARN = new(0.95f, 0.78f, 0.15f);
        static readonly Color COL_MISS = new(0.92f, 0.28f, 0.25f);
        static readonly Color COL_INFO = new(0.45f, 0.72f, 0.95f);
        static readonly Color COL_SECT = new(0.18f, 0.18f, 0.22f);

        const string PKG = "Packages/com.genesis.roomscan/Runtime/Shaders/";

        [MenuItem("RoomScan/Setup Scene")]
        static void Open()
        {
            var w = GetWindow<RoomScanSetupWizard>("Room Scan Setup");
            w.minSize = new Vector2(420, 480);
        }

        void OnEnable()  => Refresh();
        void OnFocus()   => Refresh();

        void Update()
        {
            if (EditorApplication.timeSinceStartup - _lastRefresh > REFRESH_SEC)
            {
                Refresh();
                Repaint();
            }
        }

        // =================================================================
        //  REFRESH
        // =================================================================

        void Refresh()
        {
            _lastRefresh = EditorApplication.timeSinceStartup;

            _arSession = FindAny<ARSession>();
            _arOcclusion = FindAny<AROcclusionManager>();

            // Try to find camera rig — look for OVRCameraRig or XROrigin
            _cameraRig = null;
            var xrOrigin = FindAny<Unity.XR.CoreUtils.XROrigin>();
            if (xrOrigin != null)
                _cameraRig = xrOrigin.gameObject;
            if (_cameraRig == null)
            {
                var ovrRig = FindComponentByTypeName("OVRCameraRig");
                if (ovrRig != null) _cameraRig = ovrRig.gameObject;
            }

            _depthCapture = FindAny<DepthCapture>();
            _volumeIntegrator = FindAny<VolumeIntegrator>();
            _chunkManager = FindAny<ChunkManager>();
            _textureProjector = FindAny<TextureProjector>();
            _roomScanner = FindAny<RoomScanner>();
            _cameraProvider = FindAny<PassthroughCameraProvider>();
            _pcaComponent = FindAny<PassthroughCameraAccess>();

            _depthCaptureWired = _depthCapture != null && AreFieldsAssigned(_depthCapture,
                "depthNormalCompute", "depthDilationCompute");
            _volumeWired = _volumeIntegrator != null && AreFieldsAssigned(_volumeIntegrator,
                "compute");
            _projectorWired = _textureProjector != null && AreFieldsAssigned(_textureProjector,
                "projectionCompute");
            _chunkMatWired = _chunkManager != null && AreFieldsAssigned(_chunkManager,
                "scanMeshMaterial");
        }

        // =================================================================
        //  GUI
        // =================================================================

        void OnGUI()
        {
            DrawHeader();
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            GUILayout.Space(4);

            DrawPrerequisites();
            DrawComponents();
            DrawShaderWiring();

            GUILayout.Space(12);
            DrawMasterButton();
            GUILayout.Space(8);

            EditorGUILayout.EndScrollView();
        }

        void DrawHeader()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            GUILayout.Label("Room Scan Setup", EditorStyles.boldLabel);
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Refresh", EditorStyles.toolbarButton, GUILayout.Width(60)))
                Refresh();
            EditorGUILayout.EndHorizontal();
        }

        // -- Prerequisites ------------------------------------------------

        void DrawPrerequisites()
        {
            BeginSection("PREREQUISITES");

            StatusRow("ARSession", _arSession != null);
            StatusRow("Camera Rig (OVRCameraRig / XROrigin)", _cameraRig != null);
            StatusRow("AROcclusionManager", _arOcclusion != null);

            if (_arSession == null)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add ARSession", GUILayout.Width(200)))
                    FixARSession();
                EditorGUILayout.EndHorizontal();
            }

            if (_cameraRig == null)
            {
                EditorGUILayout.HelpBox(
                    "Add a Camera Rig via  Meta > Tools > Building Blocks.\n" +
                    "The wizard will add AROcclusionManager to it automatically.",
                    MessageType.Info);
            }
            else if (_arOcclusion == null)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add AROcclusionManager", GUILayout.Width(200)))
                    FixAROcclusion();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        void FixARSession()
        {
            var go = FindByName("AR Session");
            if (go == null)
            {
                go = new GameObject("AR Session");
                Undo.RegisterCreatedObjectUndo(go, "Create AR Session");
            }

            if (go.GetComponent<ARSession>() == null)
                Undo.AddComponent<ARSession>(go);

            MarkDirty();
            Refresh();
        }

        void FixAROcclusion()
        {
            if (_cameraRig == null) return;

            // Find the camera — typically CenterEyeAnchor or Camera child
            Camera cam = _cameraRig.GetComponentInChildren<Camera>();
            if (cam == null)
            {
                Debug.LogWarning("[RoomScan Setup] No Camera found under camera rig");
                return;
            }

            GameObject target = cam.gameObject;

            // Need ARCameraManager as well for AROcclusionManager to work
            if (target.GetComponent<ARCameraManager>() == null)
                Undo.AddComponent<ARCameraManager>(target);

            if (target.GetComponent<AROcclusionManager>() == null)
                Undo.AddComponent<AROcclusionManager>(target);

            MarkDirty();
            Refresh();
        }

        // -- Components ---------------------------------------------------

        void DrawComponents()
        {
            BeginSection("ROOM SCAN COMPONENTS");

            StatusRow("DepthCapture", _depthCapture != null);
            StatusRow("VolumeIntegrator", _volumeIntegrator != null);
            StatusRow("ChunkManager", _chunkManager != null);
            StatusRow("TextureProjector", _textureProjector != null);
            StatusRow("RoomScanner", _roomScanner != null);
            StatusRow("PassthroughCameraProvider", _cameraProvider != null);
            StatusRow("PassthroughCameraAccess", _pcaComponent != null);

            bool anyMissing = _depthCapture == null || _volumeIntegrator == null ||
                              _chunkManager == null || _textureProjector == null ||
                              _roomScanner == null || _cameraProvider == null ||
                              _pcaComponent == null;

            if (anyMissing)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add All Missing", GUILayout.Width(160)))
                    FixComponents();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        void FixComponents()
        {
            // Create or find the root GameObject
            GameObject root = null;
            if (_roomScanner != null)
                root = _roomScanner.gameObject;
            else if (_depthCapture != null)
                root = _depthCapture.gameObject;

            if (root == null)
            {
                root = FindByName("RoomScan");
                if (root == null)
                {
                    root = new GameObject("RoomScan");
                    Undo.RegisterCreatedObjectUndo(root, "Create RoomScan");
                }
            }

            if (root.GetComponent<DepthCapture>() == null)
                Undo.AddComponent<DepthCapture>(root);
            if (root.GetComponent<VolumeIntegrator>() == null)
                Undo.AddComponent<VolumeIntegrator>(root);
            if (root.GetComponent<ChunkManager>() == null)
                Undo.AddComponent<ChunkManager>(root);
            if (root.GetComponent<TextureProjector>() == null)
                Undo.AddComponent<TextureProjector>(root);
            if (root.GetComponent<PassthroughCameraAccess>() == null)
                Undo.AddComponent<PassthroughCameraAccess>(root);
            if (root.GetComponent<PassthroughCameraProvider>() == null)
                Undo.AddComponent<PassthroughCameraProvider>(root);
            if (root.GetComponent<RoomScanner>() == null)
                Undo.AddComponent<RoomScanner>(root);

            MarkDirty();
            Refresh();
        }

        // -- Shader / Material Wiring ------------------------------------

        void DrawShaderWiring()
        {
            BeginSection("SHADER & MATERIAL WIRING");

            StatusRow("DepthCapture compute shaders", _depthCaptureWired);
            StatusRow("VolumeIntegrator compute shader", _volumeWired);
            StatusRow("TextureProjector compute shader", _projectorWired);
            StatusRow("ChunkManager scan material", _chunkMatWired);

            bool needsFix = !_depthCaptureWired || !_volumeWired ||
                            !_projectorWired || !_chunkMatWired;
            if (needsFix)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Wire All Shaders", GUILayout.Width(160)))
                    FixShaderWiring();
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        void FixShaderWiring()
        {
            // DepthCapture
            if (_depthCapture != null)
            {
                var so = new SerializedObject(_depthCapture);
                AssignCompute(so, "depthNormalCompute", PKG + "DepthNormals.compute");
                AssignCompute(so, "depthDilationCompute", PKG + "DepthDilation.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_depthCapture);
            }

            // VolumeIntegrator
            if (_volumeIntegrator != null)
            {
                var so = new SerializedObject(_volumeIntegrator);
                AssignCompute(so, "compute", PKG + "VolumeIntegration.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_volumeIntegrator);
            }

            // TextureProjector
            if (_textureProjector != null)
            {
                var so = new SerializedObject(_textureProjector);
                AssignCompute(so, "projectionCompute", PKG + "TextureProjection.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_textureProjector);
            }

            // ChunkManager — needs a Material
            if (_chunkManager != null)
            {
                var so = new SerializedObject(_chunkManager);
                var prop = so.FindProperty("scanMeshMaterial");
                if (prop != null && prop.objectReferenceValue == null)
                {
                    Material mat = GetOrCreateScanMaterial();
                    if (mat != null)
                        prop.objectReferenceValue = mat;
                }
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_chunkManager);
            }

            MarkDirty();
            Refresh();
        }

        static void AssignCompute(SerializedObject so, string fieldName, string assetPath)
        {
            var prop = so.FindProperty(fieldName);
            if (prop == null) return;
            if (prop.objectReferenceValue != null) return;

            var asset = AssetDatabase.LoadAssetAtPath<ComputeShader>(assetPath);
            if (asset != null)
                prop.objectReferenceValue = asset;
            else
                Debug.LogWarning($"[RoomScan Setup] Could not find {assetPath}");
        }

        static Material GetOrCreateScanMaterial()
        {
            const string pkgMatPath = "Packages/com.genesis.roomscan/Runtime/Materials/ScanMesh.mat";
            var pkgMat = AssetDatabase.LoadAssetAtPath<Material>(pkgMatPath);
            if (pkgMat != null) return pkgMat;

            // Fallback: create in project if package material not found
            const string matPath = "Assets/RoomScan/ScanMesh.mat";
            var existing = AssetDatabase.LoadAssetAtPath<Material>(matPath);
            if (existing != null) return existing;

            Shader shader = Shader.Find("Genesis/ScanMeshVertexColor");
            if (shader == null)
            {
                Debug.LogWarning("[RoomScan Setup] Shader 'Genesis/ScanMeshVertexColor' not found");
                return null;
            }

            if (!AssetDatabase.IsValidFolder("Assets/RoomScan"))
                AssetDatabase.CreateFolder("Assets", "RoomScan");

            var mat = new Material(shader) { name = "ScanMesh", enableInstancing = true };
            AssetDatabase.CreateAsset(mat, matPath);
            AssetDatabase.SaveAssets();
            return mat;
        }

        // -- Master Button ------------------------------------------------

        void DrawMasterButton()
        {
            var style = new GUIStyle(GUI.skin.button)
            {
                fontStyle = FontStyle.Bold,
                fontSize = 14,
                fixedHeight = 36
            };

            if (GUILayout.Button("\u2261  Setup Everything", style))
                SetupEverything();
        }

        void SetupEverything()
        {
            if (_cameraRig == null)
            {
                EditorUtility.DisplayDialog("Room Scan Setup",
                    "No Camera Rig found in the scene.\n\n" +
                    "Add a Camera Rig via Meta > Tools > Building Blocks first, " +
                    "then run this wizard again.",
                    "OK");
                return;
            }

            if (_arSession == null) FixARSession();
            if (_arOcclusion == null) FixAROcclusion();
            FixComponents();
            FixShaderWiring();

            // Wire RoomScanner references to sibling components
            Refresh();
            if (_roomScanner != null)
            {
                var so = new SerializedObject(_roomScanner);
                SetRef(so, "depthCapture", _depthCapture);
                SetRef(so, "volumeIntegrator", _volumeIntegrator);
                SetRef(so, "chunkManager", _chunkManager);
                SetRef(so, "textureProjector", _textureProjector);
                SetRef(so, "cameraProvider", _cameraProvider);
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_roomScanner);
            }

            MarkDirty();
            Refresh();

            Debug.Log("[RoomScan Setup] Scene setup complete.");
        }

        static void SetRef(SerializedObject so, string field, Object value)
        {
            var prop = so.FindProperty(field);
            if (prop != null && prop.objectReferenceValue == null && value != null)
                prop.objectReferenceValue = value;
        }

        // =================================================================
        //  GUI HELPERS
        // =================================================================

        void BeginSection(string title)
        {
            GUILayout.Space(6);
            var rect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none,
                GUILayout.ExpandWidth(true), GUILayout.Height(22));
            EditorGUI.DrawRect(rect, COL_SECT);
            var labelRect = new Rect(rect.x + 8, rect.y + 2, rect.width - 16, rect.height);
            var prev = GUI.color;
            GUI.color = Color.white;
            GUI.Label(labelRect, title, EditorStyles.boldLabel);
            GUI.color = prev;
        }

        static void EndSection() => GUILayout.Space(2);

        void StatusRow(string label, bool ok)
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.Space(12);

            string icon = ok ? "\u2713" : "\u2717";
            Color col = ok ? COL_OK : COL_MISS;
            string detail = ok ? "OK" : "Missing";

            var prev = GUI.color;
            GUI.color = col;
            GUILayout.Label(icon, EditorStyles.boldLabel, GUILayout.Width(18));
            GUI.color = prev;

            GUILayout.Label(label, GUILayout.ExpandWidth(true));

            prev = GUI.color;
            GUI.color = col;
            GUILayout.Label(detail, EditorStyles.miniLabel, GUILayout.Width(60));
            GUI.color = prev;

            EditorGUILayout.EndHorizontal();
        }

        // =================================================================
        //  UTILITY
        // =================================================================

        static T FindAny<T>() where T : Object =>
            Object.FindObjectsByType<T>(FindObjectsInactive.Include,
                FindObjectsSortMode.None).FirstOrDefault();

        static Component FindComponentByTypeName(string typeName)
        {
            foreach (var root in SceneRoots())
            {
                var found = root.GetComponentsInChildren<Component>(true)
                    .FirstOrDefault(c => c != null && c.GetType().Name == typeName);
                if (found != null) return found;
            }
            return null;
        }

        static bool AreFieldsAssigned(Object target, params string[] fieldNames)
        {
            var so = new SerializedObject(target);
            foreach (string name in fieldNames)
            {
                var prop = so.FindProperty(name);
                if (prop == null || prop.objectReferenceValue == null)
                    return false;
            }
            return true;
        }

        static GameObject FindByName(string exact)
        {
            foreach (var root in SceneRoots())
            {
                var t = DeepFind(root.transform,
                    tr => tr.name.Equals(exact, System.StringComparison.Ordinal));
                if (t != null) return t.gameObject;
            }
            return null;
        }

        static Transform DeepFind(Transform root, System.Func<Transform, bool> pred)
        {
            if (pred(root)) return root;
            for (int i = 0; i < root.childCount; i++)
            {
                var hit = DeepFind(root.GetChild(i), pred);
                if (hit != null) return hit;
            }
            return null;
        }

        static GameObject[] SceneRoots() =>
            UnityEngine.SceneManagement.SceneManager.GetActiveScene().GetRootGameObjects();

        static void MarkDirty() =>
            EditorSceneManager.MarkSceneDirty(
                UnityEngine.SceneManagement.SceneManager.GetActiveScene());
    }
}
