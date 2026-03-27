using System.IO;
using System.Linq;
using System.Xml.Linq;
using GaussianSplatting.Runtime;
using Genesis.RoomScan.GSplat;
using Genesis.RoomScan.UI;
using Meta.XR;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Rendering.Universal;
using UnityEngine.UIElements;
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
        MeshExtractor _meshExtractor;
        RoomScanner _roomScanner;
        PassthroughCameraProvider _cameraProvider;
        PassthroughCameraAccess _pcaComponent;
        CameraDebugOverlay _cameraDebug;
        DepthDebugOverlay _depthDebug;
        TriplanarCache _triplanarCache;
        RoomScanPersistence _persistence;
        KeyframeCollector _keyframeCollector;
        PointCloudExporter _pointCloudExporter;
        GSplatManager _gsplatManager;
        GaussianSplatRenderer _ugsRenderer;
        GSplatServerClient _gsplatServerClient;
        DebugMenuController _debugMenu;
        RoomScanInputHandler _inputHandler;
        RoomAnchorManager _roomAnchor;
        EventSystem _eventSystem;
        OVRInputModule _ovrInputModule;
        VRDocumentRaycaster _vrRaycaster;
        ControllerRayDriver _rayDriver;
        PanelInputConfiguration _panelInputConfig;

        bool _depthCaptureWired, _volumeWired, _meshMatWired, _triplanarWired, _computeShaderWired;
        bool _refinedShaderWired, _atlasBakeComputeWired;
        bool _ugsRendererWired;
        bool _ugsRenderFeatureAdded;
        bool _deferredRendering;
        bool _boundarylessManifest;
        bool _cleartextAllowed;
        bool _insecureHttpAllowed;

        // Style
        static readonly Color COL_OK   = new(0.25f, 0.82f, 0.35f);
        static readonly Color COL_WARN = new(0.95f, 0.78f, 0.15f);
        static readonly Color COL_MISS = new(0.92f, 0.28f, 0.25f);
        static readonly Color COL_INFO = new(0.45f, 0.72f, 0.95f);
        static readonly Color COL_SECT = new(0.18f, 0.18f, 0.22f);

        const string PKG = "Packages/com.genesis.roomscan/Runtime/Shaders/";
        const string GSPLAT_PKG = "Packages/com.genesis.roomscan/Runtime/GSplat/Shaders/";

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
            _meshExtractor = FindAny<MeshExtractor>();
            _roomScanner = FindAny<RoomScanner>();
            _cameraProvider = FindAny<PassthroughCameraProvider>();
            _pcaComponent = FindAny<PassthroughCameraAccess>();
            _cameraDebug = FindAny<CameraDebugOverlay>();
            _depthDebug = FindAny<DepthDebugOverlay>();
            _triplanarCache = FindAny<TriplanarCache>();
            _persistence = FindAny<RoomScanPersistence>();
            _keyframeCollector = FindAny<KeyframeCollector>();
            _pointCloudExporter = FindAny<PointCloudExporter>();
            _gsplatManager = FindAny<GSplatManager>();
            _ugsRenderer = FindAny<GaussianSplatRenderer>();
            _gsplatServerClient = FindAny<GSplatServerClient>();
            _debugMenu = FindAny<DebugMenuController>();
            _inputHandler = FindAny<RoomScanInputHandler>();
            _roomAnchor = FindAny<RoomAnchorManager>();
            _eventSystem = FindAny<EventSystem>();
            _ovrInputModule = FindAny<OVRInputModule>();
            _vrRaycaster = FindAny<VRDocumentRaycaster>();
            _rayDriver = FindAny<ControllerRayDriver>();
            _panelInputConfig = FindAny<PanelInputConfiguration>();

            _depthCaptureWired = _depthCapture != null && AreFieldsAssigned(_depthCapture,
                "depthNormalCompute", "depthDilationCompute", "bilateralFilterCompute");
            _volumeWired = _volumeIntegrator != null && AreFieldsAssigned(_volumeIntegrator,
                "compute");
            _meshMatWired = _meshExtractor != null && AreFieldsAssigned(_meshExtractor,
                "scanMeshMaterial");
            _triplanarWired = _triplanarCache != null && AreFieldsAssigned(_triplanarCache,
                "bakeCompute");
            _computeShaderWired = _meshExtractor != null && AreFieldsAssigned(_meshExtractor,
                "surfaceNetsCompute");
            _refinedShaderWired = _roomScanner != null && AreFieldsAssigned(_roomScanner,
                "refinedMeshShader");
            _atlasBakeComputeWired = _roomScanner != null && AreFieldsAssigned(_roomScanner,
                "atlasBakeCompute");
            _ugsRendererWired = _ugsRenderer != null && AreFieldsAssigned(_ugsRenderer,
                "m_ShaderSplats", "m_ShaderComposite", "m_ShaderDebugPoints", "m_ShaderDebugBoxes", "m_CSSplatUtilities");
            _ugsRenderFeatureAdded = HasUGSRenderFeature();
            _deferredRendering = IsDeferredRendering();

            _boundarylessManifest = ManifestHasBoundaryless();
            _cleartextAllowed = ManifestHasCleartextTraffic();
            _insecureHttpAllowed = PlayerSettings.insecureHttpOption != InsecureHttpOption.NotAllowed;

            RefreshServerUrl();
        }

        bool _serverUrlCurrent;
        string _detectedLanIp;
        string _currentServerUrl;

        void RefreshServerUrl()
        {
            _serverUrlCurrent = false;
            _detectedLanIp = GetLanIp();
            _currentServerUrl = null;

            if (_gsplatServerClient == null || string.IsNullOrEmpty(_detectedLanIp)) return;

            var so = new SerializedObject(_gsplatServerClient);
            var prop = so.FindProperty("serverUrl");
            if (prop == null) return;

            _currentServerUrl = prop.stringValue;
            string expected = $"http://{_detectedLanIp}:8420";
            _serverUrlCurrent = _currentServerUrl == expected;
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
            DrawProjectSettings();
            DrawComponents();
            DrawShaderWiring();
            DrawNativePlugins();

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

        // -- Project Settings ---------------------------------------------

        const string MANIFEST_PATH = "Assets/Plugins/Android/AndroidManifest.xml";
        const string BOUNDARYLESS_FEATURE = "com.oculus.feature.BOUNDARYLESS_APP";

        void DrawProjectSettings()
        {
            BeginSection("PROJECT SETTINGS");

            StatusRow("AndroidManifest boundaryless entry", _boundarylessManifest);
            StatusRow("AndroidManifest cleartext HTTP (LAN)", _cleartextAllowed);
            StatusRow("Player Settings: Allow HTTP", _insecureHttpAllowed);

            if (!_boundarylessManifest)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Add Boundaryless Manifest", GUILayout.Width(200)))
                {
                    FixBoundarylessManifest();
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (!_cleartextAllowed)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Allow Cleartext HTTP", GUILayout.Width(200)))
                {
                    FixCleartextTraffic();
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            if (!_insecureHttpAllowed)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Allow HTTP in Player Settings", GUILayout.Width(200)))
                {
                    PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
                    Debug.Log("[RoomScan Setup] Set Player Settings > insecureHttpOption to AlwaysAllowed");
                    Refresh();
                }
                EditorGUILayout.EndHorizontal();
            }

            EndSection();
        }

        static bool ManifestHasBoundaryless()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);
            if (!File.Exists(fullPath)) return false;

            try
            {
                var doc = XDocument.Load(fullPath);
                XNamespace android = "http://schemas.android.com/apk/res/android";
                return doc.Root?.Elements("uses-feature")
                    .Any(e => e.Attribute(android + "name")?.Value == BOUNDARYLESS_FEATURE) ?? false;
            }
            catch
            {
                return false;
            }
        }

        static void FixBoundarylessManifest()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);

            if (!File.Exists(fullPath))
            {
                EditorUtility.DisplayDialog("Room Scan Setup",
                    "AndroidManifest.xml not found at:\n" + MANIFEST_PATH + "\n\n" +
                    "Build the project once or create a custom manifest first.",
                    "OK");
                return;
            }

            try
            {
                var doc = XDocument.Load(fullPath);
                XNamespace android = "http://schemas.android.com/apk/res/android";

                bool exists = doc.Root?.Elements("uses-feature")
                    .Any(e => e.Attribute(android + "name")?.Value == BOUNDARYLESS_FEATURE) ?? false;
                if (exists) return;

                var element = new XElement("uses-feature",
                    new XAttribute(android + "name", BOUNDARYLESS_FEATURE),
                    new XAttribute(android + "required", "true"));

                doc.Root?.Add(element);
                doc.Save(fullPath);

                AssetDatabase.Refresh();
                Debug.Log($"[RoomScan Setup] Added {BOUNDARYLESS_FEATURE} to AndroidManifest.xml");
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] Failed to update manifest: {ex.Message}");
            }
        }

        // -- Cleartext HTTP -----------------------------------------------

        const string ANDROIDLIB_DIR = "Assets/Plugins/Android/NetworkSecurityConfig.androidlib";
        const string ANDROIDLIB_NSC = ANDROIDLIB_DIR + "/res/xml/network_security_config.xml";

        static bool ManifestHasCleartextTraffic()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);
            if (!File.Exists(fullPath)) return false;

            try
            {
                var doc = XDocument.Load(fullPath);
                XNamespace android = "http://schemas.android.com/apk/res/android";
                var app = doc.Root?.Element("application");
                if (app == null) return false;

                string val = app.Attribute(android + "usesCleartextTraffic")?.Value;
                if (val != "true") return false;

                string nscFull = Path.Combine(Application.dataPath, "..", ANDROIDLIB_NSC);
                return File.Exists(nscFull);
            }
            catch
            {
                return false;
            }
        }

        static void FixCleartextTraffic()
        {
            string manifestFull = Path.Combine(Application.dataPath, "..", MANIFEST_PATH);

            if (!File.Exists(manifestFull))
            {
                EditorUtility.DisplayDialog("Room Scan Setup",
                    "AndroidManifest.xml not found at:\n" + MANIFEST_PATH + "\n\n" +
                    "Build the project once or create a custom manifest first.",
                    "OK");
                return;
            }

            try
            {
                var doc = XDocument.Load(manifestFull);
                XNamespace android = "http://schemas.android.com/apk/res/android";
                var app = doc.Root?.Element("application");
                if (app == null) return;

                // android:usesCleartextTraffic="true"
                var cleartext = app.Attribute(android + "usesCleartextTraffic");
                if (cleartext == null)
                    app.Add(new XAttribute(android + "usesCleartextTraffic", "true"));
                else
                    cleartext.Value = "true";

                // android:networkSecurityConfig="@xml/network_security_config"
                var nscAttr = app.Attribute(android + "networkSecurityConfig");
                if (nscAttr == null)
                    app.Add(new XAttribute(android + "networkSecurityConfig", "@xml/network_security_config"));
                else
                    nscAttr.Value = "@xml/network_security_config";

                doc.Save(manifestFull);
                Debug.Log("[RoomScan Setup] Added cleartext HTTP attributes to AndroidManifest.xml");

                // Unity 6+ requires Android resources in an .androidlib, not raw res/
                string libRoot = Path.Combine(Application.dataPath, "..", ANDROIDLIB_DIR);
                string nscDir = Path.Combine(libRoot, "res", "xml");
                if (!Directory.Exists(nscDir))
                    Directory.CreateDirectory(nscDir);

                string nscFull = Path.Combine(nscDir, "network_security_config.xml");
                if (!File.Exists(nscFull))
                {
                    const string nscContent =
                        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
                        "<network-security-config>\n" +
                        "    <base-config cleartextTrafficPermitted=\"true\">\n" +
                        "        <trust-anchors>\n" +
                        "            <certificates src=\"system\" />\n" +
                        "        </trust-anchors>\n" +
                        "    </base-config>\n" +
                        "</network-security-config>\n";
                    File.WriteAllText(nscFull, nscContent);
                }

                // AndroidManifest.xml for the library module
                string libManifest = Path.Combine(libRoot, "AndroidManifest.xml");
                if (!File.Exists(libManifest))
                {
                    const string libManifestContent =
                        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
                        "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\"\n" +
                        "    package=\"com.genesis.roomscan.netsecconfig\">\n" +
                        "</manifest>\n";
                    File.WriteAllText(libManifest, libManifestContent);
                }

                // project.properties marks it as a library
                string projProps = Path.Combine(libRoot, "project.properties");
                if (!File.Exists(projProps))
                    File.WriteAllText(projProps, "android.library=true\n");

                Debug.Log($"[RoomScan Setup] Created {ANDROIDLIB_DIR} with network_security_config.xml");
                AssetDatabase.Refresh();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] Failed to enable cleartext traffic: {ex.Message}");
            }
        }

        // -- Components ---------------------------------------------------

        void DrawComponents()
        {
            BeginSection("ROOM SCAN COMPONENTS");

            StatusRow("DepthCapture", _depthCapture != null);
            StatusRow("VolumeIntegrator", _volumeIntegrator != null);
            StatusRow("MeshExtractor", _meshExtractor != null);
            StatusRow("RoomScanner", _roomScanner != null);
            StatusRow("RoomAnchorManager (MRUK + SpatialAnchor)", _roomAnchor != null);

            var ovrConfig = OVRProjectConfig.CachedProjectConfig;
            bool anchorSupportOk = ovrConfig != null
                && ovrConfig.anchorSupport != OVRProjectConfig.AnchorSupport.Disabled;
            StatusRow("OVRProjectConfig anchor support", anchorSupportOk);
            if (!anchorSupportOk && ovrConfig != null)
            {
                if (GUILayout.Button("Fix: Enable Spatial Anchor Support"))
                {
                    ovrConfig.anchorSupport = OVRProjectConfig.AnchorSupport.Enabled;
                    OVRProjectConfig.CommitProjectConfig(ovrConfig);
                }
            }
            StatusRow("PassthroughCameraProvider", _cameraProvider != null);
            StatusRow("PassthroughCameraAccess", _pcaComponent != null);
            StatusRow("CameraDebugOverlay", _cameraDebug != null);
            StatusRow("DepthDebugOverlay", _depthDebug != null);
            StatusRow("TriplanarCache", _triplanarCache != null);
            StatusRow("RoomScanPersistence", _persistence != null);
            StatusRow("KeyframeCollector (GS export)", _keyframeCollector != null);
            StatusRow("PointCloudExporter (GS export)", _pointCloudExporter != null);
            StatusRow("GSplatManager (PLY loader)", _gsplatManager != null);
            StatusRow("GaussianSplatRenderer (UGS)", _ugsRenderer != null);
            StatusRow("GSplatServerClient (PC training)", _gsplatServerClient != null);
            if (_gsplatServerClient != null)
            {
                if (_serverUrlCurrent)
                {
                    StatusRow($"  Server URL → {_detectedLanIp}:8420", true);
                }
                else
                {
                    string stale = string.IsNullOrEmpty(_currentServerUrl) ? "(empty)" : _currentServerUrl;
                    StatusRow($"  Server URL STALE: {stale} (LAN: {_detectedLanIp})", false);
                    if (GUILayout.Button("Fix Server URL"))
                    {
                        ConfigureServerUrl();
                        Refresh();
                    }
                }
            }
            StatusRow("DebugMenuController (HUD)", _debugMenu != null);
            StatusRow("RoomScanInputHandler (bindings)", _inputHandler != null);
            StatusRow("EventSystem + OVRInputModule", _eventSystem != null && _ovrInputModule != null);
            StatusRow("VRDocumentRaycaster (UI pointer)", _vrRaycaster != null);
            StatusRow("ControllerRayDriver (laser)", _rayDriver != null);
            StatusRow("PanelInputConfiguration", _panelInputConfig != null);

            bool anyMissing = _depthCapture == null || _volumeIntegrator == null ||
                              _meshExtractor == null ||
                              _roomScanner == null || _roomAnchor == null || _cameraProvider == null ||
                              _pcaComponent == null || _cameraDebug == null ||
                              _triplanarCache == null ||
                              _persistence == null || _keyframeCollector == null ||
                              _pointCloudExporter == null ||
                              _gsplatManager == null || _ugsRenderer == null ||
                              _gsplatServerClient == null ||
                              _debugMenu == null || _inputHandler == null ||
                              _eventSystem == null || _ovrInputModule == null ||
                              _vrRaycaster == null || _rayDriver == null ||
                              _panelInputConfig == null;

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

            // PassthroughCameraAccess isn't pulled in by RequireComponent
            if (root.GetComponent<PassthroughCameraAccess>() == null)
                Undo.AddComponent<PassthroughCameraAccess>(root);

            // Adding RoomScanner auto-adds all [RequireComponent] siblings:
            // DepthCapture, VolumeIntegrator, MeshExtractor,
            // PassthroughCameraProvider, TriplanarCache, RoomScanPersistence,
            // KeyframeCollector, PointCloudExporter, GSplatManager,
            // GSplatServerClient, RoomAnchorManager (GaussianSplatRenderer created by GSplatManager on child GO)
            if (root.GetComponent<RoomScanner>() == null)
                Undo.AddComponent<RoomScanner>(root);

            // GaussianSplatRenderer lives on a child GO so its transform can be
            // set independently for room-anchor relocation. GSplatManager.Awake()
            // creates this at runtime, but we also need it in the editor for shader wiring.
            var splatChild = root.transform.Find("SplatRenderer");
            if (splatChild == null)
            {
                var go = new GameObject("SplatRenderer");
                go.transform.SetParent(root.transform, false);
                Undo.RegisterCreatedObjectUndo(go, "Create SplatRenderer");
                splatChild = go.transform;
            }
            if (splatChild.GetComponent<GaussianSplatRenderer>() == null)
                Undo.AddComponent<GaussianSplatRenderer>(splatChild.gameObject);

            // Optional components not covered by RequireComponent
            if (root.GetComponent<RoomScanInputHandler>() == null)
                Undo.AddComponent<RoomScanInputHandler>(root);

            // Debug overlays — disabled by default
            if (root.GetComponent<CameraDebugOverlay>() == null)
            {
                var c = Undo.AddComponent<CameraDebugOverlay>(root);
                c.enabled = false;
            }
            if (root.GetComponent<DepthDebugOverlay>() == null)
            {
                var c = Undo.AddComponent<DepthDebugOverlay>(root);
                c.enabled = false;
            }

            // DebugMenu lives on a child (needs UIDocument)
            if (FindAny<DebugMenuController>() == null)
            {
                var debugGo = new GameObject("DebugMenu");
                debugGo.transform.SetParent(root.transform);
                Undo.RegisterCreatedObjectUndo(debugGo, "Create DebugMenu");

                Undo.AddComponent<UIDocument>(debugGo);
                Undo.AddComponent<DebugMenuController>(debugGo);
            }

            // Always ensure UIDocument has its assets assigned
            EnsureDebugMenuAssets();

            // Auto-configure GS training server URL with this PC's LAN IP
            ConfigureServerUrl();

            // EventSystem + VR controller UI input pipeline
            EnsureVRInputInfrastructure();

            MarkDirty();
            Refresh();
        }

        /// <summary>
        /// Sets up EventSystem, OVRInputModule, PanelInputConfiguration,
        /// VRDocumentRaycaster, and ControllerRayDriver so that VR controller
        /// rays can interact with world-space UI Toolkit panels.
        /// </summary>
        void EnsureVRInputInfrastructure()
        {
            // EventSystem
            var es = FindAny<EventSystem>();
            if (es == null)
            {
                var esGo = new GameObject("EventSystem");
                Undo.RegisterCreatedObjectUndo(esGo, "Create EventSystem");
                es = Undo.AddComponent<EventSystem>(esGo);
            }

            // OVRInputModule (replaces StandaloneInputModule for VR)
            if (es.GetComponent<OVRInputModule>() == null)
            {
                // Remove StandaloneInputModule if present — only one input module should be active
                var standalone = es.GetComponent<StandaloneInputModule>();
                if (standalone != null) Undo.DestroyObjectImmediate(standalone);

                Undo.AddComponent<OVRInputModule>(es.gameObject);
            }

            // PanelInputConfiguration (auto-creates PanelEventHandler per panel)
            if (es.GetComponent<PanelInputConfiguration>() == null)
            {
                var pic = Undo.AddComponent<PanelInputConfiguration>(es.gameObject);
                var so = new SerializedObject(pic);
                SetBool(so, "m_DefaultEventCameraIsMainCamera", true);
                SetBool(so, "m_AutoCreatePanelComponents", true);
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(pic);
            }

            // VRDocumentRaycaster (overrides GetWorldRay for controller rays)
            if (es.GetComponent<VRDocumentRaycaster>() == null)
                Undo.AddComponent<VRDocumentRaycaster>(es.gameObject);

            // ControllerRayDriver (laser visual + auto-picks active controller)
            if (es.GetComponent<ControllerRayDriver>() == null)
                Undo.AddComponent<ControllerRayDriver>(es.gameObject);
        }

        static void SetBool(SerializedObject so, string fieldName, bool value)
        {
            var prop = so.FindProperty(fieldName);
            if (prop != null) prop.boolValue = value;
        }

        // -- Shader / Material Wiring ------------------------------------

        void DrawShaderWiring()
        {
            BeginSection("SHADER & MATERIAL WIRING");

            StatusRow("DepthCapture compute shaders", _depthCaptureWired);
            StatusRow("VolumeIntegrator compute shader", _volumeWired);
            StatusRow("MeshExtractor scan material", _meshMatWired);
            StatusRow("TriplanarCache bake compute", _triplanarWired);
            StatusRow("SurfaceNetsExtract compute shader", _computeShaderWired);
            StatusRow("RefinedMesh shader (texture refine)", _refinedShaderWired);
            StatusRow("AtlasBakeCompute (GPU bake)", _atlasBakeComputeWired);
            StatusRow("UGS renderer shaders + compute", _ugsRendererWired);
            StatusRow("UGS RenderFeature on URP Renderer", _ugsRenderFeatureAdded);
            StatusRow("URP Deferred Rendering (req. by UGS)", _deferredRendering);

            bool needsFix = !_depthCaptureWired || !_volumeWired ||
                            !_meshMatWired || !_triplanarWired ||
                            !_computeShaderWired || !_refinedShaderWired ||
                            !_atlasBakeComputeWired ||
                            !_ugsRendererWired || !_ugsRenderFeatureAdded ||
                            !_deferredRendering;
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
                AssignCompute(so, "bilateralFilterCompute", PKG + "BilateralDepthFilter.compute");
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

            // TriplanarCache
            if (_triplanarCache != null)
            {
                var so = new SerializedObject(_triplanarCache);
                AssignCompute(so, "bakeCompute", PKG + "TriplanarBake.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_triplanarCache);
            }

            // DepthDebugOverlay
            if (_depthDebug != null)
            {
                var so = new SerializedObject(_depthDebug);
                AssignAsset<Shader>(so, "depthVisualizeShader", PKG + "DepthVisualize.shader");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_depthDebug);
            }

            // MeshExtractor — needs a Material + compute shader
            if (_meshExtractor != null)
            {
                var so = new SerializedObject(_meshExtractor);
                var prop = so.FindProperty("scanMeshMaterial");
                if (prop != null && prop.objectReferenceValue == null)
                {
                    Material mat = GetOrCreateScanMaterial();
                    if (mat != null)
                        prop.objectReferenceValue = mat;
                }
                AssignCompute(so, "surfaceNetsCompute", PKG + "SurfaceNetsExtract.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_meshExtractor);
            }

            // RoomScanner — refined mesh + atlas bake shaders
            if (_roomScanner != null)
            {
                var so = new SerializedObject(_roomScanner);
                AssignAsset<Shader>(so, "refinedMeshShader", PKG + "RefinedMesh.shader");
                AssignAsset<ComputeShader>(so, "atlasBakeCompute", PKG + "AtlasBakeCompute.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_roomScanner);
            }

            // UGS GaussianSplatRenderer — shaders + compute
            if (_ugsRenderer != null)
            {
                const string UGS_PKG = "Packages/org.nesnausk.gaussian-splatting/Shaders/";
                var so = new SerializedObject(_ugsRenderer);
                AssignAsset<Shader>(so, "m_ShaderSplats", UGS_PKG + "RenderGaussianSplats.shader");
                AssignAsset<Shader>(so, "m_ShaderComposite", UGS_PKG + "GaussianComposite.shader");
                AssignAsset<Shader>(so, "m_ShaderDebugPoints", UGS_PKG + "GaussianDebugRenderPoints.shader");
                AssignAsset<Shader>(so, "m_ShaderDebugBoxes", UGS_PKG + "GaussianDebugRenderBoxes.shader");
                AssignCompute(so, "m_CSSplatUtilities", UGS_PKG + "SplatUtilities.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(_ugsRenderer);
            }

            if (!_ugsRenderFeatureAdded)
                AddUGSRenderFeature();

            if (!_deferredRendering)
                SetDeferredRendering();

            MarkDirty();
            Refresh();
        }

        static void AssignCompute(SerializedObject so, string fieldName, string assetPath)
        {
            AssignAsset<ComputeShader>(so, fieldName, assetPath);
        }

        static void AssignAsset<T>(SerializedObject so, string fieldName, string assetPath) where T : Object
        {
            var prop = so.FindProperty(fieldName);
            if (prop == null) return;
            if (prop.objectReferenceValue != null) return;

            var asset = AssetDatabase.LoadAssetAtPath<T>(assetPath);
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

        static Material GetOrCreateSplatMaterial()
        {
            const string pkgMatPath = "Packages/com.genesis.roomscan/Runtime/GSplat/Materials/SplatRender.mat";
            var pkgMat = AssetDatabase.LoadAssetAtPath<Material>(pkgMatPath);
            if (pkgMat != null) return pkgMat;

            Shader shader = Shader.Find("Genesis/SplatRender");
            if (shader == null)
            {
                Debug.LogWarning("[RoomScan Setup] Shader 'Genesis/SplatRender' not found");
                return null;
            }

            if (!AssetDatabase.IsValidFolder("Assets/RoomScan"))
                AssetDatabase.CreateFolder("Assets", "RoomScan");

            var mat = new Material(shader) { name = "SplatRender" };
            const string matPath = "Assets/RoomScan/SplatRender.mat";
            AssetDatabase.CreateAsset(mat, matPath);
            AssetDatabase.SaveAssets();
            return mat;
        }

        // -- URP Renderer Feature -----------------------------------------

        static UniversalRendererData FindActiveRendererData()
        {
            var pipeline = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline
                as UniversalRenderPipelineAsset;
            if (pipeline == null) return null;

            var so = new SerializedObject(pipeline);
            var list = so.FindProperty("m_RendererDataList");
            if (list == null || list.arraySize == 0) return null;

            return list.GetArrayElementAtIndex(0).objectReferenceValue
                as UniversalRendererData;
        }

        static bool HasUGSRenderFeature()
        {
            var rd = FindActiveRendererData();
            if (rd == null) return false;
            foreach (var f in rd.rendererFeatures)
            {
                if (f != null && f.GetType().Name == "GaussianSplatURPFeature") return true;
            }
            return false;
        }

        const int RENDERING_MODE_DEFERRED = 1;

        static bool IsDeferredRendering()
        {
            var rd = FindActiveRendererData();
            if (rd == null) return false;
            var so = new SerializedObject(rd);
            var prop = so.FindProperty("m_RenderingMode");
            return prop != null && prop.intValue == RENDERING_MODE_DEFERRED;
        }

        static void SetDeferredRendering()
        {
            var rd = FindActiveRendererData();
            if (rd == null)
            {
                Debug.LogWarning("[RoomScan Setup] No active URP RendererData found");
                return;
            }

            var so = new SerializedObject(rd);
            var prop = so.FindProperty("m_RenderingMode");
            if (prop == null) return;

            if (prop.intValue != RENDERING_MODE_DEFERRED)
            {
                prop.intValue = RENDERING_MODE_DEFERRED;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(rd);
                AssetDatabase.SaveAssets();
                Debug.Log("[RoomScan Setup] Switched URP Renderer to Deferred rendering (required by UGS)");
            }
        }

        static void AddUGSRenderFeature()
        {
            var rd = FindActiveRendererData();
            if (rd == null)
            {
                Debug.LogWarning("[RoomScan Setup] No active URP RendererData found");
                return;
            }

            // Find the GaussianSplatURPFeature type from the UGS assembly
            System.Type featureType = null;
            foreach (var asm in System.AppDomain.CurrentDomain.GetAssemblies())
            {
                featureType = asm.GetType("GaussianSplatting.Runtime.GaussianSplatURPFeature");
                if (featureType != null) break;
            }
            if (featureType == null)
            {
                Debug.LogWarning("[RoomScan Setup] GaussianSplatURPFeature type not found — " +
                    "ensure UGS package has GS_ENABLE_URP defined and Unity 6+");
                return;
            }

            var feature = (ScriptableRendererFeature)CreateInstance(featureType);
            feature.name = "Gaussian Splat Renderer";
            feature.SetActive(true);

            Undo.RecordObject(rd, "Add UGS Render Feature");
            AssetDatabase.AddObjectToAsset(feature, rd);

            var so = new SerializedObject(rd);
            var features = so.FindProperty("m_RendererFeatures");
            features.arraySize++;
            features.GetArrayElementAtIndex(features.arraySize - 1).objectReferenceValue = feature;
            so.ApplyModifiedProperties();

            EditorUtility.SetDirty(rd);
            AssetDatabase.SaveAssets();
            Debug.Log("[RoomScan Setup] Added GaussianSplatURPFeature to URP Renderer");
        }

        // -- Native Plugins -----------------------------------------------

        bool _xatlasAndroid, _xatlasMacOS;

        void RefreshNativePlugins()
        {
            string pkgRoot = "Packages/com.genesis.roomscan/Runtime";
            _xatlasAndroid = System.IO.File.Exists(
                Path.GetFullPath(Path.Combine(pkgRoot, "Plugins/Android/libxatlas.so")));
            _xatlasMacOS = System.IO.File.Exists(
                Path.GetFullPath(Path.Combine(pkgRoot, "Plugins/macOS/libxatlas.bundle")));
        }

        void DrawNativePlugins()
        {
            RefreshNativePlugins();
            BeginSection("NATIVE PLUGINS");
            StatusRow("xatlas (Android ARM64)", _xatlasAndroid);
            StatusRow("xatlas (macOS Editor)", _xatlasMacOS);

            if (!_xatlasAndroid || !_xatlasMacOS)
            {
                GUILayout.Space(2);
                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                if (GUILayout.Button("Build xatlas Plugin", GUILayout.Width(200)))
                    BuildXAtlasPlugin();
                EditorGUILayout.EndHorizontal();
            }
            EndSection();
        }

        static void BuildXAtlasPlugin()
        {
            string pkgRoot = Path.GetFullPath("Packages/com.genesis.roomscan/Runtime");
            string srcDir = Path.Combine(pkgRoot, "Native/xatlas");
            string srcApi = Path.Combine(srcDir, "xatlas_c_api.cpp");
            string srcImpl = Path.Combine(srcDir, "xatlas.cpp");
            string meshoptDir = Path.Combine(pkgRoot, "Native/meshoptimizer");
            string srcSimplifier = Path.Combine(meshoptDir, "simplifier.cpp");

            if (!System.IO.File.Exists(srcApi) || !System.IO.File.Exists(srcImpl))
            {
                EditorUtility.DisplayDialog("Build xatlas",
                    $"Source files not found in:\n{srcDir}\n\nExpected xatlas.cpp and xatlas_c_api.cpp",
                    "OK");
                return;
            }

            bool hasMeshopt = System.IO.File.Exists(srcSimplifier);
            if (!hasMeshopt)
                Debug.LogWarning("[RoomScan Setup] meshoptimizer sources not found — building without mesh simplification");

            string meshoptSrc = hasMeshopt ? $" \"{srcSimplifier}\"" : "";
            string meshoptInc = hasMeshopt ? $" -I\"{meshoptDir}\"" : "";

            var builds = new System.Collections.Generic.List<(string label, string exe, string args, string outAssetPath)>();

            // macOS
            {
                string outDir = Path.Combine(pkgRoot, "Plugins/macOS");
                Directory.CreateDirectory(outDir);
                string outPath = Path.Combine(outDir, "libxatlas.bundle");
                string bArgs = $"-shared -O2 -fPIC -std=c++11 -fvisibility=hidden{meshoptInc} " +
                               $"-o \"{outPath}\" \"{srcApi}\" \"{srcImpl}\"{meshoptSrc}";
                builds.Add(("macOS xatlas", "clang++", bArgs,
                    "Packages/com.genesis.roomscan/Runtime/Plugins/macOS/libxatlas.bundle"));
            }

            // Android ARM64
#if UNITY_EDITOR_OSX || UNITY_EDITOR_LINUX || UNITY_EDITOR_WIN
            {
                string ndkClang = FindNdkClang();
                if (ndkClang != null)
                {
                    string outDir = Path.Combine(pkgRoot, "Plugins/Android");
                    Directory.CreateDirectory(outDir);
                    string outPath = Path.Combine(outDir, "libxatlas.so");
                    string bArgs = $"-shared -O2 -fPIC -std=c++11 -fvisibility=hidden{meshoptInc} " +
                                   $"-o \"{outPath}\" \"{srcApi}\" \"{srcImpl}\"{meshoptSrc}";
                    builds.Add(("Android xatlas", ndkClang, bArgs,
                        "Packages/com.genesis.roomscan/Runtime/Plugins/Android/libxatlas.so"));
                }
            }
#endif

            if (builds.Count == 0)
            {
                Debug.LogError("[RoomScan Setup] No build targets available");
                return;
            }

            StartAsyncBuilds(builds);
        }

        static string FindNdkClang()
        {
            string ndkPath = null;
            try
            {
                ndkPath = UnityEditor.Android.AndroidExternalToolsSettings.ndkRootPath;
            }
            catch
            {
                Debug.LogWarning("[RoomScan Setup] Android NDK path not configured. Skipping Android build.");
                return null;
            }

            if (string.IsNullOrEmpty(ndkPath) || !Directory.Exists(ndkPath))
            {
                Debug.LogWarning($"[RoomScan Setup] NDK not found at: {ndkPath}");
                return null;
            }

            string prebuilt = Path.Combine(ndkPath, "toolchains/llvm/prebuilt");
            if (!Directory.Exists(prebuilt)) return null;

            string[] hosts = Directory.GetDirectories(prebuilt);
            if (hosts.Length == 0) return null;

            string clangpp = Path.Combine(hosts[0], "bin/aarch64-linux-android31-clang++");
            return System.IO.File.Exists(clangpp) ? clangpp : null;
        }

        // Async build state
        static System.Collections.Generic.List<(string label, System.Diagnostics.Process proc, string outAssetPath,
            System.Text.StringBuilder stdout, System.Text.StringBuilder stderr)> _activeBuilds;
        static int _totalBuilds;

        static void StartAsyncBuilds(
            System.Collections.Generic.List<(string label, string exe, string args, string outAssetPath)> builds)
        {
            _activeBuilds = new();
            _totalBuilds = builds.Count;

            foreach (var (label, exe, args, outAssetPath) in builds)
            {
                try
                {
                    var psi = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = exe,
                        Arguments = args,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    var proc = System.Diagnostics.Process.Start(psi);
                    var stdoutBuf = new System.Text.StringBuilder();
                    var stderrBuf = new System.Text.StringBuilder();
                    proc.OutputDataReceived += (_, e) => { if (e.Data != null) stdoutBuf.AppendLine(e.Data); };
                    proc.ErrorDataReceived += (_, e) => { if (e.Data != null) stderrBuf.AppendLine(e.Data); };
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    _activeBuilds.Add((label, proc, outAssetPath, stdoutBuf, stderrBuf));
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[RoomScan Setup] Failed to start {label}: {e.Message}");
                }
            }

            if (_activeBuilds.Count == 0)
            {
                Debug.LogError("[RoomScan Setup] No builds started");
                return;
            }

            EditorApplication.update += PollXAtlasBuilds;
            EditorUtility.DisplayProgressBar("Building xatlas", "Compiling native plugins...", 0f);
        }

        static void PollXAtlasBuilds()
        {
            if (_activeBuilds == null) return;

            int done = 0;
            foreach (var (label, proc, _, _, _) in _activeBuilds)
                if (proc.HasExited) done++;

            float progress = (float)done / _totalBuilds;
            string building = done < _totalBuilds
                ? $"Compiling... ({done}/{_totalBuilds} done)"
                : "Finishing...";
            EditorUtility.DisplayProgressBar("Building xatlas", building, progress);

            if (done < _totalBuilds) return;

            // All done
            EditorApplication.update -= PollXAtlasBuilds;
            EditorUtility.ClearProgressBar();

            bool allOk = true;
            var results = new System.Text.StringBuilder();

            foreach (var (label, proc, outAssetPath, _, stderrBuf) in _activeBuilds)
            {
                string stderr = stderrBuf.ToString();
                bool ok = proc.ExitCode == 0;
                allOk &= ok;
                results.AppendLine($"  {label}: {(ok ? "OK" : $"FAILED (exit {proc.ExitCode})")}");

                if (!ok)
                    Debug.LogError($"[RoomScan Setup] {label} build failed (exit {proc.ExitCode}):\n{stderr}");
                else if (!string.IsNullOrWhiteSpace(stderr))
                    Debug.LogWarning($"[RoomScan Setup] {label} warnings:\n{stderr}");
                else
                    Debug.Log($"[RoomScan Setup] {label} build succeeded");

                proc.Dispose();
            }

            AssetDatabase.Refresh();

            // Configure plugin importers after AssetDatabase sees the new files
            EditorApplication.delayCall += () =>
            {
                foreach (var (label, _, outAssetPath, _, _) in _activeBuilds)
                    ConfigurePluginImporter(outAssetPath);
                _activeBuilds = null;
            };

            if (allOk)
                Debug.Log($"[RoomScan Setup] xatlas build complete:\n{results}");
        }

        static void ConfigurePluginImporter(string assetPath)
        {
            var importer = AssetImporter.GetAtPath(assetPath) as PluginImporter;
            if (importer == null)
            {
                Debug.LogWarning($"[RoomScan Setup] PluginImporter not found for {assetPath}");
                return;
            }

            bool isAndroid = assetPath.Contains("/Android/");

            importer.SetCompatibleWithAnyPlatform(false);
            importer.SetCompatibleWithEditor(!isAndroid);
            importer.SetCompatibleWithPlatform(BuildTarget.Android, isAndroid);
            importer.SetCompatibleWithPlatform(BuildTarget.StandaloneOSX, !isAndroid);

            if (isAndroid)
                importer.SetPlatformData(BuildTarget.Android, "CPU", "ARM64");

            if (!isAndroid)
            {
                importer.SetEditorData("CPU", "AnyCPU");
                importer.SetEditorData("OS", "OSX");
            }

            importer.SaveAndReimport();
            Debug.Log($"[RoomScan Setup] Configured plugin importer: {assetPath}" +
                      (isAndroid ? " (Android ARM64)" : " (macOS Editor)"));
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
            if (!_boundarylessManifest) FixBoundarylessManifest();
            if (!_cleartextAllowed) FixCleartextTraffic();
            if (!_insecureHttpAllowed)
            {
                PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
                Debug.Log("[RoomScan Setup] Set Player Settings > insecureHttpOption to AlwaysAllowed");
            }
            FixComponents();
            FixShaderWiring();

            RefreshNativePlugins();
            if (!_xatlasAndroid || !_xatlasMacOS)
                BuildXAtlasPlugin();

            MarkDirty();
            Refresh();

            Debug.Log("[RoomScan Setup] Scene setup complete." +
                (!_xatlasAndroid || !_xatlasMacOS ? " (xatlas build running in background)" : ""));
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

        /// <summary>
        /// Sets m_RenderMode to 1 (WorldSpace). The public renderMode property
        /// and PanelRenderMode enum are internal in Unity 6000.3.
        /// </summary>
        static void SetPanelRenderModeWorldSpace(PanelSettings panel)
        {
            const int WorldSpace = 1;
            var so = new SerializedObject(panel);
            var prop = so.FindProperty("m_RenderMode");
            if (prop != null && prop.intValue != WorldSpace)
            {
                prop.intValue = WorldSpace;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(panel);
            }
        }

        static void EnsureDebugMenuAssets()
        {
            var ctrl = FindAny<DebugMenuController>();
            if (ctrl == null) return;

            var uiDoc = ctrl.GetComponent<UIDocument>();
            if (uiDoc == null) return;

            Undo.RecordObject(uiDoc, "Assign DebugMenu UIDocument assets");

            if (uiDoc.visualTreeAsset == null)
            {
                var uxml = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(
                    "Packages/com.genesis.roomscan/Runtime/UI/DebugMenu.uxml");
                if (uxml != null) uiDoc.visualTreeAsset = uxml;
            }

            if (uiDoc.panelSettings == null)
            {
                var panel = FindOrCreatePanelSettings();
                if (panel != null) uiDoc.panelSettings = panel;
            }

            // Ensure PanelSettings is configured for world-space VR rendering.
            // renderMode / PanelRenderMode are internal in 6000.3; use SerializedObject.
            if (uiDoc.panelSettings != null)
                SetPanelRenderModeWorldSpace(uiDoc.panelSettings);

            // World-space UIDocument properties:
            // - Dynamic size mode: panel auto-sizes to the content layout (480×640 from USS)
            // - Pivot = Center: transform position = center of the visible panel
            // - PivotReferenceSize = Layout: pivot calculated from root element layout, not bounding box
            uiDoc.worldSpaceSizeMode = UIDocument.WorldSpaceSizeMode.Dynamic;
            uiDoc.pivot = Pivot.Center;
            uiDoc.pivotReferenceSize = PivotReferenceSize.Layout;

            // 480px / 100 PPU = 4.8 local units. Scale 0.08 → 0.384m wide.
            const float worldScale = 0.08f;
            if (Mathf.Abs(ctrl.transform.localScale.x - worldScale) > 0.01f)
            {
                Undo.RecordObject(ctrl.transform, "Scale DebugMenu for VR");
                ctrl.transform.localScale = Vector3.one * worldScale;
            }

            EditorUtility.SetDirty(uiDoc);
        }

        static PanelSettings FindOrCreatePanelSettings()
        {
            const string assetName = "DebugMenuPanelSettings";

            string[] guids = AssetDatabase.FindAssets($"t:PanelSettings {assetName}");
            if (guids.Length > 0)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                var existing = AssetDatabase.LoadAssetAtPath<PanelSettings>(path);
                if (existing != null) return existing;
            }

            const string dir = "Assets/Settings";
            if (!AssetDatabase.IsValidFolder(dir))
                AssetDatabase.CreateFolder("Assets", "Settings");

            const string assetPath = dir + "/" + assetName + ".asset";
            var panel = ScriptableObject.CreateInstance<PanelSettings>();
            AssetDatabase.CreateAsset(panel, assetPath);
            SetPanelRenderModeWorldSpace(panel);
            AssetDatabase.SaveAssets();
            Debug.Log($"[RoomScanWizard] Created PanelSettings (WorldSpace) at {assetPath}");
            return panel;
        }

        /// <summary>
        /// Auto-detect this PC's LAN IP and write it into the GSplatServerClient
        /// serialized field so the Quest connects to the right address at runtime.
        /// </summary>
        static void ConfigureServerUrl()
        {
            var client = FindAny<GSplatServerClient>();
            if (client == null) return;

            string lanIp = GetLanIp();
            if (string.IsNullOrEmpty(lanIp)) return;

            string url = $"http://{lanIp}:8420";

            var so = new SerializedObject(client);
            var prop = so.FindProperty("serverUrl");
            if (prop != null && prop.stringValue != url)
            {
                prop.stringValue = url;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(client);
                Debug.Log($"[RoomScanWizard] Server URL set to {url} (detected LAN IP: {lanIp})");
            }
        }

        static string GetLanIp()
        {
            try
            {
                foreach (var ni in System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces())
                {
                    if (ni.OperationalStatus != System.Net.NetworkInformation.OperationalStatus.Up)
                        continue;
                    if (ni.NetworkInterfaceType == System.Net.NetworkInformation.NetworkInterfaceType.Loopback)
                        continue;

                    var props = ni.GetIPProperties();
                    foreach (var addr in props.UnicastAddresses)
                    {
                        if (addr.Address.AddressFamily != System.Net.Sockets.AddressFamily.InterNetwork)
                            continue;
                        string ip = addr.Address.ToString();
                        if (ip.StartsWith("127.")) continue;
                        return ip;
                    }
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[RoomScanWizard] Failed to detect LAN IP: {e.Message}");
            }
            return null;
        }
    }
}
