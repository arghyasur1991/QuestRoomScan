using System.IO;
using UnityEngine;
using UnityEngine.UIElements;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Controls the debug HUD panel. Reads live status from <see cref="RoomScanner"/>
    /// and related components. Action buttons call the RoomScanner public API directly.
    /// Uses <see cref="DebugMenuFollower"/> for world-space head-tracked positioning.
    ///
    /// Clients can:
    ///   - Call <see cref="Toggle"/>, <see cref="Show"/>, <see cref="Hide"/> from any script.
    ///   - Read <see cref="IsVisible"/> to check state.
    ///   - Override button behavior by subclassing or by disabling this component
    ///     and driving the UIDocument directly.
    /// </summary>
    [RequireComponent(typeof(UIDocument), typeof(DebugMenuFollower))]
    public class DebugMenuController : MonoBehaviour
    {
        private UIDocument _doc;
        private DebugMenuFollower _follower;
        private VisualElement _root;
        private bool _visible;

        // Status labels
        private Label _valScanning;
        private Label _valMode;
        private Label _valIntegrations;
        private Label _valKeyframes;
        private Label _valRender;
        private Label _valGsTraining;
        private Label _valSavedScan;
        private Label _valGsExport;
        private Label _valFps;

        // Action buttons
        private Button _btnToggleScan;
        private Button _btnClearAll;
        private Button _btnExportPc;
        private Button _btnGsTrain;

        // FPS tracking
        private float _fpsTimer;
        private int _fpsFrames;
        private float _currentFps;

        public bool IsVisible => _visible;

        private void Awake()
        {
            _doc = GetComponent<UIDocument>();
            _follower = GetComponent<DebugMenuFollower>();
        }

        private void OnEnable()
        {
            _root = _doc.rootVisualElement;

            // Constrain the rootVisualElement to the content size so the
            // world-space collider matches the visible panel exactly.
            _root.style.width = 480;
            _root.style.height = 640;
            _root.style.overflow = Overflow.Hidden;

            _root.style.display = DisplayStyle.None;
            _visible = false;

            QueryElements();
            BindButtons();
        }

        private void Update()
        {
            UpdateFps();
            if (_visible) RefreshStatus();
        }

        // ─────────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────────

        public void Toggle()
        {
            if (_visible) Hide();
            else Show();
        }

        public void Show()
        {
            _visible = true;
            _root.style.display = DisplayStyle.Flex;

            if (_follower != null) _follower.SnapToView();

            RefreshStatus();
        }

        public void Hide()
        {
            _visible = false;
            _root.style.display = DisplayStyle.None;

            if (_follower != null) _follower.StopTracking();
        }

        // ─────────────────────────────────────────────────────────────
        //  Internal
        // ─────────────────────────────────────────────────────────────

        private void QueryElements()
        {
            _valScanning = _root.Q<Label>("val-scanning");
            _valMode = _root.Q<Label>("val-mode");
            _valIntegrations = _root.Q<Label>("val-integrations");
            _valKeyframes = _root.Q<Label>("val-keyframes");
            _valRender = _root.Q<Label>("val-render");
            _valGsTraining = _root.Q<Label>("val-gs-training");
            _valSavedScan = _root.Q<Label>("val-saved-scan");
            _valGsExport = _root.Q<Label>("val-gsexport");
            _valFps = _root.Q<Label>("val-fps");

            _btnToggleScan = _root.Q<Button>("btn-toggle-scan");
            _btnClearAll = _root.Q<Button>("btn-clear-all");
            _btnExportPc = _root.Q<Button>("btn-export-pc");
            _btnGsTrain = _root.Q<Button>("btn-gs-train");
        }

        private void BindButtons()
        {
            _btnToggleScan?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.ToggleScanning());

            _btnClearAll?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.ClearAllData());

            _btnExportPc?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance != null)
                    await RoomScanner.Instance.ExportPointCloudAsync();
            });

            _btnGsTrain?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.StartServerTraining());
        }

        private void RefreshStatus()
        {
            var scanner = RoomScanner.Instance;
            if (scanner == null) return;

            SetLabel(_valScanning, scanner.IsScanning ? "Active" : "Stopped");
            SetLabel(_valMode, scanner.Mode.ToString());
            SetLabel(_valRender, scanner.CurrentRenderMode.ToString());
            SetLabel(_valGsTraining, scanner.IsGsTrainingInProgress ? "Running..." : "Idle");

            if (_btnToggleScan != null)
                _btnToggleScan.text = scanner.IsScanning ? "Stop Scanning" : "Start Scanning";

            var vi = VolumeIntegrator.Instance;
            if (vi != null)
                SetLabel(_valIntegrations, vi.IntegrationCount.ToString());

            var kf = FindAnyObjectByType<KeyframeCollector>();
            if (kf != null)
                SetLabel(_valKeyframes, kf.SavedCount.ToString());

            var persistence = RoomScanPersistence.Instance;
            if (persistence != null)
                SetLabel(_valSavedScan, persistence.HasSavedScan() ? "Yes" : "No");

            string gsExportDir = Path.Combine(Application.persistentDataPath, "GSExport");
            bool hasExport = Directory.Exists(gsExportDir)
                && Directory.GetFiles(gsExportDir, "*.jpg", SearchOption.AllDirectories).Length > 0;
            SetLabel(_valGsExport, hasExport ? "Yes" : "No");

            SetLabel(_valFps, $"{_currentFps:F0} FPS");
        }

        private void UpdateFps()
        {
            _fpsFrames++;
            _fpsTimer += Time.unscaledDeltaTime;
            if (_fpsTimer >= 0.5f)
            {
                _currentFps = _fpsFrames / _fpsTimer;
                _fpsFrames = 0;
                _fpsTimer = 0f;
            }
        }

        private static void SetLabel(Label label, string text)
        {
            if (label != null) label.text = text;
        }
    }
}
