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
        private Button _btnSaveScan;
        private Button _btnLoadScan;
        private Button _btnClearAll;
        private Button _btnExportPc;
        private Button _btnGsTrain;

        // FPS tracking
        private float _fpsTimer;
        private int _fpsFrames;
        private float _currentFps;

        // Cached file checks (avoid per-frame I/O)
        private float _ioCheckTimer;
        private bool _hasGsExport;
        private bool _hasSavedScan;

        public bool IsVisible => _visible;

        private void Awake()
        {
            _doc = GetComponent<UIDocument>();
            _follower = GetComponent<DebugMenuFollower>();
        }

        private void OnEnable()
        {
            _root = _doc.rootVisualElement;
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
            _btnSaveScan = _root.Q<Button>("btn-save-scan");
            _btnLoadScan = _root.Q<Button>("btn-load-scan");
            _btnClearAll = _root.Q<Button>("btn-clear-all");
            _btnExportPc = _root.Q<Button>("btn-export-pc");
            _btnGsTrain = _root.Q<Button>("btn-gs-train");
        }

        private void BindButtons()
        {
            _btnToggleScan?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.ToggleScanning());

            _btnSaveScan?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnSaveScan, "Saving...");
                bool ok = await RoomScanner.Instance.SaveScanAsync();
                SetButtonReady(_btnSaveScan, "Save Scan");
                FlashStatus(_btnSaveScan, ok);
            });

            _btnLoadScan?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnLoadScan, "Loading...");
                bool ok = await RoomScanner.Instance.LoadScanAsync();
                SetButtonReady(_btnLoadScan, "Load Scan");
                FlashStatus(_btnLoadScan, ok);
            });

            _btnClearAll?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnClearAll, "Clearing...");
                await RoomScanner.Instance.ClearAllDataAsync();
                SetButtonReady(_btnClearAll, "Clear All Data");
            });

            _btnExportPc?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnExportPc, "Exporting...");
                await RoomScanner.Instance.ExportPointCloudAsync();
                SetButtonReady(_btnExportPc, "Export Point Cloud");
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

            _ioCheckTimer -= Time.deltaTime;
            if (_ioCheckTimer <= 0f)
            {
                _ioCheckTimer = 2f;

                var persistence = RoomScanPersistence.Instance;
                if (persistence != null)
                    _hasSavedScan = persistence.HasSavedScan();

                string gsExportDir = Path.Combine(Application.persistentDataPath, "GSExport");
                _hasGsExport = Directory.Exists(gsExportDir)
                    && Directory.GetFiles(gsExportDir, "*.jpg", SearchOption.AllDirectories).Length > 0;
            }
            SetLabel(_valSavedScan, _hasSavedScan ? "Yes" : "No");
            SetLabel(_valGsExport, _hasGsExport ? "Yes" : "No");

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

        private static void SetButtonBusy(Button btn, string text)
        {
            if (btn == null) return;
            btn.text = text;
            btn.SetEnabled(false);
        }

        private static void SetButtonReady(Button btn, string text)
        {
            if (btn == null) return;
            btn.text = text;
            btn.SetEnabled(true);
        }

        private static async void FlashStatus(Button btn, bool success)
        {
            if (btn == null) return;
            string original = btn.text;
            btn.text = success ? "Done!" : "Failed";
            await System.Threading.Tasks.Task.Delay(1500);
            if (btn != null) btn.text = original;
        }
    }
}
