using System.IO;
using Genesis.RoomScan.GSplat;
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

        // Scan status labels
        private Label _valScanning;
        private Label _valMode;
        private Label _valIntegrations;
        private Label _valKeyframes;
        private Label _valRender;

        // Server training fields
        private TextField _fieldServerUrl;
        private Label _valTrainState;
        private VisualElement _progressFill;
        private Label _valTrainProgress;
        private Label _valTrainIter;
        private Label _valTrainElapsed;
        private Label _valTrainBackend;
        private Label _valTrainMessage;

        // Persistence labels
        private Label _valSavedScan;
        private Label _valGsExport;
        private Label _valFps;

        // Action buttons
        private Button _btnToggleScan;
        private Button _btnRenderMode;
        private Button _btnSaveScan;
        private Button _btnLoadScan;
        private Button _btnClearAll;
        private Button _btnExportPc;
        private Button _btnGsTrain;
        private Button _btnCancelTrain;

        // FPS tracking
        private float _fpsTimer;
        private int _fpsFrames;
        private float _currentFps;

        // Cached references and file checks (avoid per-frame I/O / FindObject)
        private GSplatServerClient _cachedClient;
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

            _fieldServerUrl = _root.Q<TextField>("field-server-url");
            _valTrainState = _root.Q<Label>("val-train-state");
            _progressFill = _root.Q<VisualElement>("progress-fill");
            _valTrainProgress = _root.Q<Label>("val-train-progress");
            _valTrainIter = _root.Q<Label>("val-train-iter");
            _valTrainElapsed = _root.Q<Label>("val-train-elapsed");
            _valTrainBackend = _root.Q<Label>("val-train-backend");
            _valTrainMessage = _root.Q<Label>("val-train-message");

            _valSavedScan = _root.Q<Label>("val-saved-scan");
            _valGsExport = _root.Q<Label>("val-gsexport");
            _valFps = _root.Q<Label>("val-fps");

            _btnToggleScan = _root.Q<Button>("btn-toggle-scan");
            _btnRenderMode = _root.Q<Button>("btn-render-mode");
            _btnSaveScan = _root.Q<Button>("btn-save-scan");
            _btnLoadScan = _root.Q<Button>("btn-load-scan");
            _btnClearAll = _root.Q<Button>("btn-clear-all");
            _btnExportPc = _root.Q<Button>("btn-export-pc");
            _btnGsTrain = _root.Q<Button>("btn-gs-train");
            _btnCancelTrain = _root.Q<Button>("btn-cancel-train");
        }

        private void BindButtons()
        {
            _btnToggleScan?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.ToggleScanning());

            _btnRenderMode?.RegisterCallback<ClickEvent>(_ =>
            {
                var scanner = RoomScanner.Instance;
                if (scanner == null) return;
                if (scanner.HasDownloadedSplat &&
                    scanner.CurrentRenderMode == ScanRenderMode.Mesh)
                {
                    scanner.LoadDownloadedSplat();
                }
                else
                {
                    scanner.CycleRenderMode();
                }
            });

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

            _btnClearAll?.RegisterCallback<ClickEvent>(_ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnClearAll, "Clearing...");
                RoomScanner.Instance.ClearAllDataAsync(() =>
                    SetButtonReady(_btnClearAll, "Clear All Data"));
            });

            _btnExportPc?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnExportPc, "Exporting...");
                await RoomScanner.Instance.ExportPointCloudAsync();
                SetButtonReady(_btnExportPc, "Export Point Cloud");
            });

            _fieldServerUrl?.RegisterValueChangedCallback(evt =>
            {
                if (_cachedClient == null)
                    _cachedClient = FindAnyObjectByType<GSplatServerClient>();
                if (_cachedClient != null) _cachedClient.ServerUrl = evt.newValue;
            });

            _btnGsTrain?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.StartServerTraining());

            _btnCancelTrain?.RegisterCallback<ClickEvent>(evt =>
            {
                if (_cachedClient == null)
                    _cachedClient = FindAnyObjectByType<GSplatServerClient>();
                if (_cachedClient == null) return;
                SetButtonBusy(_btnCancelTrain, "Cancelling...");
                CancelAndResetButton();
            });
        }

        private void RefreshStatus()
        {
            var scanner = RoomScanner.Instance;
            if (scanner == null) return;

            // Scan status
            SetLabel(_valScanning, scanner.IsScanning ? "Active" : "Stopped");
            SetLabel(_valMode, scanner.Mode.ToString());
            SetLabel(_valRender, scanner.CurrentRenderMode.ToString());

            if (_btnToggleScan != null)
                _btnToggleScan.text = scanner.IsScanning ? "Stop Scanning" : "Start Scanning";

            if (_btnRenderMode != null)
            {
                string modeLabel = scanner.CurrentRenderMode.ToString();
                if (scanner.HasDownloadedSplat && scanner.CurrentRenderMode == ScanRenderMode.Mesh)
                    modeLabel += " [Splat Ready]";
                _btnRenderMode.text = $"Render: {modeLabel}";
            }

            var vi = VolumeIntegrator.Instance;
            if (vi != null)
                SetLabel(_valIntegrations, vi.IntegrationCount.ToString());

            var kf = FindAnyObjectByType<KeyframeCollector>();
            if (kf != null)
                SetLabel(_valKeyframes, kf.SavedCount.ToString());

            // Server training status
            RefreshTrainingStatus();

            // Persistence (throttled I/O)
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

        private void RefreshTrainingStatus()
        {
            if (_cachedClient == null)
                _cachedClient = FindAnyObjectByType<GSplatServerClient>();
            if (_cachedClient == null) return;

            if (_fieldServerUrl != null && _fieldServerUrl.value != _cachedClient.ServerUrl)
                _fieldServerUrl.SetValueWithoutNotify(_cachedClient.ServerUrl);

            var ts = _cachedClient.LastStatus;
            if (ts == null)
            {
                SetLabel(_valTrainState, "No data");
                return;
            }

            // Show local client activity (upload/download) as state override
            string stateDisplay = ts.state ?? "--";
            if (_cachedClient.IsUploading) stateDisplay = "Uploading...";
            else if (_cachedClient.IsDownloading) stateDisplay = "Downloading...";
            else if (_cachedClient.IsPolling && ts.state == "training") stateDisplay = "Training (polling)";
            SetLabel(_valTrainState, stateDisplay);

            float pct = ts.progress * 100f;
            if (_progressFill != null)
                _progressFill.style.width = new Length(pct, LengthUnit.Percent);
            SetLabel(_valTrainProgress, $"{pct:F0}%");

            if (ts.total_iterations > 0)
                SetLabel(_valTrainIter, $"{ts.current_iteration} / {ts.total_iterations}");
            else
                SetLabel(_valTrainIter, "--");

            SetLabel(_valTrainElapsed, ts.elapsed_seconds > 0 ? FormatElapsed(ts.elapsed_seconds) : "--");
            SetLabel(_valTrainBackend, string.IsNullOrEmpty(ts.backend) ? "--" : ts.backend);
            SetLabel(_valTrainMessage, string.IsNullOrEmpty(ts.message) ? "--" : ts.message);

            bool isTraining = ts.state == "training";
            if (_btnGsTrain != null) _btnGsTrain.SetEnabled(!isTraining);
            if (_btnCancelTrain != null) _btnCancelTrain.SetEnabled(isTraining);
        }

        private static string FormatElapsed(float seconds)
        {
            if (seconds < 60f) return $"{seconds:F0}s";
            int m = (int)(seconds / 60f);
            int s = (int)(seconds % 60f);
            if (m < 60) return $"{m}m {s}s";
            int h = m / 60;
            return $"{h}h {m % 60}m";
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

        private async void CancelAndResetButton()
        {
            if (_cachedClient != null)
                await _cachedClient.CancelTraining();
            SetButtonReady(_btnCancelTrain, "Cancel Training");
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
