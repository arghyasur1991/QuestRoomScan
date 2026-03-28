using UnityEngine;
using UnityEngine.UIElements;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Two-panel debug menu controller. Left nav switches right-panel views.
    /// Manages scan browser list, disabled states, and status refresh.
    /// </summary>
    [RequireComponent(typeof(UIDocument), typeof(DebugMenuFollower))]
    public class DebugMenuController : MonoBehaviour
    {
        private UIDocument _doc;
        private DebugMenuFollower _follower;
        private VisualElement _root;
        private bool _visible;

        // Nav buttons
        private Button _navScan, _navSaved, _navRefine, _navTraining, _navTools;
        private Button[] _navButtons;

        // Views
        private VisualElement _viewScan, _viewSaved, _viewRefine, _viewTraining, _viewTools;
        private VisualElement[] _views;

        // Scan view elements
        private Label _valScanning, _valIntegrations, _valKeyframes, _valRender, _valPackage;
        private Label _valProgress, _valPhase, _valColorCoverage, _valFrozen, _valMeshStats;
        private Button _btnToggleScan, _btnRenderMode, _btnFreezeTint, _btnSaveScan, _btnDeleteArtifact;

        // Saved scans view
        private ScrollView _scanList;
        private Label _lblNoScans;

        // Refine view
        private Label _valRefineStatus, _valHqStatus, _valMeshEnhanceStatus;
        private Label _valRefinedMeshStats, _valSimplifiedMeshStats;
        private Button _btnRefineTex, _btnHqRefine, _btnMeshEnhance;

        // Training view
        private TextField _fieldServerUrl;
        private Label _valTrainState, _valTrainProgress, _valTrainIter;
        private Label _valTrainElapsed, _valTrainBackend, _valTrainMessage;
        private VisualElement _progressFill;
        private Button _btnGsTrain, _btnCancelTrain;

        // Tools view
        private Button _btnExportPc, _btnClearAll;

        // Footer
        private Label _valFps;

        // Cached state
        private IGSplatProvider _cachedGSplat;
        private float _fpsTimer;
        private int _fpsFrames;
        private float _currentFps;
        private float _scanListRefreshTimer;

        public bool IsVisible => _visible;

        private void Awake()
        {
            _doc = GetComponent<UIDocument>();
            _follower = GetComponent<DebugMenuFollower>();
        }

        private bool _refineAvailable;
        private bool _gsplatAvailable;

        private void OnEnable()
        {
            _root = _doc.rootVisualElement;
            _root.style.display = DisplayStyle.None;
            _visible = false;

            QueryElements();
            BindButtons();
            UpdateModuleAvailability();
            SelectNav(_navScan);
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
            if (_visible) Hide(); else Show();
        }

        public void Show()
        {
            _visible = true;
            _root.style.display = DisplayStyle.Flex;
            if (_follower != null) _follower.SnapToView();
            UpdateModuleAvailability();
            PopulateScanList();
            RefreshStatus();
        }

        /// <summary>
        /// Switches the right panel to the Saved Scans view and refreshes the list.
        /// </summary>
        public void ShowSavedScans()
        {
            if (!_visible) Show();
            SelectNav(_navSaved);
            PopulateScanList();
        }

        public void Hide()
        {
            _visible = false;
            _root.style.display = DisplayStyle.None;
            if (_follower != null) _follower.StopTracking();
        }

        // ─────────────────────────────────────────────────────────────
        //  Query & Bind
        // ─────────────────────────────────────────────────────────────

        private void QueryElements()
        {
            // Nav
            _navScan = _root.Q<Button>("nav-scan");
            _navSaved = _root.Q<Button>("nav-saved");
            _navRefine = _root.Q<Button>("nav-refine");
            _navTraining = _root.Q<Button>("nav-training");
            _navTools = _root.Q<Button>("nav-tools");
            _navButtons = new[] { _navScan, _navSaved, _navRefine, _navTraining, _navTools };

            // Views
            _viewScan = _root.Q<VisualElement>("view-scan");
            _viewSaved = _root.Q<VisualElement>("view-saved");
            _viewRefine = _root.Q<VisualElement>("view-refine");
            _viewTraining = _root.Q<VisualElement>("view-training");
            _viewTools = _root.Q<VisualElement>("view-tools");
            _views = new[] { _viewScan, _viewSaved, _viewRefine, _viewTraining, _viewTools };

            // Scan view
            _valScanning = _root.Q<Label>("val-scanning");

            _valIntegrations = _root.Q<Label>("val-integrations");
            _valKeyframes = _root.Q<Label>("val-keyframes");
            _valRender = _root.Q<Label>("val-render");
            _valPackage = _root.Q<Label>("val-package");
            _valProgress = _root.Q<Label>("val-progress");
            _valPhase = _root.Q<Label>("val-phase");
            _valColorCoverage = _root.Q<Label>("val-color-coverage");
            _valFrozen = _root.Q<Label>("val-frozen");
            _valMeshStats = _root.Q<Label>("val-mesh-stats");
            _btnToggleScan = _root.Q<Button>("btn-toggle-scan");
            _btnRenderMode = _root.Q<Button>("btn-render-mode");
            _btnFreezeTint = _root.Q<Button>("btn-freeze-tint");
            _btnSaveScan = _root.Q<Button>("btn-save-scan");
            _btnDeleteArtifact = _root.Q<Button>("btn-delete-artifact");

            // Saved scans
            _scanList = _root.Q<ScrollView>("scan-list");
            _lblNoScans = _root.Q<Label>("lbl-no-scans");

            // Refine
            _valRefineStatus = _root.Q<Label>("val-refine-status");
            _valHqStatus = _root.Q<Label>("val-hq-status");
            _valMeshEnhanceStatus = _root.Q<Label>("val-mesh-enhance-status");
            _valRefinedMeshStats = _root.Q<Label>("val-refined-mesh-stats");
            _valSimplifiedMeshStats = _root.Q<Label>("val-simplified-mesh-stats");
            _btnRefineTex = _root.Q<Button>("btn-refine-tex");
            _btnHqRefine = _root.Q<Button>("btn-hq-refine");
            _btnMeshEnhance = _root.Q<Button>("btn-mesh-enhance");

            // Training
            _fieldServerUrl = _root.Q<TextField>("field-server-url");
            _valTrainState = _root.Q<Label>("val-train-state");
            _progressFill = _root.Q<VisualElement>("progress-fill");
            _valTrainProgress = _root.Q<Label>("val-train-progress");
            _valTrainIter = _root.Q<Label>("val-train-iter");
            _valTrainElapsed = _root.Q<Label>("val-train-elapsed");
            _valTrainBackend = _root.Q<Label>("val-train-backend");
            _valTrainMessage = _root.Q<Label>("val-train-message");
            _btnGsTrain = _root.Q<Button>("btn-gs-train");
            _btnCancelTrain = _root.Q<Button>("btn-cancel-train");

            // Tools
            _btnExportPc = _root.Q<Button>("btn-export-pc");
            _btnClearAll = _root.Q<Button>("btn-clear-all");

            // Footer
            _valFps = _root.Q<Label>("val-fps");
        }

        private void BindButtons()
        {
            // Nav switching
            _navScan?.RegisterCallback<ClickEvent>(_ => SelectNav(_navScan));
            _navSaved?.RegisterCallback<ClickEvent>(_ => { SelectNav(_navSaved); PopulateScanList(); });
            _navRefine?.RegisterCallback<ClickEvent>(_ => SelectNav(_navRefine));
            _navTraining?.RegisterCallback<ClickEvent>(_ => SelectNav(_navTraining));
            _navTools?.RegisterCallback<ClickEvent>(_ => SelectNav(_navTools));

            // Scan view buttons
            _btnToggleScan?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.ToggleScanning());

            _btnRenderMode?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.CycleRenderMode());

            _btnFreezeTint?.RegisterCallback<ClickEvent>(_ =>
            {
                var s = RoomScanner.Instance;
                if (s != null) s.ShowFreezeTint = !s.ShowFreezeTint;
            });

            _btnSaveScan?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnSaveScan, "Saving...");
                bool ok = await RoomScanner.Instance.SaveScanAsync();
                SetButtonReady(_btnSaveScan, "Save Scan");
                FlashStatus(_btnSaveScan, ok);
            });

            _btnDeleteArtifact?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.DeleteActiveArtifact());

            // Refine
            _btnRefineTex?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.StartTextureRefinement());

            _btnHqRefine?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.StartHQRefinement());

            _btnMeshEnhance?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.StartMeshEnhancement());

            // Training
            _fieldServerUrl?.RegisterValueChangedCallback(evt =>
            {
                EnsureGSplat();
                if (_cachedGSplat != null) _cachedGSplat.ServerUrl = evt.newValue;
            });

            _btnGsTrain?.RegisterCallback<ClickEvent>(_ =>
                RoomScanner.Instance?.StartServerTraining());

            _btnCancelTrain?.RegisterCallback<ClickEvent>(_ =>
            {
                EnsureGSplat();
                if (_cachedGSplat == null) return;
                SetButtonBusy(_btnCancelTrain, "Cancelling...");
                CancelAndResetButton();
            });

            // Tools
            _btnExportPc?.RegisterCallback<ClickEvent>(async _ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnExportPc, "Exporting...");
                await RoomScanner.Instance.ExportPointCloudAsync();
                SetButtonReady(_btnExportPc, "Export Point Cloud");
            });

            _btnClearAll?.RegisterCallback<ClickEvent>(_ =>
            {
                if (RoomScanner.Instance == null) return;
                SetButtonBusy(_btnClearAll, "Clearing...");
                RoomScanner.Instance.ClearAllDataAsync(() =>
                    SetButtonReady(_btnClearAll, "Clear All Data"));
            });
        }

        // ─────────────────────────────────────────────────────────────
        //  Module Availability
        // ─────────────────────────────────────────────────────────────

        private void UpdateModuleAvailability()
        {
            var scanner = RoomScanner.Instance;
            _refineAvailable = scanner != null && scanner.HasTextureRefinementModule;
            _gsplatAvailable = scanner != null && scanner.GSplatProvider != null;

            SetNavAvailable(_navRefine, _refineAvailable);
            SetNavAvailable(_navTraining, _gsplatAvailable);
        }

        private static void SetNavAvailable(Button btn, bool available)
        {
            if (btn == null) return;
            btn.SetEnabled(available);
            btn.EnableInClassList("nav-btn--unavailable", !available);
        }

        // ─────────────────────────────────────────────────────────────
        //  Navigation
        // ─────────────────────────────────────────────────────────────

        private void SelectNav(Button selected)
        {
            if (selected != null && !selected.enabledSelf) return;

            for (int i = 0; i < _navButtons.Length; i++)
            {
                if (_navButtons[i] == null) continue;

                bool isActive = _navButtons[i] == selected;
                _navButtons[i].EnableInClassList("nav-btn--active", isActive);
                if (_views[i] != null)
                    _views[i].style.display = isActive ? DisplayStyle.Flex : DisplayStyle.None;
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  Scan List
        // ─────────────────────────────────────────────────────────────

        private void PopulateScanList()
        {
            if (_scanList == null) return;
            _scanList.Clear();

            var persistence = RoomScanPersistence.Instance;
            if (persistence == null) return;

            var packages = persistence.ListPackages();
            bool empty = packages.Count == 0;
            if (_lblNoScans != null)
                _lblNoScans.style.display = empty ? DisplayStyle.Flex : DisplayStyle.None;
            if (_scanList != null)
                _scanList.style.display = empty ? DisplayStyle.None : DisplayStyle.Flex;

            foreach (var pkg in packages)
            {
                var entry = CreateScanEntry(pkg);
                _scanList.Add(entry);
            }

            // Update nav badge count
            if (_navSaved != null)
                _navSaved.text = packages.Count > 0
                    ? $"Saved Scans ({packages.Count})"
                    : "Saved Scans";
        }

        private VisualElement CreateScanEntry(ScanPackageEntry pkg)
        {
            var row = new VisualElement();
            row.AddToClassList("scan-entry");

            var info = new VisualElement();
            info.AddToClassList("scan-entry-info");

            var nameLabel = new Label(pkg.displayName ?? pkg.id);
            nameLabel.AddToClassList("scan-entry-name");
            info.Add(nameLabel);

            var ts = System.DateTimeOffset.FromUnixTimeSeconds(pkg.timestamp).LocalDateTime;
            var meta = new Label(ts.ToString("yyyy-MM-dd HH:mm"));
            meta.AddToClassList("scan-entry-meta");
            info.Add(meta);

            var badges = new VisualElement();
            badges.AddToClassList("scan-entry-badges");
            if (pkg.hasKeyframes) AddBadge(badges, "KF");
            if (pkg.hasTriplanar) AddBadge(badges, "Tri");
            if (pkg.hasSplat) AddBadge(badges, "Splat");
            if (pkg.hasRefined) AddBadge(badges, "Refined");
            if (pkg.hasEnhancedMesh) AddBadge(badges, "Enh");
            if (pkg.hasHQRefined) AddBadge(badges, "HQ");
            info.Add(badges);

            row.Add(info);

            var btns = new VisualElement();
            btns.AddToClassList("scan-entry-btns");

            var loadBtn = new Button { text = "Load" };
            loadBtn.AddToClassList("scan-entry-btn");
            loadBtn.AddToClassList("scan-entry-btn--load");
            string pkgId = pkg.id;
            loadBtn.RegisterCallback<ClickEvent>(async _ =>
            {
                var scanner = RoomScanner.Instance;
                if (scanner == null) return;
                loadBtn.text = "...";
                loadBtn.SetEnabled(false);
                bool ok = await scanner.LoadPackageAsync(pkgId);
                loadBtn.text = "Load";
                loadBtn.SetEnabled(true);
                if (ok) SelectNav(_navScan);
            });
            btns.Add(loadBtn);

            var delBtn = new Button { text = "Del" };
            delBtn.AddToClassList("scan-entry-btn");
            delBtn.AddToClassList("scan-entry-btn--delete");
            delBtn.RegisterCallback<ClickEvent>(async _ =>
            {
                var p = RoomScanPersistence.Instance;
                if (p == null) return;
                delBtn.text = "...";
                delBtn.SetEnabled(false);
                await p.DeletePackageAsync(pkgId);
                PopulateScanList();
            });
            btns.Add(delBtn);

            row.Add(btns);
            return row;
        }

        private static void AddBadge(VisualElement parent, string text)
        {
            var badge = new Label(text);
            badge.AddToClassList("badge");
            parent.Add(badge);
        }

        // ─────────────────────────────────────────────────────────────
        //  Status Refresh (per-frame when visible)
        // ─────────────────────────────────────────────────────────────

        private void RefreshStatus()
        {
            var scanner = RoomScanner.Instance;
            if (scanner == null) return;

            RefreshScanView(scanner);
            RefreshRefineView(scanner);
            RefreshTrainingStatus();
            RefreshDisabledStates(scanner);

            SetLabel(_valFps, $"{_currentFps:F0} FPS");

            // Periodically refresh scan list when on saved view
            _scanListRefreshTimer -= Time.deltaTime;
            if (_scanListRefreshTimer <= 0f)
            {
                _scanListRefreshTimer = 3f;
                if (_navSaved != null)
                {
                    var p = RoomScanPersistence.Instance;
                    int count = p != null ? p.ListPackages().Count : 0;
                    _navSaved.text = count > 0 ? $"Saved Scans ({count})" : "Saved Scans";
                }
            }
        }

        private void RefreshScanView(RoomScanner scanner)
        {
            SetLabel(_valScanning, scanner.IsScanning ? "Active" : "Stopped");
            SetLabel(_valRender, scanner.CurrentRenderMode.ToString());

            var persistence = RoomScanPersistence.Instance;
            SetLabel(_valPackage, persistence != null && persistence.HasActivePackage
                ? persistence.ActivePackageId : "None");

            if (_btnToggleScan != null)
                _btnToggleScan.text = scanner.IsScanning ? "Stop Scanning" : "Start Scanning";

            if (_btnRenderMode != null)
                _btnRenderMode.text = $"Render: {scanner.CurrentRenderMode}";

            if (_btnFreezeTint != null)
                _btnFreezeTint.text = scanner.ShowFreezeTint ? "Freeze Tint: ON" : "Freeze Tint: OFF";

            var vi = VolumeIntegrator.Instance;
            if (vi != null) SetLabel(_valIntegrations, vi.IntegrationCount.ToString());

            var kf = FindAnyObjectByType<KeyframeCollector>();
            if (kf != null) SetLabel(_valKeyframes, kf.SavedCount.ToString());

            var progress = scanner.CurrentProgress;
            var cov = progress.Coverage;
            SetLabel(_valProgress, $"{progress.OverallProgress:P0}");
            SetLabel(_valPhase, progress.Phase.ToString());
            SetLabel(_valColorCoverage, $"{cov.ColorCoverage:P0}");
            SetLabel(_valFrozen, $"{cov.FrozenFraction:P0}");
            SetLabel(_valMeshStats, cov.MeshVertexCount > 0
                ? $"{cov.MeshVertexCount / 1000f:F1}K verts, {cov.MeshTriangleCount / 1000f:F1}K tris"
                : "--");

            // Delete artifact button visibility
            if (_btnDeleteArtifact != null)
            {
                var mode = scanner.CurrentRenderMode;
                bool canDelete = mode == ScanRenderMode.Splat ||
                                 mode == ScanRenderMode.Refined ||
                                 mode == ScanRenderMode.HQRefined;
                _btnDeleteArtifact.style.display = canDelete ? DisplayStyle.Flex : DisplayStyle.None;

                if (canDelete)
                {
                    _btnDeleteArtifact.text = mode switch
                    {
                        ScanRenderMode.Splat => "Delete Splat",
                        ScanRenderMode.Refined => scanner.HasEnhancedMesh ? "Delete Enhanced Mesh" : "Delete Refined",
                        ScanRenderMode.HQRefined => "Delete HQ Atlas",
                        _ => "Delete Artifact"
                    };
                    _btnDeleteArtifact.SetEnabled(persistence != null && persistence.HasActivePackage);
                }
            }
        }

        private void RefreshRefineView(RoomScanner scanner)
        {
            if (_btnRefineTex != null)
            {
                if (scanner.IsRefining)
                {
                    _btnRefineTex.text = scanner.RefineStatus ?? "Refining...";
                    _btnRefineTex.SetEnabled(false);
                }
                else
                {
                    _btnRefineTex.text = scanner.HasRefinedTexture ? "Refine (Done)" : "Refine Textures";
                    _btnRefineTex.SetEnabled(true);
                }
            }

            if (_btnHqRefine != null)
            {
                if (scanner.IsHQRefining)
                {
                    _btnHqRefine.text = scanner.HQRefineStatus ?? "HQ Refining...";
                    _btnHqRefine.SetEnabled(false);
                }
                else
                {
                    _btnHqRefine.text = scanner.HasHQRefinedTexture ? "HQ Refine (Done)" : "HQ Refine (Server)";
                    _btnHqRefine.SetEnabled(true);
                }
            }

            SetLabel(_valRefineStatus, scanner.IsRefining
                ? (scanner.RefineStatus ?? "Refining...") : (scanner.HasRefinedTexture ? "Done" : "Idle"));
            SetLabel(_valHqStatus, scanner.IsHQRefining
                ? (scanner.HQRefineStatus ?? "Processing...") : (scanner.HasHQRefinedTexture ? "Done" : "Idle"));

            if (_btnMeshEnhance != null)
            {
                if (scanner.IsMeshEnhancing)
                {
                    _btnMeshEnhance.text = scanner.MeshEnhanceStatus ?? "Enhancing...";
                    _btnMeshEnhance.SetEnabled(false);
                }
                else
                {
                    _btnMeshEnhance.text = "Enhance Mesh (Server)";
                    _btnMeshEnhance.SetEnabled(true);
                }
            }
            SetLabel(_valMeshEnhanceStatus, scanner.IsMeshEnhancing
                ? (scanner.MeshEnhanceStatus ?? "Processing...") : "Idle");

            if (scanner.LastRefinedResult.HasValue)
            {
                var r = scanner.LastRefinedResult.Value;
                SetLabel(_valRefinedMeshStats,
                    $"{r.Positions.Length / 1000f:F1}K verts, {r.Indices.Length / 3 / 1000f:F1}K tris");
            }
            else
            {
                SetLabel(_valRefinedMeshStats, "--");
            }

            if (scanner.LastSimplifiedResult.HasValue)
            {
                var s = scanner.LastSimplifiedResult.Value;
                SetLabel(_valSimplifiedMeshStats,
                    $"{s.Positions.Length / 1000f:F1}K verts, {s.Indices.Length / 3 / 1000f:F1}K tris");
            }
            else
            {
                SetLabel(_valSimplifiedMeshStats, "--");
            }
        }

        private void RefreshTrainingStatus()
        {
            EnsureGSplat();
            if (_cachedGSplat == null) return;

            if (_fieldServerUrl != null && _fieldServerUrl.value != _cachedGSplat.ServerUrl)
                _fieldServerUrl.SetValueWithoutNotify(_cachedGSplat.ServerUrl);

            string state = _cachedGSplat.TrainingState;
            if (string.IsNullOrEmpty(state))
            {
                SetLabel(_valTrainState, "No data");
                return;
            }

            string stateDisplay = state;
            if (_cachedGSplat.IsUploading) stateDisplay = "Uploading...";
            else if (_cachedGSplat.IsDownloading) stateDisplay = "Downloading...";
            else if (_cachedGSplat.IsPolling && state == "training") stateDisplay = "Training (polling)";
            SetLabel(_valTrainState, stateDisplay);

            float pct = _cachedGSplat.TrainingProgress * 100f;
            if (_progressFill != null)
                _progressFill.style.width = new Length(pct, LengthUnit.Percent);
            SetLabel(_valTrainProgress, $"{pct:F0}%");

            SetLabel(_valTrainIter, _cachedGSplat.TotalIterations > 0
                ? $"{_cachedGSplat.CurrentIteration} / {_cachedGSplat.TotalIterations}" : "--");
            SetLabel(_valTrainElapsed, _cachedGSplat.ElapsedSeconds > 0 ? FormatElapsed(_cachedGSplat.ElapsedSeconds) : "--");
            SetLabel(_valTrainBackend, string.IsNullOrEmpty(_cachedGSplat.TrainingBackend) ? "--" : _cachedGSplat.TrainingBackend);
            SetLabel(_valTrainMessage, string.IsNullOrEmpty(_cachedGSplat.TrainingMessage) ? "--" : _cachedGSplat.TrainingMessage);
        }

        private void RefreshDisabledStates(RoomScanner scanner)
        {
            var vi = VolumeIntegrator.Instance;
            var persistence = RoomScanPersistence.Instance;
            bool hasVolume = vi != null && vi.IntegrationCount > 0;
            bool hasActivePackage = persistence != null && persistence.HasActivePackage;

            // Save Scan: disabled if no volume data or while scanning
            if (_btnSaveScan != null && !_btnSaveScan.text.Contains("..."))
                _btnSaveScan.SetEnabled(hasVolume && !scanner.IsScanning);

            // GS Training: entirely disabled if module absent
            if (_btnGsTrain != null)
            {
                bool canTrain = _gsplatAvailable && !scanner.IsGsTrainingInProgress && (hasVolume || hasActivePackage);
                _btnGsTrain.SetEnabled(canTrain);
            }

            bool isTraining = _cachedGSplat?.TrainingState == "training";
            if (_btnCancelTrain != null && !_btnCancelTrain.text.Contains("..."))
                _btnCancelTrain.SetEnabled(isTraining);

            // Refine: entirely disabled if module absent
            if (_btnRefineTex != null && !scanner.IsRefining)
                _btnRefineTex.SetEnabled(_refineAvailable && (hasVolume || hasActivePackage));

            if (_btnHqRefine != null && !scanner.IsHQRefining)
            {
                bool hasServer = _cachedGSplat != null &&
                    !string.IsNullOrEmpty(_cachedGSplat.ServerUrl);
                _btnHqRefine.SetEnabled(_refineAvailable && (hasVolume || hasActivePackage) && hasServer);
            }

            if (_btnMeshEnhance != null && !scanner.IsMeshEnhancing)
            {
                bool hasServer = _cachedGSplat != null &&
                    !string.IsNullOrEmpty(_cachedGSplat.ServerUrl);
                bool hasRefined = scanner.HasRefinedTexture || scanner.LastRefinedResult.HasValue;
                _btnMeshEnhance.SetEnabled(_refineAvailable && hasRefined && hasServer);
            }

            // Export Point Cloud: disabled if no volume
            if (_btnExportPc != null && !_btnExportPc.text.Contains("..."))
                _btnExportPc.SetEnabled(hasVolume);
        }

        // ─────────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────────

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
            if (_cachedGSplat != null)
                await _cachedGSplat.CancelTraining();
            SetButtonReady(_btnCancelTrain, "Cancel Training");
        }

        private void EnsureGSplat()
        {
            if (_cachedGSplat != null) return;
            var scanner = RoomScanner.Instance;
            if (scanner == null) return;
            foreach (var c in scanner.GetComponents<MonoBehaviour>())
            {
                if (c is IGSplatProvider provider)
                {
                    _cachedGSplat = provider;
                    return;
                }
            }
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
