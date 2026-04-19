using System;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Result returned by <see cref="RoomScanSession"/> operations.
    /// Contains the refined mesh, atlas texture, and the package ID used for persistence.
    /// </summary>
    public struct ScanResult
    {
        public Mesh Mesh;
        public Texture2D Atlas;
        public string PackageId;
    }

    /// <summary>
    /// High-level facade for game developers who need a simple scan → mesh → render workflow.
    /// Wraps <see cref="RoomScanner"/> and <see cref="RoomScanPersistence"/> into a
    /// minimal API with awaitable operations and a single result type.
    /// <para>
    /// Attach this component alongside <see cref="RoomScanner"/> or access via
    /// <see cref="Instance"/> after it initializes. The Setup Scene wizard's
    /// "Apply Game-Ready Setup" preset adds it automatically.
    /// </para>
    /// </summary>
    [RequireComponent(typeof(RoomScanner))]
    public class RoomScanSession : MonoBehaviour
    {
        public static RoomScanSession Instance { get; private set; }

        /// <summary>Raised each frame during scanning with the latest progress.</summary>
        public event Action<ScanProgress> ProgressUpdated;

        private RoomScanner _scanner;
        private RoomScanPersistence _persistence;

        private void Awake()
        {
            Instance = this;
            _scanner = GetComponent<RoomScanner>();
            _persistence = GetComponent<RoomScanPersistence>();
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        private void Update()
        {
            if (_scanner != null && _scanner.IsScanning)
                ProgressUpdated?.Invoke(_scanner.CurrentProgress);
        }

        /// <summary>Begins a new scan session. The room mesh builds in real-time as the user looks around.</summary>
        public void StartScan()
        {
            if (_scanner == null) { Logger.Error("RoomScanSession: RoomScanner not found"); return; }
            _scanner.StartScanning();
        }

        /// <summary>
        /// Stops scanning, runs on-device texture refinement (UV unwrap + atlas bake + simplification),
        /// saves to a permanent package, and releases heavy GPU resources (~400-500 MB).
        /// Returns a <see cref="ScanResult"/> with the game-ready mesh and atlas.
        /// </summary>
        public async Task<ScanResult> FinalizeScanAsync()
        {
            if (_scanner == null)
                throw new InvalidOperationException("RoomScanSession: RoomScanner not found");

            _scanner.StopScanning();

            if (!_scanner.HasTextureRefinementModule)
                throw new InvalidOperationException(
                    "RoomScanSession: TextureRefinement module required for FinalizeScanAsync");

            if (!_scanner.HasRefinedTexture)
            {
                var tcs = new TaskCompletionSource<bool>();
                void OnReady(Mesh _, Texture2D __) => tcs.TrySetResult(true);
                _scanner.RefinedMeshReady += OnReady;
                _scanner.StartTextureRefinement();

                var timeout = Task.Delay(TimeSpan.FromMinutes(5));
                var completed = await Task.WhenAny(tcs.Task, timeout);
                _scanner.RefinedMeshReady -= OnReady;

                if (completed == timeout)
                    throw new TimeoutException("Texture refinement timed out (5 min)");
            }

            bool saved = await _scanner.SaveScanAsync();
            if (!saved)
                Logger.Warning("RoomScanSession: save failed — result is in memory only");

            _scanner.ReleaseScanResources();

            return new ScanResult
            {
                Mesh = _scanner.RefinedMesh,
                Atlas = _scanner.RefinedAtlas,
                PackageId = _persistence?.ActivePackageId
            };
        }

        /// <summary>
        /// Loads only the refined mesh + atlas from a previously saved package.
        /// Skips TSDF reconstruction — typically completes in under 1 second.
        /// </summary>
        public async Task<ScanResult> LoadAsync(string packageId)
        {
            if (_scanner == null)
                throw new InvalidOperationException("RoomScanSession: RoomScanner not found");

            bool ok = await _scanner.LoadRefinedOnlyAsync(packageId);
            if (!ok)
                throw new InvalidOperationException($"Failed to load package: {packageId}");

            return new ScanResult
            {
                Mesh = _scanner.RefinedMesh,
                Atlas = _scanner.RefinedAtlas,
                PackageId = packageId
            };
        }

        /// <summary>
        /// Loads the most recently saved scan package. Convenience wrapper around <see cref="LoadAsync"/>.
        /// </summary>
        public async Task<ScanResult> LoadLatestAsync()
        {
            if (_persistence == null)
                throw new InvalidOperationException("RoomScanSession: RoomScanPersistence not found");

            var packages = _persistence.ListPackages();
            if (packages.Count == 0)
                throw new InvalidOperationException("No saved scan packages found");

            return await LoadAsync(packages[0].id);
        }

        /// <summary>Returns true if at least one saved scan package exists on disk.</summary>
        public bool HasSavedScan => _persistence != null && _persistence.HasAnyPackage();

        /// <summary>Whether a scan is currently in progress.</summary>
        public bool IsScanning => _scanner != null && _scanner.IsScanning;

        /// <summary>Releases heavy GPU resources. Called automatically by <see cref="FinalizeScanAsync"/>.</summary>
        public void ReleaseScanResources() => _scanner?.ReleaseScanResources();
    }
}
