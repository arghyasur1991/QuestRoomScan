namespace Genesis.RoomScan
{
    /// <summary>
    /// Implement on optional <c>MonoBehaviour</c> components that extend <see cref="RoomScanner"/>.
    /// Modules are discovered automatically via <c>GetComponents&lt;IRoomScanModule&gt;</c> at startup
    /// and receive lifecycle callbacks from the scanner.
    /// </summary>
    public interface IRoomScanModule
    {
        /// <summary>Human-readable name shown in the inspector and logs.</summary>
        string ModuleName { get; }

        /// <summary>
        /// Called once during <see cref="RoomScanner.Start"/> after all core components
        /// have been initialised. Subscribe to scanner events here.
        /// </summary>
        void OnModuleInitialize(RoomScanner scanner);

        /// <summary>Called when scanning begins.</summary>
        void OnScanStarted() { }

        /// <summary>Called when scanning stops.</summary>
        void OnScanStopped() { }
    }
}
