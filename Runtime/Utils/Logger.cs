namespace Genesis.RoomScan
{
    /// <summary>
    /// Log severity levels for the RoomScan package.
    /// Set via <see cref="RoomScanner"/> inspector or <c>Logger.Level</c> at runtime.
    /// </summary>
    public enum LogLevel
    {
        Verbose,
        Info,
        Warning,
        Error,
        None
    }

    /// <summary>
    /// Centralised logger for the Genesis RoomScan package.
    /// All internal logging routes through here so consumers can
    /// control verbosity with a single <see cref="Level"/> knob.
    /// </summary>
    internal static class Logger
    {
        public static LogLevel Level { get; set; } = LogLevel.Info;

        internal static void Verbose(string msg)
        {
            if (Level <= LogLevel.Verbose)
                UnityEngine.Debug.Log($"[RoomScan] {msg}");
        }

        internal static void Info(string msg)
        {
            if (Level <= LogLevel.Info)
                UnityEngine.Debug.Log($"[RoomScan] {msg}");
        }

        internal static void Warning(string msg)
        {
            if (Level <= LogLevel.Warning)
                UnityEngine.Debug.LogWarning($"[RoomScan] {msg}");
        }



        internal static void Error(string msg)
        {
            if (Level <= LogLevel.Error)
                UnityEngine.Debug.LogError($"[RoomScan] {msg}");
        }
    }
}
