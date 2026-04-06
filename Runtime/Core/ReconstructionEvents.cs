using System;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Static event hub for reconstruction pipeline lifecycle.
    /// Fired by <c>ObjectReconstructionModule</c> (via InternalsVisibleTo),
    /// consumed by components like <see cref="CpuBoostBridge"/> in the core assembly.
    /// </summary>
    public static class ReconstructionEvents
    {
        public static event Action PipelineStarted;
        public static event Action PipelineFinished;

        internal static void FireStarted() => PipelineStarted?.Invoke();
        internal static void FireFinished() => PipelineFinished?.Invoke();
    }
}
