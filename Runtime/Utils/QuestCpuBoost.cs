using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Static API for temporarily boosting Quest CPU/GPU performance levels.
    /// Call <see cref="Begin"/> before heavy work, <see cref="End"/> when done.
    /// No-op on non-Android platforms.
    /// </summary>
    public static class QuestCpuBoost
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        private static OVRManager.ProcessorPerformanceLevel _savedCpu;
        private static OVRManager.ProcessorPerformanceLevel _savedGpu;
        private static bool _active;

        public static void Begin()
        {
            if (_active) return;
            _savedCpu = OVRManager.suggestedCpuPerfLevel;
            _savedGpu = OVRManager.suggestedGpuPerfLevel;
            OVRManager.suggestedCpuPerfLevel = OVRManager.ProcessorPerformanceLevel.Boost;
            _active = true;
            Debug.Log($"[QuestCpuBoost] Boosted (was CPU={_savedCpu} GPU={_savedGpu})");
        }

        public static void End()
        {
            if (!_active) return;
            OVRManager.suggestedCpuPerfLevel = _savedCpu;
            OVRManager.suggestedGpuPerfLevel = _savedGpu;
            _active = false;
            Debug.Log($"[QuestCpuBoost] Restored CPU={_savedCpu} GPU={_savedGpu}");
        }
#else
        public static void Begin() { }
        public static void End() { }
#endif
    }
}
