using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Temporarily boosts the Quest CPU to <c>Boost</c> level during reconstruction
    /// inference, then restores the previous level. Prevents thermal governor from
    /// downclocking mid-inference, which causes frame jitter.
    /// </summary>
    public class CpuBoostBridge : MonoBehaviour
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        private OVRManager.ProcessorPerformanceLevel _savedCpuLevel;
        private bool _boosted;

        void OnEnable()
        {
            ReconstructionEvents.PipelineStarted += OnPipelineStarted;
            ReconstructionEvents.PipelineFinished += OnPipelineFinished;
        }

        void OnDisable()
        {
            ReconstructionEvents.PipelineStarted -= OnPipelineStarted;
            ReconstructionEvents.PipelineFinished -= OnPipelineFinished;
            RestoreCpuLevel();
        }

        private void OnPipelineStarted()
        {
            _savedCpuLevel = OVRManager.suggestedCpuPerfLevel;
            OVRManager.suggestedCpuPerfLevel = OVRManager.ProcessorPerformanceLevel.Boost;
            _boosted = true;
            Logger.Info($"[CpuBoost] CPU boosted for reconstruction (was {_savedCpuLevel})");
        }

        private void OnPipelineFinished()
        {
            RestoreCpuLevel();
        }

        private void RestoreCpuLevel()
        {
            if (!_boosted) return;
            OVRManager.suggestedCpuPerfLevel = _savedCpuLevel;
            _boosted = false;
            Logger.Info($"[CpuBoost] CPU restored to {_savedCpuLevel}");
        }
#endif
    }
}
