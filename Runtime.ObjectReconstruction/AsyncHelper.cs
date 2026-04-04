#if HAS_AI_INFERENCE
using System.Diagnostics;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Async utilities for non-blocking inference. In Play mode, yields to the next
    /// frame via the Unity synchronization context. In Edit mode (where Task.Yield is
    /// a no-op), uses Task.Delay to actually release the main thread.
    /// </summary>
    internal static class AsyncHelper
    {
        /// <summary>
        /// Yields control back to Unity. Works in both Play mode and Edit mode.
        /// </summary>
        internal static async Task YieldFrame()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                await Task.Delay(1);
                return;
            }
#endif
            await Task.Yield();
        }

        /// <summary>
        /// Tracks elapsed time and yields when the frame budget is exceeded.
        /// Use one instance per async operation to avoid stalling the main thread.
        /// </summary>
        internal sealed class FrameBudget
        {
            private readonly Stopwatch _sw = Stopwatch.StartNew();
            private readonly int _budgetMs;

            /// <param name="budgetMs">
            /// Max milliseconds to run before yielding. 8ms default keeps well
            /// under a single VR frame (11ms at 90Hz).
            /// </param>
            internal FrameBudget(int budgetMs = 8) => _budgetMs = budgetMs;

            /// <summary>
            /// Call after each unit of work. Yields and resets the timer only when
            /// the budget is exceeded, avoiding unnecessary yield overhead.
            /// </summary>
            internal async Task YieldIfNeeded()
            {
                if (_sw.ElapsedMilliseconds < _budgetMs) return;
                await YieldFrame();
                _sw.Restart();
            }

            internal void Reset() => _sw.Restart();
        }
    }
}
#endif
