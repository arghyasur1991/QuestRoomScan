#if HAS_AI_INFERENCE
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Async utilities for non-blocking inference. In Play mode, yields to the next
    /// frame via the Unity synchronization context. In Edit mode, uses a pluggable
    /// delegate (set by editor code) to yield via EditorApplication.delayCall.
    /// </summary>
    internal static class AsyncHelper
    {
        /// <summary>
        /// Editor code sets this to provide real edit-mode yielding via
        /// EditorApplication.delayCall. When null, falls back to Task.Yield.
        /// </summary>
        internal static Func<Task> EditModeYield;

        /// <summary>
        /// Yields control back to Unity. In Play mode uses Task.Yield (defers to
        /// next frame via UnitySynchronizationContext). In Edit mode uses the
        /// plugged-in EditorApplication.delayCall-based yield.
        /// </summary>
        internal static async Task YieldFrame()
        {
            if (EditModeYield != null)
            {
                await EditModeYield();
                return;
            }
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

        /// <summary>
        /// Non-blocking GPU readback of a ComputeBuffer region to managed array.
        /// Returns a Task that completes when the GPU data is available on CPU.
        /// </summary>
        internal static Task<T[]> ReadbackAsync<T>(ComputeBuffer buffer, int count) where T : struct
        {
            var tcs = new TaskCompletionSource<T[]>();
            int byteSize = count * Marshal.SizeOf<T>();
            AsyncGPUReadback.Request(buffer, byteSize, 0, request =>
            {
                if (request.hasError)
                {
                    tcs.SetException(new InvalidOperationException("AsyncGPUReadback failed"));
                    return;
                }
                var native = request.GetData<T>();
                var result = new T[count];
                NativeArray<T>.Copy(native, result, count);
                tcs.SetResult(result);
            });
            return tcs.Task;
        }
    }
}
#endif
