#if HAS_AI_INFERENCE
using System;
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
        /// Tracks dispatched ops and yields after a fixed count per frame.
        /// GPU dispatches are near-instant on CPU, so time-based budgeting
        /// doesn't work — it queues hundreds of layers before yielding,
        /// saturating the GPU and starving rendering. Op-counting ensures
        /// the GPU gets regular breaks for rendering.
        /// </summary>
        internal sealed class FrameBudget
        {
            private int _opsThisFrame;
            private readonly int _maxOpsPerFrame;

            /// <param name="maxOpsPerFrame">
            /// Max GPU ops to dispatch before yielding a frame. Lower values
            /// give smoother FPS but longer total inference time.
            /// At 72 Hz with ~3000 TripoSR layers, 4 ops/frame ≈ 10s inference.
            /// </param>
            internal FrameBudget(int maxOpsPerFrame = 4) => _maxOpsPerFrame = maxOpsPerFrame;

            /// <summary>
            /// Call after each dispatched op. Yields a frame once the budget is hit.
            /// </summary>
            internal async Task YieldIfNeeded()
            {
                if (++_opsThisFrame < _maxOpsPerFrame) return;
                _opsThisFrame = 0;
                await YieldFrame();
            }

            internal void Reset() => _opsThisFrame = 0;
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
