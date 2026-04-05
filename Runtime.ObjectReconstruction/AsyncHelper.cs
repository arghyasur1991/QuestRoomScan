#if HAS_ONNXRUNTIME
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Async utilities for non-blocking operations. In Play mode, yields to the next
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
        /// When true, <see cref="YieldFrame"/> becomes a no-op.
        /// Set by editor code when maximum throughput is needed.
        /// </summary>
        internal static bool SuppressYields;

        /// <summary>
        /// Yields control back to Unity. In Play mode uses Task.Yield (defers to
        /// next frame via UnitySynchronizationContext). In Edit mode uses the
        /// plugged-in EditorApplication.delayCall-based yield.
        /// When <see cref="SuppressYields"/> is true, returns immediately.
        /// </summary>
        internal static async Task YieldFrame()
        {
            if (SuppressYields) return;
            if (EditModeYield != null)
            {
                await EditModeYield();
                return;
            }
            await Task.Yield();
        }

        /// <summary>
        /// Non-blocking GPU readback of a ComputeBuffer region to a new managed array.
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

        /// <summary>
        /// Non-blocking GPU readback into a pre-existing destination array (avoids allocation).
        /// Grows the destination if needed.
        /// </summary>
        internal static Task<float[]> ReadbackAsync(
            ComputeBuffer buffer, ref float[] destination, int count)
        {
            if (destination == null || destination.Length < count)
                destination = new float[count];

            var dest = destination;
            var tcs = new TaskCompletionSource<float[]>();
            int byteSize = count * sizeof(float);
            AsyncGPUReadback.Request(buffer, byteSize, 0, request =>
            {
                if (request.hasError)
                {
                    tcs.SetException(new InvalidOperationException("AsyncGPUReadback failed"));
                    return;
                }
                var native = request.GetData<float>();
                NativeArray<float>.Copy(native, dest, count);
                tcs.SetResult(dest);
            });
            return tcs.Task;
        }
    }
}
#endif
