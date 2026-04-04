#if HAS_AI_INFERENCE
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.InferenceEngine;
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

    /// <summary>
    /// Adaptive inference scheduler. Pre-scans model layers and classifies each as
    /// heavy (MatMul, Dense, Conv, Softmax, LayerNorm) or light (Reshape, Add, etc.).
    /// Heavy ops get exclusive frames + cooldown; light ops are batched aggressively.
    /// This maximizes FPS during inference by preventing GPU starvation for rendering.
    /// </summary>
    internal static class InferenceScheduler
    {
        /// <summary>Max light ops to batch in a single frame before yielding.</summary>
        internal static int LightOpBatchSize = 20;

        /// <summary>Extra frames to yield after a heavy op for GPU cooldown.</summary>
        internal static int HeavyOpCooldownFrames = 1;

        private static readonly HashSet<Type> HeavyOps = new()
        {
            typeof(MatMul),
            typeof(MatMul2D),
            typeof(Dense),
            typeof(DenseBatched),
            typeof(Conv),
            typeof(ConvTranspose),
            typeof(Softmax),
            typeof(LogSoftmax),
            typeof(LayerNormalization),
            typeof(RMSNormalization),
            typeof(InstanceNormalization),
            typeof(Einsum),
        };

        /// <summary>
        /// Runs inference with adaptive per-layer throttling.
        /// Pre-scans the model to classify ops, then batches light ops and yields
        /// extra frames after heavy ones.
        /// </summary>
        internal static async Task RunAsync(
            Worker worker, Model model, CancellationToken ct, Tensor input = null)
        {
            var schedule = BuildSchedule(model);
            int gpuIdx = 0;
            int lightAccum = 0;

            var it = input != null
                ? worker.ScheduleIterable(input)
                : worker.ScheduleIterable();

            while (it.MoveNext())
            {
                ct.ThrowIfCancellationRequested();

                bool heavy = gpuIdx < schedule.Length && schedule[gpuIdx];
                gpuIdx++;

                if (heavy)
                {
                    lightAccum = 0;
                    for (int f = 0; f <= HeavyOpCooldownFrames; f++)
                        await AsyncHelper.YieldFrame();
                }
                else
                {
                    lightAccum++;
                    if (lightAccum >= LightOpBatchSize)
                    {
                        lightAccum = 0;
                        await AsyncHelper.YieldFrame();
                    }
                }
            }
        }

        /// <returns>Boolean array — true if the GPU layer at that index is heavy.</returns>
        private static bool[] BuildSchedule(Model model)
        {
            var schedule = new List<bool>();
            foreach (var layer in model.layers)
            {
                bool isHeavy = HeavyOps.Contains(layer.GetType());
                schedule.Add(isHeavy);
            }
            return schedule.ToArray();
        }
    }
}
#endif
