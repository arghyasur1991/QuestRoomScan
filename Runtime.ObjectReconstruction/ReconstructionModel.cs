#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the split TripoSR model (two halves). Part 1 runs the image encoder + decoder
    /// blocks 0-7, Part 2 runs blocks 8-15 + post-processor. Between the two halves, only
    /// the hidden state (~12 MB) and encoder features (~3 MB) are transferred via
    /// ReadbackAndClone (~15 MB total, a few ms). Each half is loaded and disposed independently so peak GPU
    /// memory is roughly halved.
    /// </summary>
    internal sealed class ReconstructionModel : IDisposable
    {
        private const string Part1FileName = "ObjectReconstruction/triposr_part1.sentis";
        private const string Part2FileName = "ObjectReconstruction/triposr_part2.sentis";

        internal async Task<Tensor<float>> RunAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            Tensor<float> encoderStates;
            Tensor<float> hiddenStates;

            // --- Part 1: image -> encoder_hidden_states + hidden_states ---
            {
                string path = await ModelPathResolver.ResolveAsync(Part1FileName, ct);
                var model = await Task.Run(() => ModelLoader.Load(path), ct);
                using var worker = new Worker(model, BackendType.GPUCompute);
                await AsyncHelper.YieldFrame();

                var budget = new AsyncHelper.FrameBudget();
                var it = worker.ScheduleIterable(preprocessed);
                while (it.MoveNext())
                {
                    ct.ThrowIfCancellationRequested();
                    await budget.YieldIfNeeded();
                }

                encoderStates = (worker.PeekOutput("/Reshape_output_0") as Tensor<float>).ReadbackAndClone();
                hiddenStates = (worker.PeekOutput("/backbone/transformer_blocks.7/Add_2_output_0") as Tensor<float>).ReadbackAndClone();
            }
            await AsyncHelper.YieldFrame();

            // --- Part 2: (encoder_states, hidden_states) -> scene_codes ---
            Tensor<float> sceneCodes;
            {
                string path = await ModelPathResolver.ResolveAsync(Part2FileName, ct);
                var model = await Task.Run(() => ModelLoader.Load(path), ct);
                using var worker = new Worker(model, BackendType.GPUCompute);
                await AsyncHelper.YieldFrame();

                worker.SetInput("/Reshape_output_0", encoderStates);
                worker.SetInput("/backbone/transformer_blocks.7/Add_2_output_0", hiddenStates);

                var budget = new AsyncHelper.FrameBudget();
                var it = worker.ScheduleIterable();
                while (it.MoveNext())
                {
                    ct.ThrowIfCancellationRequested();
                    await budget.YieldIfNeeded();
                }

                encoderStates.Dispose();
                hiddenStates.Dispose();

                sceneCodes = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
            }
            await AsyncHelper.YieldFrame();

            return sceneCodes;
        }

        public void Dispose() { }
    }
}
#endif
