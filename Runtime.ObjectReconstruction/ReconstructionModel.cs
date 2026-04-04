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
    /// Worker.CopyOutput (GPU MemCopy, zero CPU readback). Each half is loaded and disposed
    /// independently so peak GPU memory is roughly halved.
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

                await InferenceScheduler.RunAsync(worker, model, ct, preprocessed);

                Tensor encoderStatesTmp = null;
                worker.CopyOutput("/Reshape_output_0", ref encoderStatesTmp);
                encoderStates = encoderStatesTmp as Tensor<float>;

                Tensor hiddenStatesTmp = null;
                worker.CopyOutput("/backbone/transformer_blocks.7/Add_2_output_0", ref hiddenStatesTmp);
                hiddenStates = hiddenStatesTmp as Tensor<float>;
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

                await InferenceScheduler.RunAsync(worker, model, ct);

                encoderStates.Dispose();
                hiddenStates.Dispose();

                Tensor sceneCodesTmp = null;
                worker.CopyOutput(0, ref sceneCodesTmp);
                sceneCodes = sceneCodesTmp as Tensor<float>;
            }
            await AsyncHelper.YieldFrame();

            return sceneCodes;
        }

        public void Dispose() { }
    }
}
#endif
