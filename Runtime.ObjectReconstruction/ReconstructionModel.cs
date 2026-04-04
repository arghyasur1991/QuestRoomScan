#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the split TripoSR model (two halves). Supports two modes:
    /// <list type="bullet">
    /// <item><b>Preloaded</b> (editor): Both workers loaded once via <see cref="PreloadAsync"/>,
    ///   kept alive across runs. No per-run load/dispose overhead.</item>
    /// <item><b>Sequential</b> (Quest): Each half loaded, executed, and disposed independently
    ///   inside <see cref="RunAsync"/> to minimize peak GPU memory.</item>
    /// </list>
    /// </summary>
    internal sealed class ReconstructionModel : IDisposable
    {
        private const string Part1FileName = "ObjectReconstruction/triposr_part1.sentis";
        private const string Part2FileName = "ObjectReconstruction/triposr_part2.sentis";

        private readonly BackendType _backend;
        private Worker _worker1, _worker2;
        private Model _model1, _model2;
        private bool _preloaded;

        internal ReconstructionModel(BackendType backend = BackendType.GPUCompute) => _backend = backend;

        /// <summary>Load both model halves and keep workers alive for reuse.</summary>
        internal async Task PreloadAsync(CancellationToken ct)
        {
            if (_preloaded) return;

            string path1 = await ModelPathResolver.ResolveAsync(Part1FileName, ct);
            _model1 = await Task.Run(() => ModelLoader.Load(path1), ct);
            _worker1 = new Worker(_model1, _backend);

            string path2 = await ModelPathResolver.ResolveAsync(Part2FileName, ct);
            _model2 = await Task.Run(() => ModelLoader.Load(path2), ct);
            _worker2 = new Worker(_model2, _backend);

            _preloaded = true;
        }

        internal async Task<Tensor<float>> RunAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            return _preloaded
                ? await RunPreloadedAsync(preprocessed, ct)
                : await RunSequentialAsync(preprocessed, ct);
        }

        private async Task<Tensor<float>> RunPreloadedAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            await InferenceScheduler.RunAsync(_worker1, _model1, ct, preprocessed);

            Tensor encoderTmp = null;
            _worker1.CopyOutput("/Reshape_output_0", ref encoderTmp);
            var encoderStates = encoderTmp as Tensor<float>;

            Tensor hiddenTmp = null;
            _worker1.CopyOutput("/backbone/transformer_blocks.7/Add_2_output_0", ref hiddenTmp);
            var hiddenStates = hiddenTmp as Tensor<float>;

            _worker2.SetInput("/Reshape_output_0", encoderStates);
            _worker2.SetInput("/backbone/transformer_blocks.7/Add_2_output_0", hiddenStates);

            await InferenceScheduler.RunAsync(_worker2, _model2, ct);

            encoderStates.Dispose();
            hiddenStates.Dispose();

            Tensor sceneCodesTmp = null;
            _worker2.CopyOutput(0, ref sceneCodesTmp);
            return sceneCodesTmp as Tensor<float>;
        }

        private async Task<Tensor<float>> RunSequentialAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            Tensor<float> encoderStates;
            Tensor<float> hiddenStates;

            {
                string path = await ModelPathResolver.ResolveAsync(Part1FileName, ct);
                var model = await Task.Run(() => ModelLoader.Load(path), ct);
                using var worker = new Worker(model, _backend);
                await AsyncHelper.YieldFrame();

                await InferenceScheduler.RunAsync(worker, model, ct, preprocessed);

                Tensor encoderTmp = null;
                worker.CopyOutput("/Reshape_output_0", ref encoderTmp);
                encoderStates = encoderTmp as Tensor<float>;

                Tensor hiddenTmp = null;
                worker.CopyOutput("/backbone/transformer_blocks.7/Add_2_output_0", ref hiddenTmp);
                hiddenStates = hiddenTmp as Tensor<float>;
            }
            await AsyncHelper.YieldFrame();

            Tensor<float> sceneCodes;
            {
                string path = await ModelPathResolver.ResolveAsync(Part2FileName, ct);
                var model = await Task.Run(() => ModelLoader.Load(path), ct);
                using var worker = new Worker(model, _backend);
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

        public void Dispose()
        {
            _worker1?.Dispose();
            _worker2?.Dispose();
            _worker1 = null;
            _worker2 = null;
            _preloaded = false;
        }
    }
}
#endif
