#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the main TripoSR model (419M params). Loads a pre-quantized .sentis file
    /// from StreamingAssets. The forward pass is split over many frames via
    /// <see cref="Worker.ScheduleIterable"/>. Output stays on GPU — caller uses
    /// PeekOutput() to access the scene codes tensor without readback.
    /// </summary>
    internal sealed class ReconstructionModel : IDisposable
    {
        private const string ModelFileName = "ObjectReconstruction/triposr.sentis";

        private Worker _worker;
        private bool _loaded;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveAsync(ModelFileName, ct);
            var model = await Task.Run(() => ModelLoader.Load(path), ct);
            _worker = new Worker(model, BackendType.GPUCompute);
            _loaded = true;
            await AsyncHelper.YieldFrame();
        }

        /// <summary>
        /// Run the TripoSR forward pass on a preprocessed 512x512 image.
        /// Output stays on GPU — use PeekOutput() to get the scene codes tensor.
        /// </summary>
        internal async Task RunAsync(Tensor<float> preprocessed, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("ReconstructionModel not loaded");

            var budget = new AsyncHelper.FrameBudget();
            var it = _worker.ScheduleIterable(preprocessed);
            while (it.MoveNext())
            {
                ct.ThrowIfCancellationRequested();
                await budget.YieldIfNeeded();
            }
        }

        /// <summary>
        /// Returns the scene codes output tensor (shape 1,3,40,64,64).
        /// Valid until the next RunAsync call. Do NOT dispose this tensor.
        /// </summary>
        internal Tensor<float> PeekOutput()
        {
            return _worker.PeekOutput() as Tensor<float>;
        }

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
            _loaded = false;
        }
    }
}
#endif
