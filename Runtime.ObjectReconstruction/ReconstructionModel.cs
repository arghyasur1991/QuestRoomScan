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
    /// (Uint8 ~400MB) from StreamingAssets. The forward pass is split over many frames
    /// via <see cref="Worker.ScheduleIterable"/> to avoid GPU contention with VR rendering.
    /// </summary>
    internal sealed class ReconstructionModel : IDisposable
    {
        private static readonly string[] ModelVariants =
        {
            "ObjectReconstruction/triposr_fp32.sentis",
            "ObjectReconstruction/triposr_fp16.sentis",
            "ObjectReconstruction/triposr_uint8.sentis",
        };

        private Worker _worker;
        private bool _loaded;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveFirstAsync(ModelVariants, ct);
            Logger.Info($"[ReconstructionModel] Loading: {System.IO.Path.GetFileName(path)}");
            var model = await Task.Run(() => ModelLoader.Load(path), ct);
            _worker = new Worker(model, BackendType.GPUCompute);
            _loaded = true;
            await AsyncHelper.YieldFrame();
        }

        /// <summary>
        /// Run the TripoSR forward pass on a preprocessed 512x512 image.
        /// Returns scene_codes tensor of shape (1, 3, 40, 64, 64).
        /// </summary>
        internal async Task<Tensor<float>> InferAsync(
            Tensor<float> preprocessed, CancellationToken ct)
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

            var output = _worker.PeekOutput() as Tensor<float>;
            return output.ReadbackAndClone();
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
