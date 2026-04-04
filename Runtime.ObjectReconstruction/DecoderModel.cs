#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the NeRF decoder MLP (~170KB). Takes (N, 120) triplane features and outputs
    /// (N, 4) [density, r, g, b]. Loaded as FP32 .sentis from StreamingAssets (too small
    /// to benefit from quantization).
    /// </summary>
    internal sealed class DecoderModel : IDisposable
    {
        private const string ModelFileName = "ObjectReconstruction/nerf_decoder.sentis";

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
        /// Run the decoder on a chunk of triplane features. Input shape: (N, 120), output: (N, 4).
        /// </summary>
        internal async Task<Tensor<float>> InferAsync(
            Tensor<float> features, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("DecoderModel not loaded");

            var budget = new AsyncHelper.FrameBudget();
            var it = _worker.ScheduleIterable(features);
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
