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
    /// (N, 4) [density, r, g, b]. Output stays on GPU — caller uses PeekOutputBuffer()
    /// to access the ComputeBuffer for downstream GPU compute.
    /// </summary>
    internal sealed class DecoderModel : IDisposable
    {
        private const string ModelFileName = "ObjectReconstruction/nerf_decoder.sentis";

        private readonly BackendType _backend;
        private Worker _worker;
        private Model _model;
        private bool _loaded;

        internal DecoderModel(BackendType backend = BackendType.GPUCompute) => _backend = backend;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveAsync(ModelFileName, ct);
            _model = await Task.Run(() => ModelLoader.Load(path), ct);
            _worker = new Worker(_model, _backend);
            _loaded = true;
            await AsyncHelper.YieldFrame();
        }

        /// <summary>
        /// Run the decoder on a chunk of triplane features. Input shape: (N, 120).
        /// Output stays on GPU — use PeekOutputBuffer() to access it.
        /// </summary>
        internal async Task RunAsync(Tensor<float> features, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("DecoderModel not loaded");

            await InferenceScheduler.RunAsync(_worker, _model, ct, features);
        }

        /// <summary>
        /// Returns the ComputeBuffer backing the decoder's output tensor.
        /// Valid until the next RunAsync call. Do NOT release this buffer.
        /// </summary>
        internal ComputeBuffer PeekOutputBuffer()
        {
            var output = _worker.PeekOutput() as Tensor<float>;
            return ComputeTensorData.Pin(output).buffer;
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
