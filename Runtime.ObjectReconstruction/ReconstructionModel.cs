#if HAS_AI_INFERENCE
using System;
using System.IO;
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
        private const string ModelFileName = "ObjectReconstruction/triposr_uint8.sentis";
        private const int InputSize = 512;

        private Worker _worker;
        private bool _loaded;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveAsync(ModelFileName, ct);
            var model = ModelLoader.Load(path);
            _worker = new Worker(model, BackendType.GPUCompute);
            _loaded = true;
            await Task.Yield();
        }

        /// <summary>
        /// Run the TripoSR forward pass on a preprocessed 512x512 image.
        /// Returns scene_codes tensor of shape (1, 3, 40, 64, 64).
        /// </summary>
        internal async Task<Tensor<float>> InferAsync(
            Tensor<float> preprocessed, int layersPerFrame, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("ReconstructionModel not loaded");

            var it = _worker.ScheduleIterable(preprocessed);
            int steps = 0;
            while (it.MoveNext())
            {
                ct.ThrowIfCancellationRequested();
                if (++steps % layersPerFrame == 0) await Task.Yield();
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
