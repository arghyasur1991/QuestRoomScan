#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the u2netp background removal model. Loads a .sentis file from
    /// StreamingAssets and runs GPU inference via <see cref="Worker"/>.
    /// </summary>
    internal sealed class RembgModel : IDisposable
    {
        private const string ModelFileName = "ObjectReconstruction/u2netp.sentis";
        private const int InputSize = 320;
        private const int RembgLayersPerFrame = 20;

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

        internal async Task<Tensor<float>> InferAsync(Texture2D image, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("RembgModel not loaded");

            using var input = new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize));
            TextureConverter.ToTensor(image, input, new TextureTransform());

            var budget = new AsyncHelper.FrameBudget();
            var it = _worker.ScheduleIterable(input);
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
