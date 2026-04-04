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
        private Model _model;
        private bool _loaded;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveAsync(ModelFileName, ct);
            _model = await Task.Run(() => ModelLoader.Load(path), ct);
            _worker = new Worker(_model, BackendType.GPUCompute);
            _loaded = true;
            await AsyncHelper.YieldFrame();
        }

        // ImageNet normalization matching rembg's u2netp preprocessing
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        internal async Task<Tensor<float>> InferAsync(Texture2D image, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("RembgModel not loaded");

            var input = PrepareInput(image);
            await InferenceScheduler.RunAsync(_worker, _model, ct, input);
            input.Dispose();

            var rawOutput = _worker.PeekOutput() as Tensor<float>;
            return MinMaxNormalize(rawOutput);
        }

        /// <summary>
        /// Matches rembg preprocessing exactly:
        /// 1. Resize to 320x320 (done by TextureConverter)
        /// 2. Normalize to [0, max] (divide by per-image max)
        /// 3. Subtract ImageNet mean, divide by ImageNet std
        /// </summary>
        private static Tensor<float> PrepareInput(Texture2D image)
        {
            var tensor = new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize));
            TextureConverter.ToTensor(image, tensor, new TextureTransform());

            var data = tensor.DownloadToArray();
            float maxVal = 0f;
            for (int i = 0; i < data.Length; i++)
                if (data[i] > maxVal) maxVal = data[i];
            if (maxVal < 1e-6f) maxVal = 1e-6f;

            int channelSize = InputSize * InputSize;
            for (int c = 0; c < 3; c++)
            {
                float mean = Mean[c], std = Std[c], inv = 1f / maxVal;
                int offset = c * channelSize;
                for (int i = 0; i < channelSize; i++)
                    data[offset + i] = (data[offset + i] * inv - mean) / std;
            }

            tensor.Upload(data);
            return tensor;
        }

        /// <summary>
        /// Min-max normalization on output mask, matching rembg's post-processing.
        /// </summary>
        private static Tensor<float> MinMaxNormalize(Tensor<float> raw)
        {
            var data = raw.DownloadToArray();
            float mi = float.MaxValue, ma = float.MinValue;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] < mi) mi = data[i];
                if (data[i] > ma) ma = data[i];
            }
            float range = ma - mi;
            if (range < 1e-8f) range = 1e-8f;
            for (int i = 0; i < data.Length; i++)
                data[i] = (data[i] - mi) / range;

            var result = new Tensor<float>(raw.shape);
            result.Upload(data);
            return result;
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
