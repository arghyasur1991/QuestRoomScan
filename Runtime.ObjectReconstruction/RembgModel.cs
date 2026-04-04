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

        private readonly BackendType _backend;
        private Worker _worker;
        private Model _model;
        private bool _loaded;

        internal RembgModel(BackendType backend = BackendType.GPUCompute) => _backend = backend;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveAsync(ModelFileName, ct);
            _model = await Task.Run(() => ModelLoader.Load(path), ct);
            _worker = new Worker(_model, _backend);
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

            bool cpuPath = _backend == BackendType.CPU;
            var input = cpuPath ? PrepareInputCpu(image) : PrepareInputGpu(image);
            await InferenceScheduler.RunAsync(_worker, _model, ct, input);
            input.Dispose();

            var rawOutput = _worker.PeekOutput() as Tensor<float>;
            return MinMaxNormalize(rawOutput);
        }

        /// <summary>
        /// GPU path: TextureConverter uploads to GPU, normalize in-place.
        /// </summary>
        private static Tensor<float> PrepareInputGpu(Texture2D image)
        {
            var tensor = new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize));
            TextureConverter.ToTensor(image, tensor, new TextureTransform());

            var data = tensor.DownloadToArray();
            NormalizeImageNet(data);
            tensor.Upload(data);
            return tensor;
        }

        /// <summary>
        /// CPU path: extract pixels manually, produce a CPU-backed tensor (zero GPU).
        /// Resizes via GPU Blit (texture op, not compute buffer) then reads back pixels.
        /// </summary>
        private static Tensor<float> PrepareInputCpu(Texture2D image)
        {
            // Resize to InputSize×InputSize using GPU blit (RenderTexture, not ComputeBuffer)
            var rt = RenderTexture.GetTemporary(InputSize, InputSize, 0, RenderTextureFormat.ARGB32);
            rt.filterMode = FilterMode.Bilinear;
            Graphics.Blit(image, rt);

            var resized = new Texture2D(InputSize, InputSize, TextureFormat.RGB24, false);
            RenderTexture.active = rt;
            resized.ReadPixels(new UnityEngine.Rect(0, 0, InputSize, InputSize), 0, 0);
            resized.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            var pixels = resized.GetPixels32();
            Object.Destroy(resized);

            int channelSize = InputSize * InputSize;
            var data = new float[3 * channelSize];
            for (int y = 0; y < InputSize; y++)
            for (int x = 0; x < InputSize; x++)
            {
                int texIdx = y * InputSize + x;
                int tensorY = InputSize - 1 - y;
                int idx = tensorY * InputSize + x;
                var c = pixels[texIdx];
                data[0 * channelSize + idx] = c.r / 255f;
                data[1 * channelSize + idx] = c.g / 255f;
                data[2 * channelSize + idx] = c.b / 255f;
            }

            NormalizeImageNet(data);
            return new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize), data);
        }

        private static void NormalizeImageNet(float[] data)
        {
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
        }

        /// <summary>
        /// Min-max normalization on output mask, matching rembg's post-processing.
        /// Returns CPU-backed tensor (DownloadToArray → new Tensor with data).
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

            return new Tensor<float>(raw.shape, data);
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
