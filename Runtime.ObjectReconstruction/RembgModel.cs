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
    /// Wraps the u2netp background removal model. Loads a .sentis file from
    /// StreamingAssets and runs inference via <see cref="Worker"/> on CPU to avoid
    /// GPU tensor indexer bugs in Sentis with u2netp's Resize/MaxPool ops.
    /// </summary>
    internal sealed class RembgModel : IDisposable
    {
        private const string ModelFileName = "ObjectReconstruction/u2netp.sentis";
        private const int InputSize = 320;
        private const int RembgLayersPerFrame = 40;

        private Worker _worker;
        private bool _loaded;

        internal async Task LoadAsync(CancellationToken ct)
        {
            if (_loaded) return;

            string path = await ModelPathResolver.ResolveAsync(ModelFileName, ct);
            var model = ModelLoader.Load(path);
            _worker = new Worker(model, BackendType.CPU);
            _loaded = true;
            await Task.Yield();
        }

        /// <summary>
        /// Run u2netp on the input texture, returning a 320x320 alpha mask tensor.
        /// Populates input tensor manually via Upload() to avoid TextureConverter's
        /// internal tensor indexer which triggers Sentis axis-bounds errors.
        /// </summary>
        internal async Task<Tensor<float>> InferAsync(Texture2D image, CancellationToken ct)
        {
            if (!_loaded)
                throw new InvalidOperationException("RembgModel not loaded");

            using var input = TextureToTensorManual(image);

            var it = _worker.ScheduleIterable(input);
            int steps = 0;
            while (it.MoveNext())
            {
                ct.ThrowIfCancellationRequested();
                if (++steps % RembgLayersPerFrame == 0) await Task.Yield();
            }

            var output = _worker.PeekOutput() as Tensor<float>;
            return output.ReadbackAndClone();
        }

        /// <summary>
        /// Converts a Texture2D to a [1,3,H,W] float tensor using raw pixel data
        /// and Upload(), bypassing TextureConverter entirely.
        /// </summary>
        private static Tensor<float> TextureToTensorManual(Texture2D image)
        {
            var rt = RenderTexture.GetTemporary(InputSize, InputSize, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(image, rt);
            RenderTexture.active = rt;

            var resized = new Texture2D(InputSize, InputSize, TextureFormat.RGBA32, false);
            resized.ReadPixels(new Rect(0, 0, InputSize, InputSize), 0, 0);
            resized.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            var pixels = resized.GetPixels32();
            UnityEngine.Object.Destroy(resized);

            int chw = 3 * InputSize * InputSize;
            var data = new float[chw];
            for (int y = 0; y < InputSize; y++)
            {
                for (int x = 0; x < InputSize; x++)
                {
                    int flippedY = InputSize - 1 - y;
                    var p = pixels[flippedY * InputSize + x];
                    int pixIdx = y * InputSize + x;
                    data[0 * InputSize * InputSize + pixIdx] = p.r / 255f;
                    data[1 * InputSize * InputSize + pixIdx] = p.g / 255f;
                    data[2 * InputSize * InputSize + pixIdx] = p.b / 255f;
                }
            }

            var tensor = new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize));
            tensor.Upload(data);
            return tensor;
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
