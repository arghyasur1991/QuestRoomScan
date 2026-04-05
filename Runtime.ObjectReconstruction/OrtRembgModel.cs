#if HAS_ONNXRUNTIME
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the u2netp background removal model via ONNX Runtime.
    /// Runs inference on a background thread via Task.Run.
    /// </summary>
    internal sealed class OrtRembgModel : OrtModelBase
    {
        private const string ModelFileName = "ObjectReconstruction/u2netp.onnx";
        private const int InputSize = 320;

        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        internal async Task LoadAsync(
            ExecutionProvider ep, bool mobileOptimized, CancellationToken ct)
        {
            await LoadSessionAsync(ModelFileName, ep, mobileOptimized, ct);
        }

        /// <returns>Min-max normalized mask as float[] (320x320).</returns>
        internal async Task<float[]> InferAsync(Texture2D image, CancellationToken ct)
        {
            if (!IsLoaded)
                throw new InvalidOperationException("OrtRembgModel not loaded");

            var inputData = PrepareInput(image);
            var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, InputSize, InputSize });
            LoadInput(tensor);

            using var results = await RunDisposable();
            var output = results.First().AsTensor<float>();
            var raw = output.ToArray();
            return MinMaxNormalize(raw);
        }

        private static float[] PrepareInput(Texture2D image)
        {
            var rt = RenderTexture.GetTemporary(InputSize, InputSize, 0, RenderTextureFormat.ARGB32);
            rt.filterMode = FilterMode.Bilinear;
            Graphics.Blit(image, rt);

            var resized = new Texture2D(InputSize, InputSize, TextureFormat.RGB24, false);
            RenderTexture.active = rt;
            resized.ReadPixels(new Rect(0, 0, InputSize, InputSize), 0, 0);
            resized.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            var pixels = resized.GetPixels32();
            SafeDestroy(resized);

            int channelSize = InputSize * InputSize;
            var data = new float[3 * channelSize];

            float maxVal = 0f;
            for (int y = 0; y < InputSize; y++)
            for (int x = 0; x < InputSize; x++)
            {
                int texIdx = y * InputSize + x;
                int tensorY = InputSize - 1 - y;
                int idx = tensorY * InputSize + x;
                var c = pixels[texIdx];
                float r = c.r / 255f;
                float g = c.g / 255f;
                float b = c.b / 255f;
                data[0 * channelSize + idx] = r;
                data[1 * channelSize + idx] = g;
                data[2 * channelSize + idx] = b;
                float m = Mathf.Max(r, Mathf.Max(g, b));
                if (m > maxVal) maxVal = m;
            }

            if (maxVal < 1e-6f) maxVal = 1e-6f;
            float inv = 1f / maxVal;
            for (int ch = 0; ch < 3; ch++)
            {
                float mean = Mean[ch], std = Std[ch];
                int offset = ch * channelSize;
                for (int i = 0; i < channelSize; i++)
                    data[offset + i] = (data[offset + i] * inv - mean) / std;
            }

            return data;
        }

        private static float[] MinMaxNormalize(float[] data)
        {
            float mi = float.MaxValue, ma = float.MinValue;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] < mi) mi = data[i];
                if (data[i] > ma) ma = data[i];
            }
            float range = ma - mi;
            if (range < 1e-8f) range = 1e-8f;

            var result = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
                result[i] = (data[i] - mi) / range;
            return result;
        }

        private static void SafeDestroy(UnityEngine.Object obj)
        {
            if (obj == null) return;
#if UNITY_EDITOR
            if (!Application.isPlaying)
                UnityEngine.Object.DestroyImmediate(obj);
            else
#endif
                UnityEngine.Object.Destroy(obj);
        }
    }
}
#endif
