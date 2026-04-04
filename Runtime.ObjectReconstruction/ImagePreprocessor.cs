#if HAS_AI_INFERENCE
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Applies background removal mask and composites onto gray (0.5) background,
    /// matching the TripoSR training pipeline preprocessing.
    /// </summary>
    internal static class ImagePreprocessor
    {
        /// <summary>
        /// Composites the foreground (using alpha mask) onto a gray (0.5, 0.5, 0.5) background,
        /// applies resize_foreground (centers foreground at given ratio), and returns
        /// a 512x512 Tensor suitable for the reconstruction model.
        /// </summary>
        internal static Tensor<float> ApplyMaskAndComposite(
            Texture2D image, Tensor<float> alphaMask, float foregroundRatio)
        {
            const int outputSize = 512;
            const float grayVal = 0.5f;

            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            int maskW = alphaMask.shape[alphaMask.shape.length - 1];
            int maskH = alphaMask.shape[alphaMask.shape.length - 2];
            var maskData = alphaMask.DownloadToArray();

            var composite = new Texture2D(srcW, srcH, TextureFormat.RGB24, false);
            var outPixels = new Color32[srcW * srcH];

            for (int y = 0; y < srcH; y++)
            {
                for (int x = 0; x < srcW; x++)
                {
                    int srcIdx = y * srcW + x;
                    float mx = (float)x / srcW * maskW;
                    float my = (float)y / srcH * maskH;
                    int mi = Mathf.Clamp((int)my, 0, maskH - 1) * maskW +
                             Mathf.Clamp((int)mx, 0, maskW - 1);
                    float alpha = Mathf.Clamp01(maskData[mi]);

                    var src = srcPixels[srcIdx];
                    byte r = (byte)(src.r * alpha + grayVal * 255 * (1 - alpha));
                    byte g = (byte)(src.g * alpha + grayVal * 255 * (1 - alpha));
                    byte b = (byte)(src.b * alpha + grayVal * 255 * (1 - alpha));
                    outPixels[srcIdx] = new Color32(r, g, b, 255);
                }
            }

            composite.SetPixels32(outPixels);
            composite.Apply();

            var resized = ResizeForeground(composite, foregroundRatio, outputSize);
            Object.Destroy(composite);

            var tensor = TextureToTensorManual(resized, outputSize);
            Object.Destroy(resized);

            return tensor;
        }

        private static Tensor<float> TextureToTensorManual(Texture2D tex, int size)
        {
            var pixels = tex.GetPixels32();
            int chw = 3 * size * size;
            var data = new float[chw];
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    int flippedY = size - 1 - y;
                    var p = pixels[flippedY * size + x];
                    int pixIdx = y * size + x;
                    data[0 * size * size + pixIdx] = p.r / 255f;
                    data[1 * size * size + pixIdx] = p.g / 255f;
                    data[2 * size * size + pixIdx] = p.b / 255f;
                }
            }

            var tensor = new Tensor<float>(new TensorShape(1, 3, size, size));
            tensor.Upload(data);
            return tensor;
        }

        private static Texture2D ResizeForeground(Texture2D src, float ratio, int outputSize)
        {
            int fgSize = Mathf.RoundToInt(outputSize * ratio);
            int pad = (outputSize - fgSize) / 2;

            var rt = RenderTexture.GetTemporary(fgSize, fgSize, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(src, rt);

            var result = new Texture2D(outputSize, outputSize, TextureFormat.RGB24, false);
            var grayPixels = new Color32[outputSize * outputSize];
            byte gray = (byte)(0.5f * 255);
            for (int i = 0; i < grayPixels.Length; i++)
                grayPixels[i] = new Color32(gray, gray, gray, 255);
            result.SetPixels32(grayPixels);

            RenderTexture.active = rt;
            var fgTex = new Texture2D(fgSize, fgSize, TextureFormat.RGB24, false);
            fgTex.ReadPixels(new Rect(0, 0, fgSize, fgSize), 0, 0);
            fgTex.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            var fgPixels = fgTex.GetPixels32();
            for (int y = 0; y < fgSize; y++)
            {
                for (int x = 0; x < fgSize; x++)
                {
                    int dstX = pad + x;
                    int dstY = pad + y;
                    if (dstX < outputSize && dstY < outputSize)
                        grayPixels[dstY * outputSize + dstX] = fgPixels[y * fgSize + x];
                }
            }

            result.SetPixels32(grayPixels);
            result.Apply();
            Object.Destroy(fgTex);

            return result;
        }
    }
}
#endif
