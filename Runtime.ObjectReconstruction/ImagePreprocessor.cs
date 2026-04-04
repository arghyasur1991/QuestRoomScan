#if HAS_AI_INFERENCE
using System.Threading.Tasks;
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
        /// Replicates the Python TripoSR preprocessing exactly:
        /// 1. Apply mask to get RGBA
        /// 2. Crop to foreground bounding box (from alpha)
        /// 3. Pad to square
        /// 4. Pad to achieve foreground ratio (fg occupies ratio% of frame)
        /// 5. Composite fg*alpha + gray*(1-alpha)
        /// 6. Resize to 512x512
        /// CPU pixel work is offloaded to a background thread.
        /// </summary>
        internal static async Task<Tensor<float>> ApplyMaskAndCompositeAsync(
            Texture2D image, Tensor<float> alphaMask, float foregroundRatio)
        {
            const int outputSize = 512;

            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            int maskW = alphaMask.shape[alphaMask.shape.rank - 1];
            int maskH = alphaMask.shape[alphaMask.shape.rank - 2];
            var maskData = alphaMask.DownloadToArray();

            var resultPixels = await Task.Run(() =>
                BuildComposite(srcPixels, srcW, srcH, maskData, maskW, maskH, foregroundRatio, outputSize));

            await AsyncHelper.YieldFrame();

            var result = new Texture2D(outputSize, outputSize, TextureFormat.RGB24, false);
            result.SetPixels32(resultPixels);
            result.Apply();

            var tensor = new Tensor<float>(new TensorShape(1, 3, outputSize, outputSize));
            TextureConverter.ToTensor(result, tensor, new TextureTransform());
            SafeDestroy(result);

            return tensor;
        }

        /// <summary>
        /// Pure CPU work matching Python's resize_foreground + alpha composite.
        /// </summary>
        private static Color32[] BuildComposite(
            Color32[] srcPixels, int srcW, int srcH,
            float[] maskData, int maskW, int maskH,
            float ratio, int outSize)
        {
            // Step 1: build per-pixel alpha from mask (resample mask to src resolution)
            var alpha = new float[srcW * srcH];
            int minX = srcW, minY = srcH, maxX = 0, maxY = 0;

            for (int y = 0; y < srcH; y++)
            {
                for (int x = 0; x < srcW; x++)
                {
                    float mx = (float)x / srcW * maskW;
                    float my = (float)y / srcH * maskH;
                    int mi = Mathf.Clamp((int)my, 0, maskH - 1) * maskW +
                             Mathf.Clamp((int)mx, 0, maskW - 1);
                    float a = Mathf.Clamp01(maskData[mi]);
                    alpha[y * srcW + x] = a;

                    if (a > 0.01f)
                    {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }

            if (maxX <= minX || maxY <= minY)
            {
                // No foreground found — return gray
                var gray = new Color32[outSize * outSize];
                byte g = (byte)(0.5f * 255);
                for (int i = 0; i < gray.Length; i++)
                    gray[i] = new Color32(g, g, g, 255);
                return gray;
            }

            // Step 2: crop to foreground bbox
            int cropW = maxX - minX + 1;
            int cropH = maxY - minY + 1;

            // Step 3: pad to square
            int sqSize = Mathf.Max(cropW, cropH);
            int padX0 = (sqSize - cropW) / 2;
            int padY0 = (sqSize - cropH) / 2;

            // Step 4: pad to achieve foreground ratio
            int finalSize = Mathf.CeilToInt(sqSize / ratio);
            int outerPadX = (finalSize - sqSize) / 2;
            int outerPadY = (finalSize - sqSize) / 2;

            // Step 5: composite onto gray at output resolution
            var result = new Color32[outSize * outSize];
            byte grayByte = (byte)(0.5f * 255);
            for (int i = 0; i < result.Length; i++)
                result[i] = new Color32(grayByte, grayByte, grayByte, 255);

            float scale = (float)finalSize / outSize;

            for (int oy = 0; oy < outSize; oy++)
            {
                for (int ox = 0; ox < outSize; ox++)
                {
                    // Map output pixel back to the padded foreground coordinate system
                    float px = ox * scale - outerPadX - padX0;
                    float py = oy * scale - outerPadY - padY0;

                    int srcX = Mathf.FloorToInt(px) + minX;
                    int srcY = Mathf.FloorToInt(py) + minY;

                    if (srcX < 0 || srcX >= srcW || srcY < 0 || srcY >= srcH)
                        continue;

                    int si = srcY * srcW + srcX;
                    float a = alpha[si];
                    if (a < 0.001f) continue;

                    var s = srcPixels[si];
                    byte r = (byte)(s.r * a + grayByte * (1f - a));
                    byte g2 = (byte)(s.g * a + grayByte * (1f - a));
                    byte b = (byte)(s.b * a + grayByte * (1f - a));
                    result[oy * outSize + ox] = new Color32(r, g2, b, 255);
                }
            }

            return result;
        }

        private static void SafeDestroy(Object obj)
        {
            if (obj == null) return;
#if UNITY_EDITOR
            if (!Application.isPlaying)
                Object.DestroyImmediate(obj);
            else
#endif
                Object.Destroy(obj);
        }
    }
}
#endif
