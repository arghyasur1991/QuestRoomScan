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
        /// Preprocessing for images without alpha: uses rembg mask.
        /// </summary>
        internal static async Task<Tensor<float>> ApplyMaskAndCompositeAsync(
            Texture2D image, Tensor<float> alphaMask, float foregroundRatio)
        {
            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            int maskW = alphaMask.shape[alphaMask.shape.rank - 1];
            int maskH = alphaMask.shape[alphaMask.shape.rank - 2];
            var maskData = alphaMask.DownloadToArray();

            // Build alpha array from rembg mask, flipping y (texture=bottom-up, tensor=top-down)
            var alpha = new float[srcW * srcH];
            for (int y = 0; y < srcH; y++)
            for (int x = 0; x < srcW; x++)
            {
                int mx = Mathf.Clamp((int)((float)x / srcW * maskW), 0, maskW - 1);
                int my = maskH - 1 - Mathf.Clamp((int)((float)y / srcH * maskH), 0, maskH - 1);
                alpha[y * srcW + x] = Mathf.Clamp01(maskData[my * maskW + mx]);
            }

            return await CompositeAndResizeAsync(srcPixels, srcW, srcH, alpha, foregroundRatio);
        }

        /// <summary>
        /// Preprocessing for RGBA images: uses the texture's built-in alpha channel.
        /// Matches Python's prepare_image path for RGBA inputs.
        /// </summary>
        internal static async Task<Tensor<float>> CompositeFromRGBAAsync(
            Texture2D image, float foregroundRatio)
        {
            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            // Extract alpha from RGBA texture (GetPixels32 is bottom-up)
            var alpha = new float[srcW * srcH];
            for (int i = 0; i < srcPixels.Length; i++)
                alpha[i] = srcPixels[i].a / 255f;

            return await CompositeAndResizeAsync(srcPixels, srcW, srcH, alpha, foregroundRatio);
        }

        /// <summary>
        /// Shared logic: crop fg bbox → pad square → pad ratio → composite → bilinear resize to 512.
        /// Matches Python's resize_foreground + ImagePreprocessor exactly.
        /// </summary>
        private static async Task<Tensor<float>> CompositeAndResizeAsync(
            Color32[] srcPixels, int srcW, int srcH, float[] alpha, float ratio)
        {
            const int outputSize = 512;

            var (compositePixels, compositeSize) = await Task.Run(() =>
                BuildNativeComposite(srcPixels, srcW, srcH, alpha, ratio));

            await AsyncHelper.YieldFrame();

            // Create texture with mipmaps for antialiased downsample (matches Python antialias=True)
            var compositeTex = new Texture2D(compositeSize, compositeSize, TextureFormat.RGB24, true);
            compositeTex.filterMode = FilterMode.Trilinear;
            compositeTex.SetPixels32(compositePixels);
            compositeTex.Apply(true);

            // GPU bilinear resize to 512×512 via RenderTexture
            var rt = RenderTexture.GetTemporary(outputSize, outputSize, 0, RenderTextureFormat.ARGB32);
            rt.filterMode = FilterMode.Bilinear;
            Graphics.Blit(compositeTex, rt);
            SafeDestroy(compositeTex);

            var resized = new Texture2D(outputSize, outputSize, TextureFormat.RGB24, false);
            RenderTexture.active = rt;
            resized.ReadPixels(new Rect(0, 0, outputSize, outputSize), 0, 0);
            resized.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);

            var tensor = new Tensor<float>(new TensorShape(1, 3, outputSize, outputSize));
            TextureConverter.ToTensor(resized, tensor, new TextureTransform());
            SafeDestroy(resized);

            return tensor;
        }

        /// <summary>
        /// Builds the composite at native resolution (not 512×512).
        /// Returns the pixel array and the square size.
        /// Matches Python's resize_foreground: crop bbox → pad square → pad ratio → composite.
        /// </summary>
        private static (Color32[] pixels, int size) BuildNativeComposite(
            Color32[] srcPixels, int srcW, int srcH, float[] alpha, float ratio)
        {
            // Find foreground bounding box from alpha
            int minX = srcW, minY = srcH, maxX = 0, maxY = 0;
            for (int y = 0; y < srcH; y++)
            for (int x = 0; x < srcW; x++)
            {
                if (alpha[y * srcW + x] > 0f)
                {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }

            if (maxX <= minX || maxY <= minY)
            {
                int sz = 64;
                var gray = new Color32[sz * sz];
                byte g = 128;
                for (int i = 0; i < gray.Length; i++)
                    gray[i] = new Color32(g, g, g, 255);
                return (gray, sz);
            }

            // Crop dimensions (Python: fg = image[y1:y2, x1:x2] — exclusive of y2,x2)
            int cropW = maxX - minX;
            int cropH = maxY - minY;
            if (cropW < 1) cropW = 1;
            if (cropH < 1) cropH = 1;

            // Pad to square
            int sqSize = Mathf.Max(cropW, cropH);
            int padX0 = (sqSize - cropW) / 2;
            int padY0 = (sqSize - cropH) / 2;

            // Pad to achieve foreground ratio — int() truncation to match Python's int(size / ratio)
            int finalSize = (int)(sqSize / ratio);
            if (finalSize < sqSize) finalSize = sqSize;
            int outerPadX = (finalSize - sqSize) / 2;
            int outerPadY = (finalSize - sqSize) / 2;

            // Composite at native finalSize resolution (NOT 512)
            var result = new Color32[finalSize * finalSize];
            byte grayByte = 128;
            for (int i = 0; i < result.Length; i++)
                result[i] = new Color32(grayByte, grayByte, grayByte, 255);

            for (int fy = 0; fy < finalSize; fy++)
            for (int fx = 0; fx < finalSize; fx++)
            {
                // Map from padded frame back to source coordinates
                int cropX = fx - outerPadX - padX0;
                int cropY = fy - outerPadY - padY0;
                int srcX = cropX + minX;
                int srcY = cropY + minY;

                if (srcX < 0 || srcX >= srcW || srcY < 0 || srcY >= srcH)
                    continue;

                int si = srcY * srcW + srcX;
                float a = alpha[si];
                if (a < 0.001f) continue;

                var s = srcPixels[si];
                byte r = (byte)(s.r * a + grayByte * (1f - a));
                byte g = (byte)(s.g * a + grayByte * (1f - a));
                byte b = (byte)(s.b * a + grayByte * (1f - a));
                result[fy * finalSize + fx] = new Color32(r, g, b, 255);
            }

            return (result, finalSize);
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
