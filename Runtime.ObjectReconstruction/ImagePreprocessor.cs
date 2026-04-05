#if HAS_ONNXRUNTIME
using System.Threading.Tasks;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Applies background removal mask and composites onto gray (0.5) background,
    /// matching the TripoSR training pipeline preprocessing.
    /// All outputs are raw float[] NCHW arrays for ORT consumption.
    /// </summary>
    internal static class ImagePreprocessor
    {
        /// <summary>
        /// Preprocessing for images without alpha: uses rembg mask.
        /// Matches Python's rembg flow: uint8 quantization → GPU bilinear upscale → composite.
        /// </summary>
        internal static async Task<float[]> ApplyMaskAndCompositeAsync(
            Texture2D image, float[] alphaMask, int maskW, int maskH, float foregroundRatio)
        {
            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            var alpha = UpscaleMaskGPU(alphaMask, maskW, maskH, srcW, srcH);

            return await CompositeAndResizeAsync(srcPixels, srcW, srcH, alpha, foregroundRatio);
        }

        /// <summary>
        /// Upscales a NCHW mask to (dstW, dstH) using GPU bilinear filtering.
        /// Quantizes to uint8 first to match Python rembg's (mask*255).astype(uint8) → PIL.resize.
        /// Returns alpha in bottom-up layout (matching GetPixels32).
        /// </summary>
        private static float[] UpscaleMaskGPU(float[] maskData, int maskW, int maskH, int dstW, int dstH)
        {
            var maskTex = new Texture2D(maskW, maskH, TextureFormat.RGBA32, 1, true);
            maskTex.filterMode = FilterMode.Bilinear;
            maskTex.wrapMode = TextureWrapMode.Clamp;
            var maskPixels = new Color32[maskW * maskH];
            for (int ty = 0; ty < maskH; ty++)
            for (int tx = 0; tx < maskW; tx++)
            {
                int tensorY = maskH - 1 - ty;
                byte v = (byte)(Mathf.Clamp01(maskData[tensorY * maskW + tx]) * 255f);
                maskPixels[ty * maskW + tx] = new Color32(v, v, v, 255);
            }
            maskTex.SetPixels32(maskPixels);
            maskTex.Apply();

            var maskRT = RenderTexture.GetTemporary(dstW, dstH, 0, RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Linear);
            maskRT.filterMode = FilterMode.Bilinear;
            Graphics.Blit(maskTex, maskRT);
            SafeDestroy(maskTex);

            var upscaled = new Texture2D(dstW, dstH, TextureFormat.RGBA32, 1, true);
            RenderTexture.active = maskRT;
            upscaled.ReadPixels(new Rect(0, 0, dstW, dstH), 0, 0);
            upscaled.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(maskRT);

            var upPixels = upscaled.GetPixels32();
            SafeDestroy(upscaled);

            var alpha = new float[dstW * dstH];
            for (int i = 0; i < alpha.Length; i++)
                alpha[i] = upPixels[i].r / 255f;

            return alpha;
        }

        /// <summary>
        /// Preprocessing for RGBA images: uses the texture's built-in alpha channel.
        /// </summary>
        internal static async Task<float[]> CompositeFromRGBAAsync(
            Texture2D image, float foregroundRatio)
        {
            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            var alpha = new float[srcW * srcH];
            for (int i = 0; i < srcPixels.Length; i++)
                alpha[i] = srcPixels[i].a / 255f;

            return await CompositeAndResizeAsync(srcPixels, srcW, srcH, alpha, foregroundRatio);
        }

        /// <summary>
        /// Shared logic: crop fg bbox → pad square → pad ratio → composite → bilinear resize to 512 → NCHW float[].
        /// </summary>
        private static async Task<float[]> CompositeAndResizeAsync(
            Color32[] srcPixels, int srcW, int srcH, float[] alpha, float ratio)
        {
            const int outputSize = 512;

            var (compositePixels, compositeSize) = await Task.Run(() =>
                BuildNativeComposite(srcPixels, srcW, srcH, alpha, ratio));

            await AsyncHelper.YieldFrame();

            var compositeTex = new Texture2D(compositeSize, compositeSize, TextureFormat.RGB24, true);
            compositeTex.filterMode = FilterMode.Trilinear;
            compositeTex.SetPixels32(compositePixels);
            compositeTex.Apply(true);

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

            var result = TextureToNCHW(resized, outputSize);
            SafeDestroy(resized);

            return result;
        }

        /// <summary>
        /// Converts a Texture2D to a CPU float[] in NCHW layout.
        /// Uses unsafe pointer access for performance.
        /// </summary>
        internal static unsafe float[] TextureToNCHW(Texture2D tex, int size)
        {
            var pixels = tex.GetPixelData<byte>(0);
            byte* srcPtr = (byte*)pixels.GetUnsafeReadOnlyPtr();
            var result = new float[3 * size * size];

            int channelSize = size * size;
            int bytesPerPixel = 3; // RGB24

            fixed (float* dstPtr = result)
            {
                System.Threading.Tasks.Parallel.For(0, size, y =>
                {
                    int unityY = size - 1 - y;
                    for (int x = 0; x < size; x++)
                    {
                        int srcIdx = (unityY * size + x) * bytesPerPixel;
                        int dstIdx = y * size + x;
                        dstPtr[0 * channelSize + dstIdx] = srcPtr[srcIdx + 0] / 255f;
                        dstPtr[1 * channelSize + dstIdx] = srcPtr[srcIdx + 1] / 255f;
                        dstPtr[2 * channelSize + dstIdx] = srcPtr[srcIdx + 2] / 255f;
                    }
                });
            }
            return result;
        }

        /// <summary>
        /// Builds the composite at native resolution (not 512×512).
        /// Returns the pixel array and the square size.
        /// Matches Python's resize_foreground: crop bbox → pad square → pad ratio → composite.
        /// </summary>
        private static (Color32[] pixels, int size) BuildNativeComposite(
            Color32[] srcPixels, int srcW, int srcH, float[] alpha, float ratio)
        {
            const float bboxThreshold = 0.5f;
            int minX = srcW, minY = srcH, maxX = 0, maxY = 0;
            for (int y = 0; y < srcH; y++)
            for (int x = 0; x < srcW; x++)
            {
                if (alpha[y * srcW + x] > bboxThreshold)
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

            int cropW = maxX - minX;
            int cropH = maxY - minY;
            if (cropW < 1) cropW = 1;
            if (cropH < 1) cropH = 1;

            int sqSize = Mathf.Max(cropW, cropH);
            int padX0 = (sqSize - cropW) / 2;
            int padY0 = (sqSize - cropH) / 2;

            int finalSize = (int)(sqSize / ratio);
            if (finalSize < sqSize) finalSize = sqSize;
            int outerPadX = (finalSize - sqSize) / 2;
            int outerPadY = (finalSize - sqSize) / 2;

            var result = new Color32[finalSize * finalSize];
            byte grayByte = 128;
            for (int i = 0; i < result.Length; i++)
                result[i] = new Color32(grayByte, grayByte, grayByte, 255);

            for (int fy = 0; fy < finalSize; fy++)
            for (int fx = 0; fx < finalSize; fx++)
            {
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
