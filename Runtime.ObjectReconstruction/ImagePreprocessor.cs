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
        /// Matches Python's rembg flow: uint8 quantization → GPU bilinear upscale → composite.
        /// </summary>
        internal static async Task<Tensor<float>> ApplyMaskAndCompositeAsync(
            Texture2D image, Tensor<float> alphaMask, float foregroundRatio,
            bool cpuOnly = false)
        {
            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            int maskW = alphaMask.shape[alphaMask.shape.rank - 1];
            int maskH = alphaMask.shape[alphaMask.shape.rank - 2];
            var maskData = alphaMask.DownloadToArray();

            var alpha = UpscaleMaskGPU(maskData, maskW, maskH, srcW, srcH);

            return await CompositeAndResizeAsync(srcPixels, srcW, srcH, alpha, foregroundRatio, cpuOnly);
        }

        /// <summary>
        /// Upscales a NCHW mask tensor to (dstW, dstH) using GPU bilinear filtering.
        /// Quantizes to uint8 first to match Python rembg's (mask*255).astype(uint8) → PIL.resize.
        /// Returns alpha in bottom-up layout (matching GetPixels32).
        /// </summary>
        private static float[] UpscaleMaskGPU(float[] maskData, int maskW, int maskH, int dstW, int dstH)
        {
            // Build mask texture (uint8, linear color space to avoid sRGB distortion)
            // y-flip: tensor is top-down, Unity textures are bottom-up
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

            // GPU bilinear resize to source dimensions
            var maskRT = RenderTexture.GetTemporary(dstW, dstH, 0, RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Linear);
            maskRT.filterMode = FilterMode.Bilinear;
            Graphics.Blit(maskTex, maskRT);
            SafeDestroy(maskTex);

            // Read back upscaled mask
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
        /// Matches Python's prepare_image path for RGBA inputs.
        /// </summary>
        internal static async Task<Tensor<float>> CompositeFromRGBAAsync(
            Texture2D image, float foregroundRatio, bool cpuOnly = false)
        {
            int srcW = image.width;
            int srcH = image.height;
            var srcPixels = image.GetPixels32();

            // Extract alpha from RGBA texture (GetPixels32 is bottom-up)
            var alpha = new float[srcW * srcH];
            for (int i = 0; i < srcPixels.Length; i++)
                alpha[i] = srcPixels[i].a / 255f;

            return await CompositeAndResizeAsync(srcPixels, srcW, srcH, alpha, foregroundRatio, cpuOnly);
        }

        /// <summary>
        /// Shared logic: crop fg bbox → pad square → pad ratio → composite → bilinear resize to 512.
        /// Matches Python's resize_foreground + ImagePreprocessor exactly.
        /// </summary>
        private static async Task<Tensor<float>> CompositeAndResizeAsync(
            Color32[] srcPixels, int srcW, int srcH, float[] alpha, float ratio,
            bool cpuOnly = false)
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

            Tensor<float> tensor;
            if (cpuOnly)
            {
                tensor = TextureToCpuTensor(resized, outputSize);
            }
            else
            {
                tensor = new Tensor<float>(new TensorShape(1, 3, outputSize, outputSize));
                TextureConverter.ToTensor(resized, tensor, new TextureTransform());
            }
            SafeDestroy(resized);

            return tensor;
        }

        /// <summary>
        /// Converts a Texture2D to a CPU-backed NCHW tensor without touching the GPU.
        /// Produces identical output to TextureConverter.ToTensor with default TextureTransform.
        /// </summary>
        internal static Tensor<float> TextureToCpuTensor(Texture2D tex, int size)
        {
            var pixels = tex.GetPixels32();
            int channelSize = size * size;
            var data = new float[3 * channelSize];

            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                // GetPixels32 is bottom-up, tensor is top-down (NCHW)
                int texIdx = y * size + x;
                int tensorY = size - 1 - y;
                int idx = tensorY * size + x;
                var c = pixels[texIdx];
                data[0 * channelSize + idx] = c.r / 255f;
                data[1 * channelSize + idx] = c.g / 255f;
                data[2 * channelSize + idx] = c.b / 255f;
            }

            return new Tensor<float>(new TensorShape(1, 3, size, size), data);
        }

        /// <summary>
        /// Builds the composite at native resolution (not 512×512).
        /// Returns the pixel array and the square size.
        /// Matches Python's resize_foreground: crop bbox → pad square → pad ratio → composite.
        /// </summary>
        private static (Color32[] pixels, int size) BuildNativeComposite(
            Color32[] srcPixels, int srcW, int srcH, float[] alpha, float ratio)
        {
            // Use threshold 0.5 for bbox computation to match rembg post_process_mask behavior.
            // Soft alpha edges (values 0.01–0.49) inflate the bbox, shrinking the object in
            // the final 512x512 frame and degrading TripoSR quality. The actual compositing
            // still uses the full smooth alpha for clean anti-aliased edges.
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
