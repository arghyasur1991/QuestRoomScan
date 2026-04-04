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
        /// Set to a directory path to save intermediate debug images at each stage.
        /// Null disables debug output. Matches Python's debug_preprocess.py stages.
        /// </summary>
        internal static string DebugOutputDir;
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

            // Save debug: alpha mask visualization
            if (DebugOutputDir != null)
                SaveAlphaMask(alpha, srcW, srcH, System.IO.Path.Combine(DebugOutputDir, "unity_alpha_mask.png"));

            // Create texture with mipmaps for antialiased downsample (matches Python antialias=True)
            var compositeTex = new Texture2D(compositeSize, compositeSize, TextureFormat.RGB24, true);
            compositeTex.filterMode = FilterMode.Trilinear;
            compositeTex.SetPixels32(compositePixels);
            compositeTex.Apply(true);

            // Save debug: native-res composite (before 512 resize)
            if (DebugOutputDir != null)
            {
                var pngBytes = compositeTex.EncodeToPNG();
                System.IO.File.WriteAllBytes(
                    System.IO.Path.Combine(DebugOutputDir, $"unity_composite_{compositeSize}x{compositeSize}.png"), pngBytes);
                Logger.Info($"[ImagePreprocessor] Saved native composite: {compositeSize}×{compositeSize}");
            }

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

        /// <summary>
        /// Saves a rembg mask tensor as a grayscale PNG.
        /// </summary>
        internal static void SaveMaskDebugImage(Tensor<float> mask, string path)
        {
            var data = mask.DownloadToArray();
            var shape = mask.shape;
            int h = shape[shape.rank - 2];
            int w = shape[shape.rank - 1];
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            var pixels = new Color32[w * h];

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int ty = h - 1 - y;
                byte v = (byte)(Mathf.Clamp01(data[y * w + x]) * 255);
                pixels[ty * w + x] = new Color32(v, v, v, 255);
            }

            tex.SetPixels32(pixels);
            tex.Apply();
            System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
            SafeDestroy(tex);
            Logger.Info($"[ImagePreprocessor] Saved mask: {path} ({w}×{h})");
        }

        private static void SaveAlphaMask(float[] alpha, int w, int h, string path)
        {
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            var pixels = new Color32[w * h];
            for (int i = 0; i < alpha.Length; i++)
            {
                byte v = (byte)(Mathf.Clamp01(alpha[i]) * 255);
                pixels[i] = new Color32(v, v, v, 255);
            }
            tex.SetPixels32(pixels);
            tex.Apply();
            System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
            SafeDestroy(tex);
            Logger.Info($"[ImagePreprocessor] Saved alpha mask: {path} ({w}×{h})");
        }

        internal static void SaveDebugImage(Tensor<float> tensor, string path)
        {
            var shape = tensor.shape;
            int h = shape[2], w = shape[3];
            var data = tensor.DownloadToArray();
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            var pixels = new Color32[w * h];

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int ty = h - 1 - y;
                float r = data[0 * h * w + y * w + x];
                float g = data[1 * h * w + y * w + x];
                float b = data[2 * h * w + y * w + x];
                pixels[ty * w + x] = new Color32(
                    (byte)(Mathf.Clamp01(r) * 255),
                    (byte)(Mathf.Clamp01(g) * 255),
                    (byte)(Mathf.Clamp01(b) * 255), 255);
            }

            tex.SetPixels32(pixels);
            tex.Apply();
            System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
            SafeDestroy(tex);

            float min = float.MaxValue, max = float.MinValue;
            float sumR = 0, sumG = 0, sumB = 0;
            int chSize = h * w;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }
            for (int i = 0; i < chSize; i++) sumR += data[i];
            for (int i = 0; i < chSize; i++) sumG += data[chSize + i];
            for (int i = 0; i < chSize; i++) sumB += data[2 * chSize + i];

            int cx = w / 2, cy = h / 2;
            Logger.Info($"[ImagePreprocessor] Debug tensor shape: (1, 3, {h}, {w})");
            Logger.Info($"[ImagePreprocessor] Range: [{min:F4}, {max:F4}]");
            Logger.Info($"[ImagePreprocessor] Mean: [{sumR / chSize:F4}, {sumG / chSize:F4}, {sumB / chSize:F4}]");
            Logger.Info($"[ImagePreprocessor] Center ({cx},{cy}): R={data[cy * w + cx]:F4} G={data[chSize + cy * w + cx]:F4} B={data[2 * chSize + cy * w + cx]:F4}");
            Logger.Info($"[ImagePreprocessor] Corner (0,0): R={data[0]:F4} G={data[chSize]:F4} B={data[2 * chSize]:F4}");
            Logger.Info($"[ImagePreprocessor] Saved debug image: {path}");
        }

        /// <summary>
        /// Saves a (1,3,H,W) tensor as raw float32 binary for Python comparison.
        /// </summary>
        internal static void DumpTensorBinary(Tensor<float> tensor, string path)
        {
            var data = tensor.DownloadToArray();
            var bytes = new byte[data.Length * sizeof(float)];
            System.Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            System.IO.File.WriteAllBytes(path, bytes);

            var shape = tensor.shape;
            var metaPath = path + ".meta.txt";
            System.IO.File.WriteAllText(metaPath,
                $"dtype=float32\nshape={shape[0]},{shape[1]},{shape[2]},{shape[3]}\n");
            Logger.Info($"[ImagePreprocessor] Dumped tensor: {path} ({data.Length} floats, {bytes.Length} bytes)");
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
