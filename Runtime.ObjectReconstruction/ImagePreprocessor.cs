#if HAS_ONNXRUNTIME
using System;
using System.Threading.Tasks;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// RGB24 byte[] frame with top-left origin.
    /// Extracted once from Texture2D via unsafe pointer copy, then used throughout the pipeline.
    /// </summary>
    internal struct Frame
    {
        public byte[] data;
        public int width;
        public int height;

        public Frame(byte[] data, int width, int height)
        {
            this.data = data;
            this.width = width;
            this.height = height;
        }
    }

    /// <summary>
    /// Applies background removal mask and composites onto gray (0.5) background,
    /// matching the TripoSR training pipeline preprocessing.
    /// All outputs are raw float[] NCHW arrays for ORT consumption.
    ///
    /// Pipeline: Texture2D → Frame (once) → all processing on byte[] → NCHW float[].
    /// </summary>
    internal static class ImagePreprocessor
    {
        /// <summary>
        /// Extracts a Texture2D (RGB24) into a Frame with top-left origin.
        /// Uses GetPixelData + unsafe pointer + Parallel.For with Y-flip.
        /// </summary>
        internal static unsafe Frame Texture2DToFrame(Texture2D tex)
        {
            int w = tex.width, h = tex.height;
            int rowBytes = w * 3;
            var pixelData = tex.GetPixelData<byte>(0);
            byte* srcPtr = (byte*)pixelData.GetUnsafeReadOnlyPtr();
            var imageData = new byte[pixelData.Length];

            fixed (byte* dstPtrFixed = imageData)
            {
                byte* srcLocal = srcPtr;
                byte* dstLocal = dstPtrFixed;
                Parallel.For(0, h, y =>
                {
                    byte* srcRow = srcLocal + (h - 1 - y) * rowBytes;
                    byte* dstRow = dstLocal + y * rowBytes;
                    Buffer.MemoryCopy(srcRow, dstRow, rowBytes, rowBytes);
                });
            }
            return new Frame(imageData, w, h);
        }

        /// <summary>
        /// Writes a Frame (top-left origin) back into a Texture2D (RGB24) with Y-flip.
        /// </summary>
        internal static unsafe Texture2D FrameToTexture2D(Frame frame)
        {
            var tex = new Texture2D(frame.width, frame.height, TextureFormat.RGB24, false);
            var pixelData = tex.GetPixelData<byte>(0);
            byte* texPtr = (byte*)pixelData.GetUnsafePtr();
            int rowBytes = frame.width * 3;

            fixed (byte* srcPtrFixed = frame.data)
            {
                byte* srcLocal = srcPtrFixed;
                Parallel.For(0, frame.height, y =>
                {
                    int unityY = frame.height - 1 - y;
                    byte* srcRow = srcLocal + y * rowBytes;
                    byte* dstRow = texPtr + unityY * rowBytes;
                    Buffer.MemoryCopy(srcRow, dstRow, rowBytes, rowBytes);
                });
            }
            tex.Apply();
            return tex;
        }

        /// <summary>
        /// Preprocessing for images without alpha: uses rembg mask.
        /// Matches Python's rembg flow: uint8 quantization → GPU bilinear upscale → composite.
        /// </summary>
        internal static async Task<float[]> ApplyMaskAndCompositeAsync(
            Texture2D image, float[] alphaMask, int maskW, int maskH, float foregroundRatio)
        {
            var srcFrame = Texture2DToFrame(EnsureRGB24(image));
            var alpha = UpscaleMaskGPU(alphaMask, maskW, maskH, srcFrame.width, srcFrame.height);
            return await CompositeAndResizeAsync(srcFrame, alpha, foregroundRatio);
        }

        /// <summary>
        /// Upscales a NCHW mask to (dstW, dstH) using GPU bilinear filtering.
        /// Quantizes to uint8 first to match Python rembg's (mask*255).astype(uint8) → PIL.resize.
        /// Returns alpha in top-left layout (matching Frame).
        /// </summary>
        private static unsafe float[] UpscaleMaskGPU(float[] maskData, int maskW, int maskH, int dstW, int dstH)
        {
            var maskTex = new Texture2D(maskW, maskH, TextureFormat.R8, false, true);
            maskTex.filterMode = FilterMode.Bilinear;
            maskTex.wrapMode = TextureWrapMode.Clamp;
            var pixelData = maskTex.GetPixelData<byte>(0);
            byte* texPtr = (byte*)pixelData.GetUnsafePtr();

            for (int ty = 0; ty < maskH; ty++)
            for (int tx = 0; tx < maskW; tx++)
            {
                int tensorY = maskH - 1 - ty;
                texPtr[ty * maskW + tx] = (byte)(Mathf.Clamp01(maskData[tensorY * maskW + tx]) * 255f);
            }
            maskTex.Apply();

            var maskRT = RenderTexture.GetTemporary(dstW, dstH, 0, RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Linear);
            maskRT.filterMode = FilterMode.Bilinear;
            Graphics.Blit(maskTex, maskRT);
            SafeDestroy(maskTex);

            var upscaled = new Texture2D(dstW, dstH, TextureFormat.RGBA32, false, true);
            RenderTexture.active = maskRT;
            upscaled.ReadPixels(new Rect(0, 0, dstW, dstH), 0, 0);
            upscaled.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(maskRT);

            var upPixelData = upscaled.GetPixelData<byte>(0);
            byte* upPtr = (byte*)upPixelData.GetUnsafeReadOnlyPtr();
            var alpha = new float[dstW * dstH];
            const int bpp = 4; // RGBA32

            fixed (float* alphaPtrFixed = alpha)
            {
                byte* upLocal = upPtr;
                float* alphaLocal = alphaPtrFixed;
                Parallel.For(0, dstH, y =>
                {
                    int unityY = dstH - 1 - y;
                    for (int x = 0; x < dstW; x++)
                        alphaLocal[y * dstW + x] = upLocal[(unityY * dstW + x) * bpp] / 255f;
                });
            }
            SafeDestroy(upscaled);
            return alpha;
        }

        /// <summary>
        /// Preprocessing for RGBA images: uses the texture's built-in alpha channel.
        /// </summary>
        internal static async Task<float[]> CompositeFromRGBAAsync(
            Texture2D image, float foregroundRatio)
        {
            var (srcFrame, alpha) = ExtractRGBAFrame(EnsureReadable(image));
            return await CompositeAndResizeAsync(srcFrame, alpha, foregroundRatio);
        }

        /// <summary>
        /// Extracts RGB frame + alpha from an RGBA/BGRA texture via unsafe pointers.
        /// Separated from async method because C# forbids await in unsafe context.
        /// </summary>
        private static unsafe (Frame frame, float[] alpha) ExtractRGBAFrame(Texture2D tex)
        {
            int w = tex.width, h = tex.height;
            var pixelData = tex.GetPixelData<byte>(0);
            byte* srcPtr = (byte*)pixelData.GetUnsafeReadOnlyPtr();
            int bpp = GetBytesPerPixel(tex.format);
            bool isBGRA = tex.format == TextureFormat.BGRA32;
            bool hasAlphaChannel = bpp == 4;

            var frameData = new byte[w * h * 3];
            var alpha = new float[w * h];

            fixed (byte* dstPtrFixed = frameData)
            fixed (float* alphaPtrFixed = alpha)
            {
                byte* srcLocal = srcPtr;
                byte* dstLocal = dstPtrFixed;
                float* alphaLocal = alphaPtrFixed;

                Parallel.For(0, h, y =>
                {
                    int unityY = h - 1 - y;
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = (unityY * w + x) * bpp;
                        int dstIdx = (y * w + x) * 3;
                        int flatIdx = y * w + x;

                        if (isBGRA)
                        {
                            dstLocal[dstIdx + 0] = srcLocal[srcIdx + 2];
                            dstLocal[dstIdx + 1] = srcLocal[srcIdx + 1];
                            dstLocal[dstIdx + 2] = srcLocal[srcIdx + 0];
                        }
                        else
                        {
                            dstLocal[dstIdx + 0] = srcLocal[srcIdx + 0];
                            dstLocal[dstIdx + 1] = srcLocal[srcIdx + 1];
                            dstLocal[dstIdx + 2] = srcLocal[srcIdx + 2];
                        }
                        alphaLocal[flatIdx] = hasAlphaChannel
                            ? srcLocal[srcIdx + 3] / 255f
                            : 1f;
                    }
                });
            }

            return (new Frame(frameData, w, h), alpha);
        }

        /// <summary>
        /// Shared logic: crop fg bbox → pad square → pad ratio → composite → bilinear resize to 512 → NCHW float[].
        /// </summary>
        private static async Task<float[]> CompositeAndResizeAsync(
            Frame srcFrame, float[] alpha, float ratio)
        {
            const int outputSize = 512;

            var (compositeFrame, compositeSize) = await Task.Run(() =>
                BuildNativeComposite(srcFrame, alpha, ratio));

            await AsyncHelper.YieldFrame();

            var compositeTex = FrameToTexture2D(compositeFrame);
            compositeTex.filterMode = FilterMode.Trilinear;
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
        /// Converts a Texture2D (RGB24) to NCHW float[].
        /// Uses unsafe pointer access + Parallel.For for maximum throughput.
        /// </summary>
        internal static unsafe float[] TextureToNCHW(Texture2D tex, int size)
        {
            var pixels = tex.GetPixelData<byte>(0);
            byte* srcPtr = (byte*)pixels.GetUnsafeReadOnlyPtr();
            var result = new float[3 * size * size];

            int channelSize = size * size;
            const int bytesPerPixel = 3;

            fixed (float* dstPtrFixed = result)
            {
                byte* srcLocal = srcPtr;
                float* dstLocal = dstPtrFixed;

                Parallel.For(0, size, y =>
                {
                    int unityY = size - 1 - y;
                    for (int x = 0; x < size; x++)
                    {
                        int srcIdx = (unityY * size + x) * bytesPerPixel;
                        int dstIdx = y * size + x;
                        dstLocal[0 * channelSize + dstIdx] = srcLocal[srcIdx + 0] / 255f;
                        dstLocal[1 * channelSize + dstIdx] = srcLocal[srcIdx + 1] / 255f;
                        dstLocal[2 * channelSize + dstIdx] = srcLocal[srcIdx + 2] / 255f;
                    }
                });
            }
            return result;
        }

        /// <summary>
        /// Converts a Frame (top-left origin, RGB24 byte[]) to NCHW float[].
        /// No Y-flip needed — Frame is already top-left.
        /// </summary>
        internal static unsafe float[] FrameToNCHW(Frame frame)
        {
            int w = frame.width, h = frame.height;
            int channelSize = w * h;
            var result = new float[3 * channelSize];

            fixed (byte* srcPtrFixed = frame.data)
            fixed (float* dstPtrFixed = result)
            {
                byte* srcLocal = srcPtrFixed;
                float* dstLocal = dstPtrFixed;

                Parallel.For(0, h, y =>
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = (y * w + x) * 3;
                        int dstIdx = y * w + x;
                        dstLocal[0 * channelSize + dstIdx] = srcLocal[srcIdx + 0] / 255f;
                        dstLocal[1 * channelSize + dstIdx] = srcLocal[srcIdx + 1] / 255f;
                        dstLocal[2 * channelSize + dstIdx] = srcLocal[srcIdx + 2] / 255f;
                    }
                });
            }
            return result;
        }

        /// <summary>
        /// Builds the composite at native resolution (not 512x512).
        /// Operates entirely on Frame byte[] — no Color32 allocations.
        /// Matches Python's resize_foreground: crop bbox → pad square → pad ratio → composite.
        /// </summary>
        private static unsafe (Frame frame, int size) BuildNativeComposite(
            Frame src, float[] alpha, float ratio)
        {
            int srcW = src.width, srcH = src.height;
            const float bboxThreshold = 0.5f;
            int minX = srcW, minY = srcH, maxX = 0, maxY = 0;

            fixed (float* alphaPtr = alpha)
            {
                float* aLocal = alphaPtr;
                for (int y = 0; y < srcH; y++)
                for (int x = 0; x < srcW; x++)
                {
                    if (aLocal[y * srcW + x] > bboxThreshold)
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
                int sz = 64;
                var grayData = new byte[sz * sz * 3];
                Array.Fill(grayData, (byte)128);
                return (new Frame(grayData, sz, sz), sz);
            }

            int cropW = maxX - minX;
            int cropH = maxY - minY;
            if (cropW < 1) cropW = 1;
            if (cropH < 1) cropH = 1;

            int sqSize = Math.Max(cropW, cropH);
            int padX0 = (sqSize - cropW) / 2;
            int padY0 = (sqSize - cropH) / 2;

            int finalSize = (int)(sqSize / ratio);
            if (finalSize < sqSize) finalSize = sqSize;
            int outerPadX = (finalSize - sqSize) / 2;
            int outerPadY = (finalSize - sqSize) / 2;

            var resultData = new byte[finalSize * finalSize * 3];
            Array.Fill(resultData, (byte)128);

            fixed (byte* srcPtrFixed = src.data)
            fixed (byte* dstPtrFixed = resultData)
            fixed (float* alphaPtrFixed = alpha)
            {
                byte* srcLocal = srcPtrFixed;
                byte* dstLocal = dstPtrFixed;
                float* aLocal = alphaPtrFixed;
                const byte grayByte = 128;

                Parallel.For(0, finalSize, fy =>
                {
                    for (int fx = 0; fx < finalSize; fx++)
                    {
                        int cropX = fx - outerPadX - padX0;
                        int cropY = fy - outerPadY - padY0;
                        int srcX = cropX + minX;
                        int srcY = cropY + minY;

                        if (srcX < 0 || srcX >= srcW || srcY < 0 || srcY >= srcH)
                            continue;

                        int si = srcY * srcW + srcX;
                        float a = aLocal[si];
                        if (a < 0.001f) continue;

                        int srcIdx = si * 3;
                        int dstIdx = (fy * finalSize + fx) * 3;
                        float invA = 1f - a;
                        dstLocal[dstIdx + 0] = (byte)(srcLocal[srcIdx + 0] * a + grayByte * invA);
                        dstLocal[dstIdx + 1] = (byte)(srcLocal[srcIdx + 1] * a + grayByte * invA);
                        dstLocal[dstIdx + 2] = (byte)(srcLocal[srcIdx + 2] * a + grayByte * invA);
                    }
                });
            }

            return (new Frame(resultData, finalSize, finalSize), finalSize);
        }

        private static Texture2D EnsureRGB24(Texture2D tex)
        {
            if (tex.format == TextureFormat.RGB24 && tex.isReadable)
                return tex;
            return ConvertToRGB24(tex);
        }

        private static Texture2D EnsureReadable(Texture2D tex)
        {
            if (tex.isReadable) return tex;
            return ConvertToRGB24(tex);
        }

        private static Texture2D ConvertToRGB24(Texture2D source)
        {
            var rt = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(source, rt);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var readable = new Texture2D(source.width, source.height, TextureFormat.RGB24, false);
            readable.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            readable.Apply();
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
            return readable;
        }

        private static int GetBytesPerPixel(TextureFormat format)
        {
            switch (format)
            {
                case TextureFormat.RGBA32:
                case TextureFormat.BGRA32:
                case TextureFormat.ARGB32:
                    return 4;
                case TextureFormat.RGB24:
                    return 3;
                default:
                    return 4;
            }
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
