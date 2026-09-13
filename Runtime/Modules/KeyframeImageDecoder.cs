using System;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Keyframe JPEG → RGBA32 pixels off the main thread.
    ///
    /// <para><see cref="ImageConversion.LoadImage"/> decodes on the calling
    /// thread: 15-30 ms for a 1280×960 JPEG on Quest, once per keyframe per
    /// bake pass, which is a dropped frame each time. On Android the decode
    /// runs on a worker through <c>BitmapFactory</c> (JNI, thread attached
    /// for the call) into a direct <c>ByteBuffer</c> over a
    /// <see cref="NativeArray{T}"/>; the main thread only uploads pixels.
    /// Anywhere else — or if the JNI path fails once — falls back to
    /// <c>LoadImage</c>.</para>
    /// </summary>
    internal static class KeyframeImageDecoder
    {
        /// <summary>A decoded image or, when <see cref="Rgba"/> is not created,
        /// the still-encoded bytes for the main-thread fallback.</summary>
        internal sealed class Decoded : IDisposable
        {
            public int Width, Height;
            public NativeArray<byte> Rgba;   // bottom-up rows, RGBA32
            public byte[] Encoded;           // fallback: LoadImage on the main thread
            /// <summary>Worker time spent in the file read and in the decode (profile).</summary>
            public double ReadMs, DecodeMs;
            public bool IsFallback => !Rgba.IsCreated;

            /// <summary>Main thread. Upload into <paramref name="tex"/>, resizing
            /// it to the image. Returns false when nothing could be decoded.</summary>
            public bool ApplyTo(Texture2D tex)
            {
                if (Rgba.IsCreated)
                {
                    if (tex.width != Width || tex.height != Height || tex.format != TextureFormat.RGBA32)
                        tex.Reinitialize(Width, Height, TextureFormat.RGBA32, false);
                    tex.LoadRawTextureData(Rgba);
                    tex.Apply(false, false);
                    return true;
                }
                return Encoded != null && ImageConversion.LoadImage(tex, Encoded);
            }

            public void Dispose()
            {
                if (Rgba.IsCreated) Rgba.Dispose();
                Encoded = null;
            }
        }

        static bool _nativeDisabled;

#if UNITY_ANDROID && !UNITY_EDITOR
        const bool AndroidPlayer = true;
#else
        const bool AndroidPlayer = false;
#endif

        internal static bool NativeDecodeAvailable => AndroidPlayer && !_nativeDisabled;

        /// <summary>Read the file and decode it, all on worker threads.</summary>
        internal static Task<Decoded> ReadAndDecodeAsync(string path)
        {
            // One worker hop for both: the read is a few ms of I/O and the
            // decode 15-30 ms of CPU, neither belongs anywhere near the frame.
            return Task.Run(() =>
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                byte[] jpg = File.ReadAllBytes(path);
                double readMs = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                var d = Decode(jpg);
                if (d != null)
                {
                    d.ReadMs = readMs;
                    d.DecodeMs = sw.Elapsed.TotalMilliseconds;
                }
                return d;
            });
        }

        /// <summary>Worker thread. Native decode when available, else the bytes
        /// for a main-thread <c>LoadImage</c>.</summary>
        static Decoded Decode(byte[] jpg)
        {
            if (jpg == null || jpg.Length == 0) return null;
            if (NativeDecodeAvailable)
            {
                var d = DecodeAndroid(jpg);
                if (d != null) return d;
            }
            return new Decoded { Encoded = jpg };
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        static Decoded DecodeAndroid(byte[] jpg)
        {
            bool attached = false;
            IntPtr bufferRef = IntPtr.Zero;
            NativeArray<byte> raw = default;
            try
            {
                attached = AndroidJNI.AttachCurrentThread() == 0;
                using var factory = new AndroidJavaClass("android.graphics.BitmapFactory");
                using var opts = new AndroidJavaObject("android.graphics.BitmapFactory$Options");
                using var cfg = new AndroidJavaClass("android.graphics.Bitmap$Config");
                using var argb = cfg.GetStatic<AndroidJavaObject>("ARGB_8888");
                opts.Set("inPreferredConfig", argb);
                opts.Set("inScaled", false);

                using var bmp = factory.CallStatic<AndroidJavaObject>("decodeByteArray", jpg, 0, jpg.Length, opts);
                if (bmp == null) return null;

                int w = bmp.Call<int>("getWidth");
                int h = bmp.Call<int>("getHeight");
                int rowBytes = bmp.Call<int>("getRowBytes");
                int byteCount = bmp.Call<int>("getByteCount");
                if (w <= 0 || h <= 0 || rowBytes < w * 4 || byteCount < rowBytes * h) return null;
                // The byte layout below assumes ARGB_8888 (R,G,B,A in memory);
                // a decoder that ignored the preference would hand us 565 rows.
                using var config = bmp.Call<AndroidJavaObject>("getConfig");
                if (config == null || config.Call<string>("name") != "ARGB_8888")
                {
                    Logger.Warning("[KeyframeImageDecoder] Bitmap is not ARGB_8888; using LoadImage.");
                    _nativeDisabled = true;
                    return null;
                }

                raw = new NativeArray<byte>(byteCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                bufferRef = AndroidJNI.NewDirectByteBuffer(raw);
                if (bufferRef == IntPtr.Zero) return null;

                // copyPixelsToBuffer(java.nio.Buffer): exact signature via JNI so
                // the ByteBuffer argument is not matched against Buffer by reflection.
                IntPtr bmpClass = AndroidJNI.GetObjectClass(bmp.GetRawObject());
                IntPtr mid = AndroidJNI.GetMethodID(bmpClass, "copyPixelsToBuffer", "(Ljava/nio/Buffer;)V");
                AndroidJNI.DeleteLocalRef(bmpClass);
                if (mid == IntPtr.Zero) return null;
                var args = new jvalue[1];
                args[0].l = bufferRef;
                AndroidJNI.CallVoidMethod(bmp.GetRawObject(), mid, args);
                if (AndroidJNI.ExceptionOccurred() != IntPtr.Zero)
                {
                    AndroidJNI.ExceptionClear();
                    return null;
                }
                bmp.Call("recycle");

                // Bitmap rows are top-down; Unity textures are bottom-up. Flip
                // while dropping any row padding.
                var rgba = new NativeArray<byte>(w * h * 4, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                int rowLen = w * 4;
                for (int y = 0; y < h; y++)
                    NativeArray<byte>.Copy(raw, (h - 1 - y) * rowBytes, rgba, y * rowLen, rowLen);

                return new Decoded { Width = w, Height = h, Rgba = rgba };
            }
            catch (Exception e)
            {
                _nativeDisabled = true;
                Logger.Warning($"[KeyframeImageDecoder] Native JPEG decode unavailable ({e.Message}); using LoadImage on the main thread.");
                return null;
            }
            finally
            {
                if (bufferRef != IntPtr.Zero) AndroidJNI.DeleteLocalRef(bufferRef);
                if (raw.IsCreated) raw.Dispose();
                if (attached) AndroidJNI.DetachCurrentThread();
            }
        }
#else
        static Decoded DecodeAndroid(byte[] jpg) => null;
#endif
    }
}
