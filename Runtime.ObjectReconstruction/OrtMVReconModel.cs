#if HAS_ONNXRUNTIME
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    internal struct MVReconTiming
    {
        internal float LoadMs;
        internal float PreprocessMs;
        internal float InferMs;
        internal float MeshMs;
        internal float TotalMs => LoadMs + PreprocessMs + InferMs + MeshMs;

        public override string ToString() =>
            $"load={LoadMs:F0}ms preprocess={PreprocessMs:F0}ms infer={InferMs:F0}ms " +
            $"mesh={MeshMs:F0}ms total={TotalMs:F0}ms";
    }

    /// <summary>
    /// Multi-view reconstruction via ONNX Runtime. Takes N input views with known
    /// camera poses and produces a 64^3 occupancy volume + color volume.
    ///
    /// ONNX inputs:
    ///   images  [1, N, 3, 160, 160] — ImageNet-normalized RGB
    ///   w2c_cv  [1, N, 3, 4]        — world-to-camera (OpenCV convention)
    ///
    /// ONNX outputs:
    ///   density [1, 1, 64, 64, 64]  — occupancy logits
    ///   color   [1, 3, 64, 64, 64]  — color logits
    /// </summary>
    internal sealed class OrtMVReconModel : OrtModelBase
    {
        private const string ModelFileName = "ObjectReconstruction/mv_recon.onnx";
        private const int InputSize = 160;
        private const int NumViews = 3;
        internal const int VolumeRes = 64;

        internal MVReconTiming LastTiming { get; private set; }

        internal async Task LoadAsync(ExecutionProvider ep, bool mobileOptimized, CancellationToken ct)
        {
            await LoadSessionAsync(ModelFileName, ep, mobileOptimized, ct);
        }

        /// <summary>
        /// Run full multi-view reconstruction.
        /// Returns (density, color) as flat float arrays of length 64^3 and 3*64^3.
        /// Density values are raw logits (apply sigmoid for probability).
        /// </summary>
        internal async Task<(float[] density, float[] color)> RunAsync(
            float[] imagesNCHW, float[] w2cFlat, CancellationToken ct)
        {
            int imgLen = NumViews * 3 * InputSize * InputSize;
            int w2cLen = NumViews * 3 * 4;

            if (imagesNCHW.Length != imgLen)
                throw new ArgumentException(
                    $"Expected images length {imgLen}, got {imagesNCHW.Length}");
            if (w2cFlat.Length != w2cLen)
                throw new ArgumentException(
                    $"Expected w2c length {w2cLen}, got {w2cFlat.Length}");

            var imagesTensor = new DenseTensor<float>(
                imagesNCHW, new[] { 1, NumViews, 3, InputSize, InputSize });
            var w2cTensor = new DenseTensor<float>(
                w2cFlat, new[] { 1, NumViews, 3, 4 });

            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[0], imagesTensor));
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[1], w2cTensor));

            await RunPreallocated();
            ct.ThrowIfCancellationRequested();

            var densityTensor = GetPreallocatedOutput<float>("density");
            var colorTensor = GetPreallocatedOutput<float>("color");

            var densityBuf = densityTensor.Buffer;
            var colorBuf = colorTensor.Buffer;
            var density = new float[densityBuf.Length];
            var color = new float[colorBuf.Length];
            densityBuf.Span.CopyTo(density);
            colorBuf.Span.CopyTo(color);

            return (density, color);
        }

        #region Camera Math

        /// <summary>
        /// Convert Blender camera-to-world [4,4] → OpenCV world-to-camera [3,4].
        /// C# equivalent of the Python c2w_blender_to_w2c_cv function.
        /// </summary>
        internal static float[] BlenderC2WToW2C(float[][] c2wBlender)
        {
            int n = c2wBlender.Length;
            var result = new float[n * 3 * 4];

            for (int v = 0; v < n; v++)
            {
                var c2w = c2wBlender[v];

                // c2w_cv = c2w_blender @ flip (flip Y and Z)
                var c2wCv = new float[16];
                for (int r = 0; r < 4; r++)
                {
                    c2wCv[r * 4 + 0] =  c2w[r * 4 + 0];
                    c2wCv[r * 4 + 1] = -c2w[r * 4 + 1];
                    c2wCv[r * 4 + 2] = -c2w[r * 4 + 2];
                    c2wCv[r * 4 + 3] =  c2w[r * 4 + 3];
                }

                var w2c = Invert4x4(c2wCv);

                int off = v * 12;
                for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    result[off + r * 4 + c] = w2c[r * 4 + c];
            }

            return result;
        }

        private static float[] Invert4x4(float[] m)
        {
            var inv = new float[16];

            inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] +
                     m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
            inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] -
                     m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
            inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] +
                     m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
            inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] -
                      m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];

            float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
            if (MathF.Abs(det) < 1e-10f) return inv;
            float invDet = 1f / det;

            inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] -
                     m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
            inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] +
                     m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
            inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] -
                     m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
            inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] +
                      m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];

            inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] +
                     m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
            inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] -
                     m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
            inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] +
                      m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
            inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] -
                      m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];

            inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] -
                     m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
            inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] +
                     m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
            inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] -
                      m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
            inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] +
                      m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

            for (int i = 0; i < 16; i++) inv[i] *= invDet;
            return inv;
        }

        #endregion

        #region Image Preprocessing

        private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };
        private const float GrayBg = 127f / 255f;

        /// <summary>
        /// Convert N Texture2D views into a single NCHW float[] with ImageNet normalization.
        /// Output layout: [N, 3, InputSize, InputSize] flattened.
        /// RGBA images are composited on gray (0.5) background to match training pipeline.
        /// Images are center-cropped to square then resized to InputSize.
        /// </summary>
        internal static unsafe float[] PreprocessViews(Texture2D[] views)
        {
            int n = views.Length;
            int channelSize = InputSize * InputSize;
            int viewSize = 3 * channelSize;
            var result = new float[n * viewSize];

            for (int v = 0; v < n; v++)
            {
                var tex = EnsureReadableSquare(views[v], InputSize);
                var pixels = tex.GetPixelData<byte>(0);
                int bpp = tex.format == TextureFormat.RGBA32 ? 4 : 3;
                bool hasAlpha = bpp == 4;
                byte* srcPtr = (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(pixels);

                int baseOff = v * viewSize;
                fixed (float* dstFixed = result)
                {
                    byte* srcLocal = srcPtr;
                    float* dst = dstFixed + baseOff;
                    int sz = InputSize;
                    int cs = channelSize;

                    Parallel.For(0, sz, y =>
                    {
                        int unityY = sz - 1 - y;
                        for (int x = 0; x < sz; x++)
                        {
                            int srcIdx = (unityY * sz + x) * bpp;
                            int dstIdx = y * sz + x;

                            float r = srcLocal[srcIdx + 0] / 255f;
                            float g = srcLocal[srcIdx + 1] / 255f;
                            float b = srcLocal[srcIdx + 2] / 255f;

                            if (hasAlpha)
                            {
                                float a = srcLocal[srcIdx + 3] / 255f;
                                float invA = 1f - a;
                                r = r * a + GrayBg * invA;
                                g = g * a + GrayBg * invA;
                                b = b * a + GrayBg * invA;
                            }

                            dst[0 * cs + dstIdx] = (r - ImageNetMean[0]) / ImageNetStd[0];
                            dst[1 * cs + dstIdx] = (g - ImageNetMean[1]) / ImageNetStd[1];
                            dst[2 * cs + dstIdx] = (b - ImageNetMean[2]) / ImageNetStd[2];
                        }
                    });
                }

                if (tex != views[v])
                    SafeDestroy(tex);
            }

            return result;
        }

        private static Texture2D EnsureReadableSquare(Texture2D src, int targetSize)
        {
            int w = src.width, h = src.height;
            int sq = Mathf.Min(w, h);
            int cropX = (w - sq) / 2;
            int cropY = (h - sq) / 2;

            bool srcHasAlpha = src.format == TextureFormat.RGBA32 ||
                               src.format == TextureFormat.BGRA32 ||
                               src.format == TextureFormat.ARGB32;
            var dstFormat = srcHasAlpha ? TextureFormat.RGBA32 : TextureFormat.RGB24;

            var rt = RenderTexture.GetTemporary(targetSize, targetSize, 0, RenderTextureFormat.ARGB32);
            float scaleX = (float)sq / w;
            float scaleY = (float)sq / h;
            float offsetX = (float)cropX / w;
            float offsetY = (float)cropY / h;
            Graphics.Blit(src, rt, new Vector2(scaleX, scaleY), new Vector2(offsetX, offsetY));

            RenderTexture.active = rt;
            var result = new Texture2D(targetSize, targetSize, dstFormat, false);
            result.ReadPixels(new Rect(0, 0, targetSize, targetSize), 0, 0);
            result.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
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

        #endregion

        #region Volume Mesh Extraction

        /// <summary>
        /// Extract mesh from a 64^3 density logit volume using marching cubes.
        /// Applies sigmoid then thresholds at the given level.
        /// </summary>
        internal static async Task<Mesh> ExtractMeshFromVolume(
            float[] densityLogits, float[] colorLogits, float threshold, CancellationToken ct)
        {
            const int res = VolumeRes;
            int total = res * res * res;

            var density = new float[total];
            var colors = colorLogits != null ? new float[total * 3] : null;

            await Task.Run(() =>
            {
                Parallel.For(0, total, i =>
                {
                    density[i] = Sigmoid(densityLogits[i]);
                });

                if (colors != null)
                {
                    Parallel.For(0, total, i =>
                    {
                        colors[i * 3 + 0] = Sigmoid(colorLogits[0 * total + i]);
                        colors[i * 3 + 1] = Sigmoid(colorLogits[1 * total + i]);
                        colors[i * 3 + 2] = Sigmoid(colorLogits[2 * total + i]);
                    });
                }
            });
            ct.ThrowIfCancellationRequested();

            Vector3[] verts = null;
            int[] tris = null;
            await Task.Run(() =>
                CpuMarchingCubes64(density, res, threshold, out verts, out tris));
            ct.ThrowIfCancellationRequested();

            if (verts == null || verts.Length == 0)
                return new Mesh();

            Color[] vertColors = null;
            if (colors != null)
            {
                vertColors = new Color[verts.Length];
                await Task.Run(() =>
                {
                    Parallel.For(0, verts.Length, i =>
                    {
                        var p = verts[i];
                        float fx = (p.x + 0.5f) * (res - 1);
                        float fy = (p.y + 0.5f) * (res - 1);
                        float fz = (p.z + 0.5f) * (res - 1);

                        int ix = Mathf.Clamp(Mathf.RoundToInt(fx), 0, res - 1);
                        int iy = Mathf.Clamp(Mathf.RoundToInt(fy), 0, res - 1);
                        int iz = Mathf.Clamp(Mathf.RoundToInt(fz), 0, res - 1);
                        int idx = (iz * res * res + iy * res + ix) * 3;

                        vertColors[i] = new Color(colors[idx], colors[idx + 1], colors[idx + 2]);
                    });
                });
            }

            var mesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            if (vertColors != null) mesh.SetColors(vertColors);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        /// <summary>
        /// Marching cubes on a flat density[res^3] array.
        /// Vertices output in [-0.5, 0.5]^3 normalized coordinates.
        /// </summary>
        private static unsafe void CpuMarchingCubes64(
            float[] density, int res, float threshold,
            out Vector3[] vertices, out int[] triangles)
        {
            float voxSize = 1f / (res - 1);
            int resRes = res * res;
            int sliceCount = res - 1;

            var sliceVerts = new System.Collections.Generic.List<Vector3>[sliceCount];
            var sliceTris = new System.Collections.Generic.List<int>[sliceCount];

            fixed (float* denPtrFixed = density)
            {
                float* denPtr = denPtrFixed;

                Parallel.For(0, sliceCount, iz =>
                {
                    var vList = new System.Collections.Generic.List<Vector3>(256);
                    var tList = new System.Collections.Generic.List<int>(256);

                    for (int iy = 0; iy < res - 1; iy++)
                    for (int ix = 0; ix < res - 1; ix++)
                    {
                        int i0 = iz * resRes + iy * res + ix;
                        float c0 = denPtr[i0];
                        float c1 = denPtr[i0 + 1];
                        float c2 = denPtr[i0 + res + 1];
                        float c3 = denPtr[i0 + res];
                        float c4 = denPtr[i0 + resRes];
                        float c5 = denPtr[i0 + resRes + 1];
                        float c6 = denPtr[i0 + resRes + res + 1];
                        float c7 = denPtr[i0 + resRes + res];

                        int cubeIndex = 0;
                        if (c0 < threshold) cubeIndex |= 1;
                        if (c1 < threshold) cubeIndex |= 2;
                        if (c2 < threshold) cubeIndex |= 4;
                        if (c3 < threshold) cubeIndex |= 8;
                        if (c4 < threshold) cubeIndex |= 16;
                        if (c5 < threshold) cubeIndex |= 32;
                        if (c6 < threshold) cubeIndex |= 64;
                        if (c7 < threshold) cubeIndex |= 128;

                        if (cubeIndex == 0 || cubeIndex == 255) continue;
                        int edgeBits = MCEdgeTable[cubeIndex];
                        if (edgeBits == 0) continue;

                        var ev = new Vector3[12];
                        float* corners = stackalloc float[] { c0, c1, c2, c3, c4, c5, c6, c7 };

                        for (int e = 0; e < 12; e++)
                        {
                            if ((edgeBits & (1 << e)) == 0) continue;
                            int a = MCEdgeA[e], b = MCEdgeB[e];
                            float va = corners[a], vb = corners[b];
                            float denom = vb - va;
                            float t = MathF.Abs(denom) > 1e-8f
                                ? Math.Clamp((threshold - va) / denom, 0f, 1f)
                                : 0.5f;

                            float ax = ix + MCCornerX[a], ay = iy + MCCornerY[a], az = iz + MCCornerZ[a];
                            float bx = ix + MCCornerX[b], by = iy + MCCornerY[b], bz = iz + MCCornerZ[b];

                            ev[e] = new Vector3(
                                (ax + t * (bx - ax)) * voxSize - 0.5f,
                                (ay + t * (by - ay)) * voxSize - 0.5f,
                                (az + t * (bz - az)) * voxSize - 0.5f);
                        }

                        for (int ti = 0; ti < 16; ti += 3)
                        {
                            int e0 = MCTriTable[cubeIndex * 16 + ti];
                            if (e0 < 0) break;
                            int e1 = MCTriTable[cubeIndex * 16 + ti + 1];
                            int e2 = MCTriTable[cubeIndex * 16 + ti + 2];

                            int baseIdx = vList.Count;
                            vList.Add(ev[e0]);
                            vList.Add(ev[e2]);
                            vList.Add(ev[e1]);
                            tList.Add(baseIdx);
                            tList.Add(baseIdx + 1);
                            tList.Add(baseIdx + 2);
                        }
                    }

                    sliceVerts[iz] = vList;
                    sliceTris[iz] = tList;
                });
            }

            int totalV = 0, totalT = 0;
            for (int i = 0; i < sliceCount; i++)
            {
                totalV += sliceVerts[i].Count;
                totalT += sliceTris[i].Count;
            }

            vertices = new Vector3[totalV];
            triangles = new int[totalT];
            int vOff = 0, tOff = 0;
            for (int i = 0; i < sliceCount; i++)
            {
                var sv = sliceVerts[i];
                var st = sliceTris[i];
                int baseV = vOff;
                sv.CopyTo(0, vertices, vOff, sv.Count);
                vOff += sv.Count;
                for (int j = 0; j < st.Count; j++)
                    triangles[tOff + j] = st[j] + baseV;
                tOff += st.Count;
            }
        }

        private static readonly int[] MCCornerX = { 0, 1, 1, 0, 0, 1, 1, 0 };
        private static readonly int[] MCCornerY = { 0, 0, 1, 1, 0, 0, 1, 1 };
        private static readonly int[] MCCornerZ = { 0, 0, 0, 0, 1, 1, 1, 1 };
        private static readonly int[] MCEdgeA = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
        private static readonly int[] MCEdgeB = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

        private static readonly int[] MCEdgeTable = CpuMeshExtractor.EdgeTablePublic;
        private static readonly int[] MCTriTable = CpuMeshExtractor.TriTablePublic;

        #endregion
    }
}
#endif
