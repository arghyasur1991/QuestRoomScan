#if HAS_ONNXRUNTIME
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Pure-CPU mesh extraction pipeline that exactly matches the Python
    /// reconstruct_compare.py / _extract_mesh_onnx_decoder flow.
    /// Runs entirely on background threads — zero GPU usage.
    ///
    /// Performance: uses Parallel.For + unsafe pointers for multi-core throughput
    /// on triplane sampling and marching cubes. Buffer pooling eliminates GC pressure.
    /// </summary>
    internal sealed class CpuMeshExtractor
    {
        private readonly int _resolution;
        private readonly float _threshold;

        internal CpuMeshExtractor(int resolution, float threshold)
        {
            _resolution = resolution;
            _threshold = threshold;
        }

        internal async Task<Mesh> ExtractAsync(
            float[] sceneCodes, int numPlanes, int channels, int planeH, int planeW,
            OrtDecoderModel decoder, CancellationToken ct)
        {
            int res = _resolution;
            int featureDim = numPlanes * channels;
            const int chunkSize = 131072;

            float halfW = planeW * 0.5f - 0.5f;
            float halfH = planeH * 0.5f - 0.5f;

            // ---- Coarse-to-fine density evaluation ----
            int coarseRes = Math.Max(res / 4, 16);
            int coarseTotal = coarseRes * coarseRes * coarseRes;
            int fineRatio = res / coarseRes;

            // Coarse pass: evaluate density on low-res grid
            var coarseDensity = new float[coarseTotal];
            var featureBuf = new float[chunkSize * featureDim];
            float coarseInvResM1 = 1f / (coarseRes - 1);

            for (int start = 0; start < coarseTotal; start += chunkSize)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(chunkSize, coarseTotal - start);

                SampleGridChunkParallel(sceneCodes, featureBuf, start, count,
                    coarseRes, coarseInvResM1, halfW, halfH, channels, planeH, planeW);

                float[] decoderOut = await decoder.RunChunkAsync(featureBuf, count);
                ApplyDensityActivationParallel(decoderOut, coarseDensity, start, count);
            }
            await AsyncHelper.YieldFrame();

            // Build occupied mask: threshold + 1-voxel dilation
            var occupied = await Task.Run(() =>
                BuildOccupiedMask(coarseDensity, coarseRes, _threshold));
            ct.ThrowIfCancellationRequested();

            int occupiedCount = 0;
            for (int i = 0; i < occupied.Length; i++)
                if (occupied[i]) occupiedCount++;

            float occupiedPct = 100f * occupiedCount / coarseTotal;
            Logger.Info($"[CpuMeshExtractor] Coarse {coarseRes}^3: {occupiedCount}/{coarseTotal} " +
                $"occupied ({occupiedPct:F1}%), fine ratio {fineRatio}x");

            // Fine pass: only sample sub-voxels inside occupied coarse cells
            int totalPoints = res * res * res;
            var density = new float[totalPoints];
            float fineInvResM1 = 1f / (res - 1);

            var fineIndices = BuildFineIndices(occupied, coarseRes, fineRatio, res);
            int fineCount = fineIndices.Length;
            Logger.Info($"[CpuMeshExtractor] Fine pass: {fineCount}/{totalPoints} points " +
                $"({100f * fineCount / totalPoints:F1}%)");

            for (int start = 0; start < fineCount; start += chunkSize)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(chunkSize, fineCount - start);

                SampleIndexedGridChunkParallel(sceneCodes, featureBuf, fineIndices, start, count,
                    res, fineInvResM1, halfW, halfH, channels, planeH, planeW);

                float[] decoderOut = await decoder.RunChunkAsync(featureBuf, count);
                ApplyIndexedDensityActivationParallel(decoderOut, density, fineIndices, start, count);

                await AsyncHelper.YieldFrame();
            }

            // Marching cubes on full-res density (unsampled voxels stay 0 = below threshold)
            Vector3[] verts = null;
            int[] tris = null;
            await Task.Run(() => MarchingCubesParallel(density, res, _threshold, out verts, out tris));
            ct.ThrowIfCancellationRequested();

            if (verts.Length == 0)
                return new Mesh();

            // Vertex colors
            var colors = new Color[verts.Length];
            for (int start = 0; start < verts.Length; start += chunkSize)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(chunkSize, verts.Length - start);

                if (featureBuf.Length < count * featureDim)
                    featureBuf = new float[count * featureDim];

                SamplePositionsParallel(sceneCodes, featureBuf, verts, start, count,
                    halfW, halfH, channels, planeH, planeW);

                float[] decoderOut = await decoder.RunChunkAsync(featureBuf, count);
                ApplyColorActivationParallel(decoderOut, colors, start, count);

                await AsyncHelper.YieldFrame();
            }

            var mesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.SetColors(colors);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        #region Parallel Triplane Sampling

        private static unsafe void SampleGridChunkParallel(
            float[] sceneCodes, float[] output, int startIdx, int count,
            int res, float invResM1, float halfW, float halfH,
            int channels, int planeH, int planeW)
        {
            int featureDim = 3 * channels;
            int resRes = res * res;

            fixed (float* scPtr = sceneCodes, outPtr = output)
            {
                float* scLocal = scPtr;
                float* outLocal = outPtr;

                Parallel.For(0, count, i =>
                {
                    int globalIdx = startIdx + i;
                    int ix = globalIdx % res;
                    int iy = (globalIdx / res) % res;
                    int iz = globalIdx / resRes;

                    float x = ix * invResM1 - 0.5f;
                    float y = iy * invResM1 - 0.5f;
                    float z = iz * invResM1 - 0.5f;

                    float* dst = outLocal + i * featureDim;
                    SampleTriplaneFeaturesUnsafe(scLocal, x, y, z,
                        channels, planeH, planeW, halfW, halfH, dst);
                });
            }
        }

        private static unsafe void SamplePositionsParallel(
            float[] sceneCodes, float[] output, Vector3[] positions, int posStart, int count,
            float halfW, float halfH, int channels, int planeH, int planeW)
        {
            int featureDim = 3 * channels;

            fixed (float* scPtr = sceneCodes, outPtr = output)
            fixed (Vector3* posPtr = positions)
            {
                float* scLocal = scPtr;
                float* outLocal = outPtr;
                Vector3* posLocal = posPtr;

                Parallel.For(0, count, i =>
                {
                    var v = posLocal[posStart + i];
                    float* dst = outLocal + i * featureDim;
                    SampleTriplaneFeaturesUnsafe(scLocal, v.x, v.y, v.z,
                        channels, planeH, planeW, halfW, halfH, dst);
                });
            }
        }

        /// <summary>
        /// Bilinear-sample 3 triplane planes at (x,y,z) in [-0.5, 0.5].
        /// Matches F.grid_sample(align_corners=False, padding_mode='zeros').
        /// All pointer arithmetic — no bounds checking.
        /// </summary>
        private static unsafe void SampleTriplaneFeaturesUnsafe(
            float* sceneCodes, float x, float y, float z,
            int channels, int planeH, int planeW,
            float halfW, float halfH, float* output)
        {
            int planeStride = channels * planeH * planeW;
            int chanStride = planeH * planeW;

            // Plane 0: (x, y), Plane 1: (x, z), Plane 2: (y, z)
            float* planeBase0 = sceneCodes;
            float* planeBase1 = sceneCodes + planeStride;
            float* planeBase2 = sceneCodes + planeStride * 2;

            SamplePlaneUnsafe(planeBase0, x * planeW + halfW, y * planeH + halfH,
                channels, planeH, planeW, chanStride, output);
            SamplePlaneUnsafe(planeBase1, x * planeW + halfW, z * planeH + halfH,
                channels, planeH, planeW, chanStride, output + channels);
            SamplePlaneUnsafe(planeBase2, y * planeW + halfW, z * planeH + halfH,
                channels, planeH, planeW, chanStride, output + channels * 2);
        }

        private static unsafe void SamplePlaneUnsafe(
            float* planeBase, float u, float v,
            int channels, int planeH, int planeW, int chanStride, float* output)
        {
            int u0 = (int)MathF.Floor(u);
            int v0 = (int)MathF.Floor(v);
            int u1 = u0 + 1;
            int v1 = v0 + 1;

            float fu = u - u0;
            float fv = v - v0;

            float w00 = (1f - fu) * (1f - fv);
            float w10 = fu * (1f - fv);
            float w01 = (1f - fu) * fv;
            float w11 = fu * fv;

            bool u0ok = (uint)u0 < (uint)planeW;
            bool u1ok = (uint)u1 < (uint)planeW;
            bool v0ok = (uint)v0 < (uint)planeH;
            bool v1ok = (uint)v1 < (uint)planeH;

            bool has00 = u0ok & v0ok;
            bool has10 = u1ok & v0ok;
            bool has01 = u0ok & v1ok;
            bool has11 = u1ok & v1ok;

            int idx00 = v0 * planeW + u0;
            int idx10 = v0 * planeW + u1;
            int idx01 = v1 * planeW + u0;
            int idx11 = v1 * planeW + u1;

            for (int c = 0; c < channels; c++)
            {
                float* ch = planeBase + c * chanStride;
                float val = 0f;
                if (has00) val += w00 * ch[idx00];
                if (has10) val += w10 * ch[idx10];
                if (has01) val += w01 * ch[idx01];
                if (has11) val += w11 * ch[idx11];
                output[c] = val;
            }
        }

        private static unsafe void SampleIndexedGridChunkParallel(
            float[] sceneCodes, float[] output, int[] globalIndices, int indexStart, int count,
            int res, float invResM1, float halfW, float halfH,
            int channels, int planeH, int planeW)
        {
            int featureDim = 3 * channels;
            int resRes = res * res;

            fixed (float* scPtr = sceneCodes, outPtr = output)
            fixed (int* idxPtr = globalIndices)
            {
                float* scLocal = scPtr;
                float* outLocal = outPtr;
                int* idxLocal = idxPtr;

                Parallel.For(0, count, i =>
                {
                    int globalIdx = idxLocal[indexStart + i];
                    int ix = globalIdx % res;
                    int iy = (globalIdx / res) % res;
                    int iz = globalIdx / resRes;

                    float x = ix * invResM1 - 0.5f;
                    float y = iy * invResM1 - 0.5f;
                    float z = iz * invResM1 - 0.5f;

                    float* dst = outLocal + i * featureDim;
                    SampleTriplaneFeaturesUnsafe(scLocal, x, y, z,
                        channels, planeH, planeW, halfW, halfH, dst);
                });
            }
        }

        #endregion

        #region Coarse-to-Fine Helpers

        private static bool[] BuildOccupiedMask(float[] coarseDensity, int coarseRes, float threshold)
        {
            int total = coarseRes * coarseRes * coarseRes;
            var raw = new bool[total];
            for (int i = 0; i < total; i++)
                raw[i] = coarseDensity[i] >= threshold;

            // Dilate by 1 voxel in all 6 directions to capture surfaces
            var dilated = new bool[total];
            int cr2 = coarseRes * coarseRes;
            for (int iz = 0; iz < coarseRes; iz++)
            for (int iy = 0; iy < coarseRes; iy++)
            for (int ix = 0; ix < coarseRes; ix++)
            {
                int idx = iz * cr2 + iy * coarseRes + ix;
                if (raw[idx]) { dilated[idx] = true; continue; }

                if ((ix > 0 && raw[idx - 1]) ||
                    (ix < coarseRes - 1 && raw[idx + 1]) ||
                    (iy > 0 && raw[idx - coarseRes]) ||
                    (iy < coarseRes - 1 && raw[idx + coarseRes]) ||
                    (iz > 0 && raw[idx - cr2]) ||
                    (iz < coarseRes - 1 && raw[idx + cr2]))
                {
                    dilated[idx] = true;
                }
            }

            return dilated;
        }

        private static int[] BuildFineIndices(bool[] occupied, int coarseRes, int fineRatio, int fineRes)
        {
            var indices = new List<int>(occupied.Length * fineRatio * fineRatio * fineRatio / 4);
            int cr2 = coarseRes * coarseRes;
            int fr2 = fineRes * fineRes;

            for (int cz = 0; cz < coarseRes; cz++)
            for (int cy = 0; cy < coarseRes; cy++)
            for (int cx = 0; cx < coarseRes; cx++)
            {
                if (!occupied[cz * cr2 + cy * coarseRes + cx]) continue;

                int fxStart = cx * fineRatio;
                int fyStart = cy * fineRatio;
                int fzStart = cz * fineRatio;
                int fxEnd = Math.Min(fxStart + fineRatio, fineRes);
                int fyEnd = Math.Min(fyStart + fineRatio, fineRes);
                int fzEnd = Math.Min(fzStart + fineRatio, fineRes);

                for (int fz = fzStart; fz < fzEnd; fz++)
                for (int fy = fyStart; fy < fyEnd; fy++)
                for (int fx = fxStart; fx < fxEnd; fx++)
                {
                    indices.Add(fz * fr2 + fy * fineRes + fx);
                }
            }

            return indices.ToArray();
        }

        #endregion

        #region Parallel Activations

        private static unsafe void ApplyDensityActivationParallel(
            float[] decoderOut, float[] density, int dstOffset, int count)
        {
            fixed (float* decPtr = decoderOut, denPtr = density)
            {
                float* decLocal = decPtr;
                float* denLocal = denPtr + dstOffset;

                Parallel.For(0, count, i =>
                {
                    denLocal[i] = MathF.Exp(decLocal[i * 4] - 1f);
                });
            }
        }

        private static unsafe void ApplyIndexedDensityActivationParallel(
            float[] decoderOut, float[] density, int[] globalIndices, int indexStart, int count)
        {
            fixed (float* decPtr = decoderOut, denPtr = density)
            fixed (int* idxPtr = globalIndices)
            {
                float* decLocal = decPtr;
                float* denLocal = denPtr;
                int* idxLocal = idxPtr;

                Parallel.For(0, count, i =>
                {
                    denLocal[idxLocal[indexStart + i]] = MathF.Exp(decLocal[i * 4] - 1f);
                });
            }
        }

        private static void ApplyColorActivationParallel(
            float[] decoderOut, Color[] colors, int dstOffset, int count)
        {
            Parallel.For(0, count, i =>
            {
                int b = i * 4;
                colors[dstOffset + i] = new Color(
                    Sigmoid(decoderOut[b + 1]),
                    Sigmoid(decoderOut[b + 2]),
                    Sigmoid(decoderOut[b + 3]));
            });
        }

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        #endregion

        #region Parallel Marching Cubes

        private static unsafe void MarchingCubesParallel(
            float[] density, int res, float threshold,
            out Vector3[] vertices, out int[] triangles)
        {
            float voxSize = 1f / (res - 1);
            float halfRes = res / 2f;
            int sliceCount = res - 1;
            int resRes = res * res;

            var sliceVerts = new List<Vector3>[sliceCount];
            var sliceTris = new List<int>[sliceCount];

            fixed (float* denPtrFixed = density)
            {
                float* denPtr = denPtrFixed;

                Parallel.For(0, sliceCount, iz =>
                {
                    var vList = new List<Vector3>(1024);
                    var tList = new List<int>(1024);

                    for (int iy = 0; iy < res - 1; iy++)
                    for (int ix = 0; ix < res - 1; ix++)
                    {
                        int i0 = iz * resRes + iy * res + ix;
                        int i1 = i0 + 1;
                        int i2 = i0 + res + 1;
                        int i3 = i0 + res;
                        int i4 = i0 + resRes;
                        int i5 = i4 + 1;
                        int i6 = i4 + res + 1;
                        int i7 = i4 + res;

                        float c0 = denPtr[i0], c1 = denPtr[i1];
                        float c2 = denPtr[i2], c3 = denPtr[i3];
                        float c4 = denPtr[i4], c5 = denPtr[i5];
                        float c6 = denPtr[i6], c7 = denPtr[i7];

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
                        int edgeBits = EdgeTable[cubeIndex];
                        if (edgeBits == 0) continue;

                        var ev = new Vector3[12];
                        EmitMCCell(ix, iy, iz, c0, c1, c2, c3, c4, c5, c6, c7,
                            threshold, edgeBits, cubeIndex, voxSize, halfRes,
                            ev, vList, tList);
                    }

                    sliceVerts[iz] = vList;
                    sliceTris[iz] = tList;
                });
            }

            // Merge slices — sequential but fast (just array concatenation)
            int totalVerts = 0, totalTris = 0;
            for (int i = 0; i < sliceCount; i++)
            {
                totalVerts += sliceVerts[i].Count;
                totalTris += sliceTris[i].Count;
            }

            vertices = new Vector3[totalVerts];
            triangles = new int[totalTris];
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

        private static unsafe void EmitMCCell(
            int ix, int iy, int iz,
            float c0, float c1, float c2, float c3,
            float c4, float c5, float c6, float c7,
            float threshold, int edgeBits, int cubeIndex,
            float voxSize, float halfRes,
            Vector3[] ev, List<Vector3> vList, List<int> tList)
        {
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
                    (ax + t * (bx - ax) + 0.5f - halfRes) * voxSize,
                    (ay + t * (by - ay) + 0.5f - halfRes) * voxSize,
                    (az + t * (bz - az) + 0.5f - halfRes) * voxSize);
            }

            for (int ti = 0; ti < 16; ti += 3)
            {
                int e0 = TriTable[cubeIndex * 16 + ti];
                if (e0 < 0) break;
                int e1 = TriTable[cubeIndex * 16 + ti + 1];
                int e2 = TriTable[cubeIndex * 16 + ti + 2];

                int baseIdx = vList.Count;
                vList.Add(ev[e0]);
                vList.Add(ev[e2]);
                vList.Add(ev[e1]);

                tList.Add(baseIdx);
                tList.Add(baseIdx + 1);
                tList.Add(baseIdx + 2);
            }
        }

        private static int Flat(int x, int y, int z, int res) => z * res * res + y * res + x;

        #endregion

        #region MC Tables

        private static readonly int[] MCCornerX = { 0, 1, 1, 0, 0, 1, 1, 0 };
        private static readonly int[] MCCornerY = { 0, 0, 1, 1, 0, 0, 1, 1 };
        private static readonly int[] MCCornerZ = { 0, 0, 0, 0, 1, 1, 1, 1 };
        private static readonly int[] MCEdgeA = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
        private static readonly int[] MCEdgeB = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

        private static readonly int[] EdgeTable =
        {
            0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
            0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
            0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
            0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
            0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
            0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
            0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
            0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
            0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
            0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
            0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
            0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
            0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
            0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
            0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
            0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
            0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
            0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
            0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
            0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
            0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
            0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
            0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
            0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
            0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
            0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
            0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
            0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
            0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
            0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
            0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
            0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
        };

        private static readonly int[] TriTable =
        {
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1,
            3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1,
            3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1,
            3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1,
            9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1,
            1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1,
            9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1,
            2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1,
            8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1,
            9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1,
            4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1,
            3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1,
            1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1,
            4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1,
            4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1,
            9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1,
            1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1,
            5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1,
            2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1,
            9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1,
            0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1,
            2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1,
            10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1,
            4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1,
            5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1,
            5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,
            9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1,
            0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1,
            1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1,
            10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1,
            8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1,
            2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,
            7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1,
            9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1,
            2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1,
            11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1,
            9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1,
            5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1,
            11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1,
            11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1,
            1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1,
            9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1,
            5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1,
            2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1,
            0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1,
            5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1,
            6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1,
            0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1,
            3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1,
            6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1,
            5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1,
            1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1,
            10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1,
            6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1,
            1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1,
            8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1,
            7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1,
            3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1,
            5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1,
            0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1,
            9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1,
            8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1,
            5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1,
            0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1,
            6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1,
            10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1,
            10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1,
            8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1,
            1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1,
            3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1,
            0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,
            10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1,
            0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1,
            3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1,
            6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1,
            9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1,
            8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1,
            3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1,
            6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1,
            0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1,
            10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1,
            10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1,
            1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1,
            2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1,
            7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1,
            7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1,
            2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1,
            1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1,
            11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1,
            8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1,
            0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1,
            7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1,
            10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1,
            2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1,
            6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1,
            7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1,
            2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1,
            1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1,
            10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1,
            10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1,
            0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1,
            7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1,
            6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1,
            8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1,
            9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1,
            6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1,
            1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1,
            4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1,
            10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1,
            8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,
            0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1,
            1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1,
            8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1,
            10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1,
            4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1,
            10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1,
            5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1,
            11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1,
            9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1,
            6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1,
            7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1,
            3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1,
            7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1,
            9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1,
            3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1,
            6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1,
            9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1,
            1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1,
            4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1,
            7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1,
            6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1,
            3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1,
            0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1,
            6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1,
            1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1,
            0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1,
            11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1,
            6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1,
            5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1,
            9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1,
            1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1,
            1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1,
            10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1,
            0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1,
            5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1,
            10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1,
            11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1,
            0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1,
            9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1,
            7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1,
            2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,
            8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1,
            9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1,
            9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1,
            1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1,
            9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1,
            9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,
            5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1,
            0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1,
            10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1,
            2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1,
            0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1,
            0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1,
            9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1,
            5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1,
            3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1,
            5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1,
            8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1,
            0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1,
            9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1,
            0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1,
            1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1,
            3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1,
            4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1,
            9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1,
            11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1,
            11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1,
            2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1,
            9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1,
            3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1,
            1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1,
            4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1,
            4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1,
            0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1,
            3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1,
            3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1,
            0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1,
            9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1,
            1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
        };

        #endregion
    }
}
#endif
