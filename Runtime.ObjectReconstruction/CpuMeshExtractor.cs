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
    /// Pipeline:
    ///   1. Generate grid positions in [-0.5, 0.5]
    ///   2. Bilinear-sample triplane features (matches F.grid_sample align_corners=False, padding_mode='zeros')
    ///   3. Run ORT decoder → [density, r, g, b]
    ///   4. Density activation: exp(raw - 1)
    ///   5. Marching cubes on density volume
    ///   6. Re-sample triplane features at vertex positions
    ///   7. Run ORT decoder for vertex colors → sigmoid(rgb)
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

        /// <summary>
        /// Full CPU mesh extraction. Caller provides scene codes and a loaded decoder.
        /// Everything runs on background thread via Task.Run.
        /// </summary>
        internal async Task<Mesh> ExtractAsync(
            float[] sceneCodes, int numPlanes, int channels, int planeH, int planeW,
            OrtDecoderModel decoder, CancellationToken ct)
        {
            int res = _resolution;
            int totalPoints = res * res * res;
            int featureDim = numPlanes * channels;
            const int chunkSize = 65536;

            var density = new float[totalPoints];

            // Pass 1: density field
            for (int start = 0; start < totalPoints; start += chunkSize)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(chunkSize, totalPoints - start);

                var features = await Task.Run(() =>
                {
                    var feat = new float[count * featureDim];
                    for (int i = 0; i < count; i++)
                    {
                        int globalIdx = start + i;
                        int ix = globalIdx % res;
                        int iy = (globalIdx / res) % res;
                        int iz = globalIdx / (res * res);

                        float invResM1 = 1f / (res - 1);
                        float x = ix * invResM1 - 0.5f;
                        float y = iy * invResM1 - 0.5f;
                        float z = iz * invResM1 - 0.5f;

                        SampleTriplaneFeatures(sceneCodes, x, y, z,
                            numPlanes, channels, planeH, planeW,
                            feat, i * featureDim);
                    }
                    return feat;
                });

                float[] decoderOut = await decoder.RunChunkAsync(features, count);

                await Task.Run(() =>
                {
                    for (int i = 0; i < count; i++)
                    {
                        float raw = decoderOut[i * 4];
                        density[start + i] = MathF.Exp(raw - 1f);
                    }
                });

                await AsyncHelper.YieldFrame();
            }

            // Pass 2: marching cubes
            Vector3[] verts = null;
            int[] tris = null;
            await Task.Run(() => MarchingCubesCPU(density, res, _threshold, out verts, out tris));
            ct.ThrowIfCancellationRequested();

            if (verts.Length == 0)
                return new Mesh();

            // Pass 3: vertex colors
            var colors = new Color[verts.Length];
            for (int start = 0; start < verts.Length; start += chunkSize)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(chunkSize, verts.Length - start);
                int startCopy = start;

                var features = await Task.Run(() =>
                {
                    var feat = new float[count * featureDim];
                    for (int i = 0; i < count; i++)
                    {
                        var v = verts[startCopy + i];
                        SampleTriplaneFeatures(sceneCodes, v.x, v.y, v.z,
                            numPlanes, channels, planeH, planeW,
                            feat, i * featureDim);
                    }
                    return feat;
                });

                float[] decoderOut = await decoder.RunChunkAsync(features, count);

                await Task.Run(() =>
                {
                    for (int i = 0; i < count; i++)
                    {
                        int b = i * 4;
                        colors[startCopy + i] = new Color(
                            Sigmoid(decoderOut[b + 1]),
                            Sigmoid(decoderOut[b + 2]),
                            Sigmoid(decoderOut[b + 3]));
                    }
                });

                await AsyncHelper.YieldFrame();
            }

            // Build mesh on main thread
            var mesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.SetColors(colors);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        #region Triplane Sampling — matches F.grid_sample(align_corners=False, padding_mode='zeros')

        /// <summary>
        /// Bilinear-sample triplane features at position (x, y, z) in [-0.5, 0.5].
        /// Exactly replicates PyTorch F.grid_sample with align_corners=False, padding_mode='zeros'.
        /// Plane 0: (x, y), Plane 1: (x, z), Plane 2: (y, z).
        /// </summary>
        private static void SampleTriplaneFeatures(
            float[] sceneCodes, float x, float y, float z,
            int numPlanes, int channels, int planeH, int planeW,
            float[] output, int outOffset)
        {
            // Map from [-0.5, 0.5] → pixel coords matching align_corners=False:
            // px = (pos + 0.5) * W - 0.5   (since pos in [-0.5, 0.5] maps to x in [-1,1] as x = pos*2)
            // Equivalently: px = pos * W + W/2 - 0.5
            float halfW = planeW * 0.5f - 0.5f;
            float halfH = planeH * 0.5f - 0.5f;

            float u0 = x * planeW + halfW;
            float v0 = y * planeH + halfH;
            float u1 = x * planeW + halfW;
            float v1 = z * planeH + halfH;
            float u2 = y * planeW + halfW;
            float v2 = z * planeH + halfH;

            int off = outOffset;
            BilinearSampleZeroPad(sceneCodes, 0, channels, planeH, planeW, u0, v0, output, ref off);
            BilinearSampleZeroPad(sceneCodes, 1, channels, planeH, planeW, u1, v1, output, ref off);
            BilinearSampleZeroPad(sceneCodes, 2, channels, planeH, planeW, u2, v2, output, ref off);
        }

        /// <summary>
        /// Bilinear interpolation with zero-padding (matching PyTorch default padding_mode='zeros').
        /// Out-of-bounds samples contribute 0 instead of clamped edge values.
        /// </summary>
        private static void BilinearSampleZeroPad(
            float[] sceneCodes, int planeIdx, int channels, int planeH, int planeW,
            float u, float v, float[] output, ref int outIdx)
        {
            int u0 = (int)MathF.Floor(u);
            int v0 = (int)MathF.Floor(v);
            int u1 = u0 + 1;
            int v1 = v0 + 1;

            float fu = u - u0;
            float fv = v - v0;

            bool u0Valid = u0 >= 0 && u0 < planeW;
            bool u1Valid = u1 >= 0 && u1 < planeW;
            bool v0Valid = v0 >= 0 && v0 < planeH;
            bool v1Valid = v1 >= 0 && v1 < planeH;

            float w00 = (1f - fu) * (1f - fv);
            float w10 = fu * (1f - fv);
            float w01 = (1f - fu) * fv;
            float w11 = fu * fv;

            int planeBase = planeIdx * channels * planeH * planeW;

            for (int c = 0; c < channels; c++)
            {
                int chanBase = planeBase + c * planeH * planeW;
                float val = 0f;

                if (u0Valid && v0Valid) val += w00 * sceneCodes[chanBase + v0 * planeW + u0];
                if (u1Valid && v0Valid) val += w10 * sceneCodes[chanBase + v0 * planeW + u1];
                if (u0Valid && v1Valid) val += w01 * sceneCodes[chanBase + v1 * planeW + u0];
                if (u1Valid && v1Valid) val += w11 * sceneCodes[chanBase + v1 * planeW + u1];

                output[outIdx++] = val;
            }
        }

        #endregion

        #region Activations

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        #endregion

        #region CPU Marching Cubes — matches Python torchmcubes / skimage

        private static void MarchingCubesCPU(
            float[] density, int res, float threshold,
            out Vector3[] vertices, out int[] triangles)
        {
            var vertList = new List<Vector3>();
            var triList = new List<int>();
            float voxSize = 1f / (res - 1);

            for (int iz = 0; iz < res - 1; iz++)
            for (int iy = 0; iy < res - 1; iy++)
            for (int ix = 0; ix < res - 1; ix++)
            {
                float[] corners = new float[8];
                corners[0] = density[Flat(ix, iy, iz, res)];
                corners[1] = density[Flat(ix + 1, iy, iz, res)];
                corners[2] = density[Flat(ix + 1, iy + 1, iz, res)];
                corners[3] = density[Flat(ix, iy + 1, iz, res)];
                corners[4] = density[Flat(ix, iy, iz + 1, res)];
                corners[5] = density[Flat(ix + 1, iy, iz + 1, res)];
                corners[6] = density[Flat(ix + 1, iy + 1, iz + 1, res)];
                corners[7] = density[Flat(ix, iy + 1, iz + 1, res)];

                int cubeIndex = 0;
                for (int j = 0; j < 8; j++)
                    if (corners[j] < threshold) cubeIndex |= (1 << j);

                if (cubeIndex == 0 || cubeIndex == 255) continue;
                int edgeBits = EdgeTable[cubeIndex];
                if (edgeBits == 0) continue;

                var edgeVerts = new Vector3[12];
                for (int e = 0; e < 12; e++)
                {
                    if ((edgeBits & (1 << e)) == 0) continue;
                    int a = MCEdgeA[e], b = MCEdgeB[e];
                    float va = corners[a], vb = corners[b];
                    float denom = vb - va;
                    float t = MathF.Abs(denom) > 1e-8f
                        ? Math.Clamp((threshold - va) / denom, 0f, 1f)
                        : 0.5f;

                    var pA = CornerPos(ix, iy, iz, a);
                    var pB = CornerPos(ix, iy, iz, b);
                    var worldA = VoxToWorld(pA, res, voxSize);
                    var worldB = VoxToWorld(pB, res, voxSize);
                    edgeVerts[e] = Vector3.Lerp(worldA, worldB, t);
                }

                for (int ti = 0; ti < 16; ti += 3)
                {
                    int e0 = TriTable[cubeIndex * 16 + ti];
                    if (e0 < 0) break;
                    int e1 = TriTable[cubeIndex * 16 + ti + 1];
                    int e2 = TriTable[cubeIndex * 16 + ti + 2];

                    int baseIdx = vertList.Count;
                    vertList.Add(edgeVerts[e0]);
                    // Reversed winding to match GPU shader (normals point outward)
                    vertList.Add(edgeVerts[e2]);
                    vertList.Add(edgeVerts[e1]);

                    triList.Add(baseIdx);
                    triList.Add(baseIdx + 1);
                    triList.Add(baseIdx + 2);
                }
            }

            vertices = vertList.ToArray();
            triangles = triList.ToArray();
        }

        private static int Flat(int x, int y, int z, int res) => z * res * res + y * res + x;

        private static Vector3 CornerPos(int cellX, int cellY, int cellZ, int corner)
        {
            return new Vector3(
                cellX + MCCornerX[corner],
                cellY + MCCornerY[corner],
                cellZ + MCCornerZ[corner]);
        }

        private static Vector3 VoxToWorld(Vector3 c, int res, float voxSize)
        {
            return new Vector3(
                (c.x + 0.5f - res / 2f) * voxSize,
                (c.y + 0.5f - res / 2f) * voxSize,
                (c.z + 0.5f - res / 2f) * voxSize);
        }

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
