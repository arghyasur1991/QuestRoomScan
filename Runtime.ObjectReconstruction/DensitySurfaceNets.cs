#if HAS_AI_INFERENCE
using System;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// CPU-side Surface Nets mesh extraction from a density field.
    /// Adapted from the GPU SurfaceNetsExtract pipeline but runs on the CPU
    /// since the density data is already readback from Sentis. For production,
    /// this should be replaced with a GPU compute variant using DensitySurfaceNets.compute.
    /// </summary>
    internal sealed class DensitySurfaceNets : IDisposable
    {
        private readonly int _resolution;
        private readonly float _threshold;
        private readonly ComputeShader _shader;

        private static readonly int3[] CornerOffsets =
        {
            new(0, 0, 0), new(1, 0, 0), new(1, 0, 1), new(0, 0, 1),
            new(0, 1, 0), new(1, 1, 0), new(1, 1, 1), new(0, 1, 1)
        };

        private static readonly int[] EdgeA = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
        private static readonly int[] EdgeB = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

        internal DensitySurfaceNets(ComputeShader shader, int resolution, float threshold)
        {
            _shader = shader;
            _resolution = resolution;
            _threshold = threshold;
        }

        internal Mesh Extract(float[] density, Color[] colors)
        {
            int res = _resolution;
            var vertMap = new int[res * res * res];
            for (int i = 0; i < vertMap.Length; i++) vertMap[i] = -1;

            var vertices = new System.Collections.Generic.List<Vector3>();
            var vertColors = new System.Collections.Generic.List<Color>();
            var indices = new System.Collections.Generic.List<int>();

            float voxSize = 1f / res;

            for (int z = 0; z < res - 1; z++)
            for (int y = 0; y < res - 1; y++)
            for (int x = 0; x < res - 1; x++)
            {
                int flatIdx = x + y * res + z * res * res;
                Vector3 posSum = Vector3.zero;
                int crossings = 0;

                for (int e = 0; e < 12; e++)
                {
                    var cA = new int3(x, y, z) + CornerOffsets[EdgeA[e]];
                    var cB = new int3(x, y, z) + CornerOffsets[EdgeB[e]];

                    float dA = SampleDensity(density, cA, res);
                    float dB = SampleDensity(density, cB, res);

                    bool insideA = dA >= _threshold;
                    bool insideB = dB >= _threshold;
                    if (insideA == insideB) continue;

                    float t = (dA - _threshold) / (dA - dB);
                    var posCoord = new Vector3(cA.x, cA.y, cA.z) +
                                   t * new Vector3(cB.x - cA.x, cB.y - cA.y, cB.z - cA.z);
                    posSum += posCoord;
                    crossings++;
                }

                if (crossings < 3) continue;

                posSum /= crossings;
                var worldPos = (posSum + Vector3.one * 0.5f - Vector3.one * res * 0.5f) * voxSize;

                vertMap[flatIdx] = vertices.Count;
                vertices.Add(worldPos);

                int colorIdx = Mathf.Clamp(
                    Mathf.RoundToInt(posSum.x) +
                    Mathf.RoundToInt(posSum.y) * res +
                    Mathf.RoundToInt(posSum.z) * res * res,
                    0, colors.Length - 1);
                vertColors.Add(colors[colorIdx]);
            }

            for (int z = 0; z < res - 1; z++)
            for (int y = 0; y < res - 1; y++)
            for (int x = 0; x < res - 1; x++)
            {
                TryEmitQuad(vertMap, density, indices, x, y, z, 1, 0, 0, 0, 0, 1, 0, 1, 0, res);
                TryEmitQuad(vertMap, density, indices, x, y, z, 0, 1, 0, 1, 0, 0, 0, 0, 1, res);
                TryEmitQuad(vertMap, density, indices, x, y, z, 0, 0, 1, 0, 1, 0, 1, 0, 0, res);
            }

            var mesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.SetVertices(vertices);
            mesh.SetColors(vertColors);
            mesh.SetTriangles(indices, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            Logger.Info($"[DensitySurfaceNets] Extracted mesh: {vertices.Count} verts, {indices.Count / 3} tris");
            return mesh;
        }

        private void TryEmitQuad(
            int[] vertMap, float[] density,
            System.Collections.Generic.List<int> indices,
            int x, int y, int z,
            int ax, int ay, int az,
            int d1x, int d1y, int d1z,
            int d2x, int d2y, int d2z,
            int res)
        {
            int nx = x + ax, ny = y + ay, nz = z + az;
            if (nx >= res || ny >= res || nz >= res) return;
            if (x - d1x < 0 || y - d1y < 0 || z - d1z < 0) return;
            if (x - d2x < 0 || y - d2y < 0 || z - d2z < 0) return;

            float va = SampleDensity(density, new int3(x, y, z), res);
            float vb = SampleDensity(density, new int3(nx, ny, nz), res);
            bool insideA = va >= _threshold;
            bool insideB = vb >= _threshold;
            if (insideA == insideB) return;

            int a = vertMap[Flatten(x, y, z, res)];
            int b = vertMap[Flatten(x - d1x, y - d1y, z - d1z, res)];
            int c = vertMap[Flatten(x - d1x - d2x, y - d1y - d2y, z - d1z - d2z, res)];
            int d = vertMap[Flatten(x - d2x, y - d2y, z - d2z, res)];
            if (a < 0 || b < 0 || c < 0 || d < 0) return;

            if (insideA)
            {
                indices.Add(c); indices.Add(b); indices.Add(a);
                indices.Add(d); indices.Add(c); indices.Add(a);
            }
            else
            {
                indices.Add(a); indices.Add(c); indices.Add(d);
                indices.Add(a); indices.Add(b); indices.Add(c);
            }
        }

        private static float SampleDensity(float[] density, int3 coord, int res)
        {
            int idx = coord.x + coord.y * res + coord.z * res * res;
            if (idx < 0 || idx >= density.Length) return 0;
            return density[idx];
        }

        private static int Flatten(int x, int y, int z, int res) => x + y * res + z * res * res;

        public void Dispose() { }

        private struct int3
        {
            public int x, y, z;
            public int3(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }
            public static int3 operator +(int3 a, int3 b) => new(a.x + b.x, a.y + b.y, a.z + b.z);
        }
    }
}
#endif
