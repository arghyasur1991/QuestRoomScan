#if HAS_ONNXRUNTIME
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    internal static class MeshPostProcess
    {
        /// <summary>
        /// Laplacian smoothing on vertex positions. Moves each vertex toward the
        /// centroid of its neighbors by <paramref name="lambda"/> per iteration.
        /// Operates in-place on the <paramref name="vertices"/> array.
        /// Uses Parallel.For for multi-core throughput.
        /// </summary>
        internal static void LaplacianSmooth(
            Vector3[] vertices, int[] triangles, int iterations, float lambda = 0.5f)
        {
            if (iterations <= 0 || vertices.Length == 0 || triangles.Length == 0)
                return;

            var adjacency = BuildAdjacency(vertices.Length, triangles);
            var buffer = new Vector3[vertices.Length];

            for (int iter = 0; iter < iterations; iter++)
            {
                Parallel.For(0, vertices.Length, i =>
                {
                    int start = adjacency.offsets[i];
                    int end = adjacency.offsets[i + 1];
                    int count = end - start;

                    if (count == 0)
                    {
                        buffer[i] = vertices[i];
                        return;
                    }

                    float cx = 0f, cy = 0f, cz = 0f;
                    for (int j = start; j < end; j++)
                    {
                        var n = vertices[adjacency.neighbors[j]];
                        cx += n.x;
                        cy += n.y;
                        cz += n.z;
                    }

                    float inv = 1f / count;
                    cx *= inv;
                    cy *= inv;
                    cz *= inv;

                    var v = vertices[i];
                    float t = 1f - lambda;
                    buffer[i] = new Vector3(
                        v.x * t + cx * lambda,
                        v.y * t + cy * lambda,
                        v.z * t + cz * lambda);
                });

                Array.Copy(buffer, vertices, vertices.Length);
            }
        }

        private struct CompactAdjacency
        {
            public int[] offsets;   // length = vertexCount + 1
            public int[] neighbors; // packed neighbor indices
        }

        private static CompactAdjacency BuildAdjacency(int vertexCount, int[] triangles)
        {
            var sets = new HashSet<int>[vertexCount];
            for (int i = 0; i < vertexCount; i++)
                sets[i] = new HashSet<int>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
                sets[a].Add(b); sets[a].Add(c);
                sets[b].Add(a); sets[b].Add(c);
                sets[c].Add(a); sets[c].Add(b);
            }

            var offsets = new int[vertexCount + 1];
            int total = 0;
            for (int i = 0; i < vertexCount; i++)
            {
                offsets[i] = total;
                total += sets[i].Count;
            }
            offsets[vertexCount] = total;

            var neighbors = new int[total];
            for (int i = 0; i < vertexCount; i++)
            {
                int off = offsets[i];
                foreach (int n in sets[i])
                    neighbors[off++] = n;
            }

            return new CompactAdjacency { offsets = offsets, neighbors = neighbors };
        }
    }
}
#endif
