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
        /// Laplacian smoothing on vertex positions. First welds duplicate vertices
        /// (marching cubes emits per-triangle vertices with no sharing), then runs
        /// smoothing on the welded topology, and finally writes back to the original
        /// arrays so vertex colors remain correctly indexed.
        /// </summary>
        internal static void LaplacianSmooth(
            Vector3[] vertices, int[] triangles, int iterations, float lambda = 0.5f)
        {
            if (iterations <= 0 || vertices.Length == 0 || triangles.Length == 0)
                return;

            // Weld: map each vertex to a canonical index based on position
            WeldVertices(vertices, triangles,
                out Vector3[] welded, out int[] weldedTris, out int[] originalToWelded);

            Logger.Info($"[MeshPostProcess] Welded {vertices.Length} -> {welded.Length} unique verts");

            // Build adjacency on the welded mesh
            var adjacency = BuildAdjacency(welded.Length, weldedTris);
            var buffer = new Vector3[welded.Length];

            for (int iter = 0; iter < iterations; iter++)
            {
                Parallel.For(0, welded.Length, i =>
                {
                    int start = adjacency.offsets[i];
                    int end = adjacency.offsets[i + 1];
                    int count = end - start;

                    if (count == 0)
                    {
                        buffer[i] = welded[i];
                        return;
                    }

                    float cx = 0f, cy = 0f, cz = 0f;
                    for (int j = start; j < end; j++)
                    {
                        var n = welded[adjacency.neighbors[j]];
                        cx += n.x;
                        cy += n.y;
                        cz += n.z;
                    }

                    float inv = 1f / count;
                    cx *= inv;
                    cy *= inv;
                    cz *= inv;

                    var v = welded[i];
                    float t = 1f - lambda;
                    buffer[i] = new Vector3(
                        v.x * t + cx * lambda,
                        v.y * t + cy * lambda,
                        v.z * t + cz * lambda);
                });

                Array.Copy(buffer, welded, welded.Length);
            }

            // Write smoothed positions back to the original (unwelded) vertex array
            // so that vertex colors and other per-vertex data remain correctly indexed.
            for (int i = 0; i < vertices.Length; i++)
                vertices[i] = welded[originalToWelded[i]];
        }

        /// <summary>
        /// Welds vertices that share the same position (within floating point tolerance).
        /// Marching cubes emits per-triangle vertices, so many vertices are at identical positions.
        /// Uses spatial hashing for O(n) performance.
        /// </summary>
        private static void WeldVertices(
            Vector3[] vertices, int[] triangles,
            out Vector3[] weldedVerts, out int[] weldedTris, out int[] vertexMap)
        {
            const float cellSize = 1e-5f;
            float invCell = 1f / cellSize;

            var positionToIndex = new Dictionary<long, int>(vertices.Length / 3);
            var uniquePositions = new List<Vector3>(vertices.Length / 3);
            vertexMap = new int[vertices.Length];

            for (int i = 0; i < vertices.Length; i++)
            {
                var v = vertices[i];
                long hash = SpatialHash(v, invCell);

                if (positionToIndex.TryGetValue(hash, out int existingIdx))
                {
                    // Verify it's actually the same position (hash collision check)
                    var existing = uniquePositions[existingIdx];
                    float dx = v.x - existing.x, dy = v.y - existing.y, dz = v.z - existing.z;
                    if (dx * dx + dy * dy + dz * dz < 1e-10f)
                    {
                        vertexMap[i] = existingIdx;
                        continue;
                    }
                    // Hash collision with different position — use linear probe
                    hash = ResolveCollision(positionToIndex, uniquePositions, v, hash, invCell);
                    if (hash >= 0)
                    {
                        vertexMap[i] = positionToIndex[hash];
                        continue;
                    }
                }

                int newIdx = uniquePositions.Count;
                uniquePositions.Add(v);
                long finalHash = SpatialHash(v, invCell);
                if (!positionToIndex.ContainsKey(finalHash))
                    positionToIndex[finalHash] = newIdx;
                else
                    positionToIndex[finalHash + i * 73856093L] = newIdx; // collision fallback
                vertexMap[i] = newIdx;
            }

            weldedVerts = uniquePositions.ToArray();
            weldedTris = new int[triangles.Length];
            for (int i = 0; i < triangles.Length; i++)
                weldedTris[i] = vertexMap[triangles[i]];
        }

        private static long SpatialHash(Vector3 v, float invCell)
        {
            long ix = (long)MathF.Floor(v.x * invCell);
            long iy = (long)MathF.Floor(v.y * invCell);
            long iz = (long)MathF.Floor(v.z * invCell);
            return ix * 73856093L ^ iy * 19349663L ^ iz * 83492791L;
        }

        private static long ResolveCollision(
            Dictionary<long, int> map, List<Vector3> positions, Vector3 v,
            long baseHash, float invCell)
        {
            // Try a few offset hashes to find a matching position
            for (int probe = 1; probe <= 27; probe++)
            {
                long h = baseHash + probe * 73856093L;
                if (map.TryGetValue(h, out int idx))
                {
                    var existing = positions[idx];
                    float dx = v.x - existing.x, dy = v.y - existing.y, dz = v.z - existing.z;
                    if (dx * dx + dy * dy + dz * dz < 1e-10f)
                        return h;
                }
            }
            return -1;
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
