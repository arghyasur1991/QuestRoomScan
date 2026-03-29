using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Per-vertex transformation progress buffer with adjacency-based spread.
    /// Drives the <c>_TransformTex</c> input for the RoomTransform shader.
    /// Supports two modes: Rift (global sweep) and Decay (persistent per-vertex).
    /// </summary>
    public class TransformationState
    {
        /// <summary>Per-vertex progress values (0 = real, 1 = fully themed).</summary>
        public float[] Progress { get; private set; }

        /// <summary>Per-vertex surface type classification (written by RoomUnderstanding).</summary>
        public byte[] SurfaceTypes { get; private set; }

        public int VertexCount => Progress?.Length ?? 0;

        // Adjacency for spread (vertex index -> list of neighbor vertex indices)
        private List<int>[] _adjacency;

        // Spread tuning
        private float _spreadThreshold = 0.3f;
        private float _spreadRate = 0.5f;

        /// <summary>
        /// Initialise for a mesh with the given vertex count.  Resets all
        /// progress to zero and rebuilds adjacency from triangle indices.
        /// </summary>
        public void Init(int vertexCount, int[] triangles)
        {
            Progress = new float[vertexCount];
            SurfaceTypes = new byte[vertexCount];
            BuildAdjacency(vertexCount, triangles);
        }

        /// <summary>
        /// Initialise from an existing mesh.
        /// </summary>
        public void Init(Mesh mesh)
        {
            if (mesh == null) return;
            Init(mesh.vertexCount, mesh.triangles);
        }

        /// <summary>Set all vertices to the given progress value (Rift mode).</summary>
        public void SetGlobal(float value)
        {
            if (Progress == null) return;
            value = Mathf.Clamp01(value);
            for (int i = 0; i < Progress.Length; i++)
                Progress[i] = value;
        }

        /// <summary>
        /// Seed a specific vertex to full progress.  Use as the origin for
        /// adjacency-based spread.
        /// </summary>
        public void Seed(int vertexIndex, float value = 1f)
        {
            if (Progress == null || vertexIndex < 0 || vertexIndex >= Progress.Length) return;
            Progress[vertexIndex] = Mathf.Clamp01(value);
        }

        /// <summary>
        /// Run one tick of adjacency-based spread.  Vertices above
        /// <paramref name="threshold"/> influence their neighbours.
        /// Call once per frame or at a fixed rate.
        /// </summary>
        public void SpreadTick(float deltaTime, float threshold = -1f, float rate = -1f)
        {
            if (Progress == null || _adjacency == null) return;
            if (threshold < 0f) threshold = _spreadThreshold;
            if (rate < 0f) rate = _spreadRate;

            float[] delta = new float[Progress.Length];

            for (int v = 0; v < Progress.Length; v++)
            {
                if (Progress[v] < threshold) continue;
                var neighbors = _adjacency[v];
                if (neighbors == null) continue;
                for (int n = 0; n < neighbors.Count; n++)
                {
                    int nb = neighbors[n];
                    if (Progress[nb] < Progress[v])
                    {
                        float push = (Progress[v] - Progress[nb]) * rate * deltaTime;
                        delta[nb] = Mathf.Max(delta[nb], push);
                    }
                }
            }

            for (int i = 0; i < Progress.Length; i++)
                Progress[i] = Mathf.Clamp01(Progress[i] + delta[i]);
        }

        /// <summary>Reset all progress to zero.</summary>
        public void Reset()
        {
            if (Progress != null)
                Array.Clear(Progress, 0, Progress.Length);
        }

        // ─────────────────────────────────────────────────────────────
        //  Persistence
        // ─────────────────────────────────────────────────────────────

        private const int FileVersion = 1;

        public byte[] Serialize()
        {
            if (Progress == null) return null;
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            bw.Write(FileVersion);
            bw.Write(Progress.Length);
            for (int i = 0; i < Progress.Length; i++)
                bw.Write(Progress[i]);
            bw.Write(SurfaceTypes.Length);
            for (int i = 0; i < SurfaceTypes.Length; i++)
                bw.Write(SurfaceTypes[i]);
            return ms.ToArray();
        }

        /// <summary>
        /// Restore progress from a previously serialized blob.
        /// Returns false if the data is invalid or the vertex count doesn't match.
        /// </summary>
        public bool Deserialize(byte[] data)
        {
            if (data == null || data.Length < 8) return false;
            try
            {
                using var ms = new MemoryStream(data);
                using var br = new BinaryReader(ms);
                int version = br.ReadInt32();
                if (version != FileVersion) return false;

                int count = br.ReadInt32();
                if (Progress == null || count != Progress.Length) return false;

                for (int i = 0; i < count; i++)
                    Progress[i] = br.ReadSingle();

                int stCount = br.ReadInt32();
                if (stCount == SurfaceTypes.Length)
                {
                    for (int i = 0; i < stCount; i++)
                        SurfaceTypes[i] = br.ReadByte();
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  UV-space texture generation for shader
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Bake per-vertex progress and surface type into a Texture2D that can be
        /// sampled in the shader via the mesh UVs.  R = progress, G = surface type / 255.
        /// Uses rasterisation into UV space — each triangle is rasterised and its
        /// vertices' values interpolated across texels.  For simplicity this initial
        /// version writes per-vertex into a 1D strip texture (width = vertexCount, height = 1)
        /// using TEXCOORD1 auto-assigned as (vertexIndex / count, 0).
        ///
        /// In the first milestone we skip the UV-space bake and instead feed
        /// <c>_TransformGlobal</c> directly to the shader for uniform progress.
        /// This method is here for the Decay-mode expansion.
        /// </summary>
        public Texture2D BakeToTexture()
        {
            if (Progress == null || Progress.Length == 0) return null;
            int w = Progress.Length;
            var tex = new Texture2D(w, 1, TextureFormat.RG16, false)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            var pixels = new Color[w];
            for (int i = 0; i < w; i++)
                pixels[i] = new Color(Progress[i], SurfaceTypes[i] / 255f, 0, 1);
            tex.SetPixels(pixels);
            tex.Apply(false);
            return tex;
        }

        // ─────────────────────────────────────────────────────────────
        //  Adjacency
        // ─────────────────────────────────────────────────────────────

        private void BuildAdjacency(int vertexCount, int[] triangles)
        {
            _adjacency = new List<int>[vertexCount];
            if (triangles == null) return;

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
                AddEdge(a, b);
                AddEdge(b, c);
                AddEdge(c, a);
            }
        }

        private void AddEdge(int a, int b)
        {
            _adjacency[a] ??= new List<int>(6);
            _adjacency[b] ??= new List<int>(6);
            if (!_adjacency[a].Contains(b)) _adjacency[a].Add(b);
            if (!_adjacency[b].Contains(a)) _adjacency[b].Add(a);
        }
    }
}
