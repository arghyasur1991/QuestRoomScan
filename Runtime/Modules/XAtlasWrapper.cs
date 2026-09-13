using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// P/Invoke wrapper for the native xatlas UV unwrapping library.
    /// Call sequence: Create -> AddMesh -> Generate -> read results -> Destroy.
    /// Thread-safe for a single atlas instance per thread.
    /// </summary>
    internal static class XAtlasWrapper
    {
#if UNITY_IOS || UNITY_WEBGL
        private const string LIB = "__Internal";
#else
        private const string LIB = "xatlas";
#endif

        [DllImport(LIB)] private static extern IntPtr xatlas_create();
        [DllImport(LIB)] private static extern void xatlas_destroy(IntPtr atlas);
        [DllImport(LIB)] private static extern int xatlas_add_mesh(
            IntPtr atlas,
            float[] positions, int positionStride,
            float[] normals, int normalStride,
            int vertexCount,
            int[] indices, int indexCount);
        [DllImport(LIB)] private static extern void xatlas_generate(IntPtr atlas, int maxResolution);
        [DllImport(LIB)] private static extern void xatlas_generate_opts(
            IntPtr atlas,
            float maxChartArea, float maxBoundaryLength,
            float normalDeviationWeight, float roundnessWeight,
            float straightnessWeight, float normalSeamWeight,
            float textureSeamWeight, float maxCost,
            uint maxIterations,
            uint maxChartSize, uint padding,
            float texelsPerUnit, uint resolution,
            int bilinear, int blockAlign, int bruteForce,
            int rotateChartsToAxis, int rotateCharts);
        [DllImport(LIB)] private static extern void xatlas_get_atlas_dims(IntPtr atlas, out int width, out int height);
        [DllImport(LIB)] private static extern int xatlas_get_vertex_count(IntPtr atlas, int meshIndex);
        [DllImport(LIB)] private static extern int xatlas_get_index_count(IntPtr atlas, int meshIndex);
        [DllImport(LIB)] private static extern void xatlas_get_vertices(
            IntPtr atlas, int meshIndex,
            float[] uvs, int[] xrefs, int maxVerts);
        [DllImport(LIB)] private static extern void xatlas_get_indices(
            IntPtr atlas, int meshIndex,
            int[] outIndices, int maxIndices);
        [DllImport(LIB)] private static extern int xatlas_get_chart_count(IntPtr atlas, int meshIndex);
        [DllImport(LIB)] private static extern void xatlas_get_chart_indices(
            IntPtr atlas, int meshIndex,
            int[] chartIndices, int maxVerts);

        [DllImport(LIB)] private static extern int meshopt_simplify_with_attrs(
            uint[] destination,
            uint[] indices, int indexCount,
            float[] vertexPositions, int vertexCount, int vertexPositionsStride,
            float[] vertexAttributes, int vertexAttributesStride,
            float[] attributeWeights, int attributeCount,
            int targetIndexCount, float targetError,
            uint options, out float resultError);

        public struct UnwrapOptions
        {
            // ChartOptions
            public float MaxChartArea;
            public float MaxBoundaryLength;
            public float NormalDeviationWeight;
            public float RoundnessWeight;
            public float StraightnessWeight;
            public float NormalSeamWeight;
            public float TextureSeamWeight;
            public float MaxCost;
            public uint MaxIterations;

            // PackOptions
            public uint MaxChartSize;
            public uint Padding;
            public float TexelsPerUnit;
            public uint Resolution;
            public bool Bilinear;
            public bool BlockAlign;
            public bool BruteForce;
            public bool RotateChartsToAxis;
            public bool RotateCharts;

            public static UnwrapOptions Default => new UnwrapOptions
            {
                MaxChartArea          = 0f,
                MaxBoundaryLength     = 0f,
                NormalDeviationWeight = 2f,
                RoundnessWeight       = 0.01f,
                StraightnessWeight    = 6f,
                NormalSeamWeight      = 4f,
                TextureSeamWeight     = 0.5f,
                MaxCost               = 1.5f,
                MaxIterations         = 1,
                MaxChartSize          = 0,
                Padding               = 2,
                TexelsPerUnit         = 0f,
                Resolution            = 2048,
                Bilinear              = true,
                BlockAlign            = true,
                BruteForce            = false,
                RotateChartsToAxis    = true,
                RotateCharts          = true
            };
        }

        public struct Result
        {
            public int AtlasWidth;
            public int AtlasHeight;
            public float[] UVs;           // [vertCount * 2] — raw UV in atlas-pixel coords
            public int[] Xrefs;           // [vertCount] — maps output vert -> input vert index
            public int[] ChartIndices;    // [vertCount] — chart assignment per output vertex (-1 if none)
            public int[] Indices;         // [indexCount] — triangle indices into output verts
            public int VertexCount;
            public int IndexCount;
            public int ChartCount;
        }

        /// <summary>
        /// Runs xatlas UV unwrap with default hardcoded options.
        /// Positions/normals are flat float arrays (x,y,z per vertex).
        /// </summary>
        public static Result Unwrap(float[] positions, float[] normals, int vertexCount,
            int[] indices, int indexCount, int maxResolution = 2048)
        {
            IntPtr atlas = xatlas_create();
            try
            {
                int err = xatlas_add_mesh(atlas,
                    positions, 12, normals, 12,
                    vertexCount, indices, indexCount);

                if (err != 0)
                {
                    Logger.Error($"[XAtlas] AddMesh failed with error code {err}");
                    return default;
                }

                xatlas_generate(atlas, maxResolution);
                return ReadResult(atlas);
            }
            finally
            {
                xatlas_destroy(atlas);
            }
        }

        /// <summary>
        /// Runs xatlas UV unwrap with full control over chart and pack options.
        /// </summary>
        public static Result Unwrap(float[] positions, float[] normals, int vertexCount,
            int[] indices, int indexCount, UnwrapOptions opts)
        {
            IntPtr atlas = xatlas_create();
            try
            {
                int err = xatlas_add_mesh(atlas,
                    positions, 12, normals, 12,
                    vertexCount, indices, indexCount);

                if (err != 0)
                {
                    Logger.Error($"[XAtlas] AddMesh failed with error code {err}");
                    return default;
                }

                xatlas_generate_opts(atlas,
                    opts.MaxChartArea, opts.MaxBoundaryLength,
                    opts.NormalDeviationWeight, opts.RoundnessWeight,
                    opts.StraightnessWeight, opts.NormalSeamWeight,
                    opts.TextureSeamWeight, opts.MaxCost,
                    opts.MaxIterations,
                    opts.MaxChartSize, opts.Padding,
                    opts.TexelsPerUnit, opts.Resolution,
                    opts.Bilinear ? 1 : 0, opts.BlockAlign ? 1 : 0, opts.BruteForce ? 1 : 0,
                    opts.RotateChartsToAxis ? 1 : 0, opts.RotateCharts ? 1 : 0);

                return ReadResult(atlas);
            }
            finally
            {
                xatlas_destroy(atlas);
            }
        }

        static Result ReadResult(IntPtr atlas)
        {
            xatlas_get_atlas_dims(atlas, out int w, out int h);

            int outVertCount = xatlas_get_vertex_count(atlas, 0);
            int outIdxCount = xatlas_get_index_count(atlas, 0);

            float[] uvs = new float[outVertCount * 2];
            int[] xrefs = new int[outVertCount];
            xatlas_get_vertices(atlas, 0, uvs, xrefs, outVertCount);

            int[] chartIndices = new int[outVertCount];
            xatlas_get_chart_indices(atlas, 0, chartIndices, outVertCount);
            int chartCount = xatlas_get_chart_count(atlas, 0);

            int[] outIndices = new int[outIdxCount];
            xatlas_get_indices(atlas, 0, outIndices, outIdxCount);

            return new Result
            {
                AtlasWidth = w,
                AtlasHeight = h,
                UVs = uvs,
                Xrefs = xrefs,
                ChartIndices = chartIndices,
                Indices = outIndices,
                VertexCount = outVertCount,
                IndexCount = outIdxCount,
                ChartCount = chartCount
            };
        }

        private const uint MeshoptSimplifyLockBorder = 1;

        /// <summary>
        /// Post-bake mesh simplification using meshopt_simplifyWithAttributes (UV-preserving).
        /// UVs are passed as vertex attributes with LockBorder to prevent seam tears.
        /// Returns a new <see cref="RefinedTextureResult"/> with compacted vertex/index arrays
        /// (atlas pixels are NOT copied — caller must set them).
        /// Throws if the native export is missing; caller should catch and skip simplification.
        /// </summary>
        public static RefinedTextureResult SimplifyWithUVs(
            Vector3[] positions, Vector3[] normals, Vector2[] uvs,
            int[] indices, float targetRatio, float targetError = 1e-2f)
        {
            int vertexCount = positions.Length;
            int indexCount = indices.Length;
            int targetIndexCount = Mathf.Max(3, Mathf.RoundToInt(indexCount * Mathf.Clamp01(targetRatio)));
            targetIndexCount = (targetIndexCount / 3) * 3;

            float[] flatPos = new float[vertexCount * 3];
            for (int i = 0; i < vertexCount; i++)
            {
                flatPos[i * 3]     = positions[i].x;
                flatPos[i * 3 + 1] = positions[i].y;
                flatPos[i * 3 + 2] = positions[i].z;
            }

            uint[] uIndices = new uint[indexCount];
            for (int i = 0; i < indexCount; i++)
                uIndices[i] = (uint)indices[i];

            uint[] outIndices = new uint[indexCount];
            int resultCount;

            float[] flatAttrs = new float[vertexCount * 2];
            for (int i = 0; i < vertexCount; i++)
            {
                flatAttrs[i * 2]     = uvs[i].x;
                flatAttrs[i * 2 + 1] = uvs[i].y;
            }
            float[] attrWeights = { 1f, 1f };

            resultCount = meshopt_simplify_with_attrs(
                outIndices,
                uIndices, indexCount,
                flatPos, vertexCount, 12,
                flatAttrs, 8,
                attrWeights, 2,
                targetIndexCount, targetError,
                MeshoptSimplifyLockBorder, out _);

            if (resultCount <= 0)
            {
                Logger.Warning("[MeshOpt] Simplification produced 0 indices, using original mesh");
                return new RefinedTextureResult
                {
                    Positions = positions,
                    Normals = normals,
                    UVs = uvs,
                    Indices = indices
                };
            }

            bool[] used = new bool[vertexCount];
            for (int i = 0; i < resultCount; i++)
                used[outIndices[i]] = true;

            int[] remap = new int[vertexCount];
            int newCount = 0;
            for (int i = 0; i < vertexCount; i++)
                remap[i] = used[i] ? newCount++ : -1;

            var outPos = new Vector3[newCount];
            var outNorm = new Vector3[newCount];
            var outUVs = new Vector2[newCount];
            for (int i = 0; i < vertexCount; i++)
            {
                if (!used[i]) continue;
                int j = remap[i];
                outPos[j]  = positions[i];
                outNorm[j] = normals[i];
                outUVs[j]  = uvs[i];
            }

            int[] outIdx = new int[resultCount];
            for (int i = 0; i < resultCount; i++)
                outIdx[i] = remap[outIndices[i]];

            Logger.Info($"[MeshOpt] Simplified {indexCount / 3} -> {resultCount / 3} tris " +
                        $"({vertexCount} -> {newCount} verts)");

            return new RefinedTextureResult
            {
                Positions = outPos,
                Normals = outNorm,
                UVs = outUVs,
                Indices = outIdx
            };
        }
    }
}
