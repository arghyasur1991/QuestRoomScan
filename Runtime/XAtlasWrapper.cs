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
    public static class XAtlasWrapper
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
        [DllImport(LIB)] private static extern void xatlas_get_atlas_dims(IntPtr atlas, out int width, out int height);
        [DllImport(LIB)] private static extern int xatlas_get_vertex_count(IntPtr atlas, int meshIndex);
        [DllImport(LIB)] private static extern int xatlas_get_index_count(IntPtr atlas, int meshIndex);
        [DllImport(LIB)] private static extern void xatlas_get_vertices(
            IntPtr atlas, int meshIndex,
            float[] uvs, int[] xrefs, int maxVerts);
        [DllImport(LIB)] private static extern void xatlas_get_indices(
            IntPtr atlas, int meshIndex,
            int[] outIndices, int maxIndices);

        public struct Result
        {
            public int AtlasWidth;
            public int AtlasHeight;
            public float[] UVs;      // [vertCount * 2] — raw UV in atlas-pixel coords
            public int[] Xrefs;      // [vertCount] — maps output vert -> input vert index
            public int[] Indices;     // [indexCount] — triangle indices into output verts
            public int VertexCount;
            public int IndexCount;
        }

        /// <summary>
        /// Runs xatlas UV unwrap. Safe to call from a background thread.
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
                    Debug.LogError($"[XAtlas] AddMesh failed with error code {err}");
                    return default;
                }

                xatlas_generate(atlas, maxResolution);
                xatlas_get_atlas_dims(atlas, out int w, out int h);

                int outVertCount = xatlas_get_vertex_count(atlas, 0);
                int outIdxCount = xatlas_get_index_count(atlas, 0);

                float[] uvs = new float[outVertCount * 2];
                int[] xrefs = new int[outVertCount];
                xatlas_get_vertices(atlas, 0, uvs, xrefs, outVertCount);

                int[] outIndices = new int[outIdxCount];
                xatlas_get_indices(atlas, 0, outIndices, outIdxCount);

                return new Result
                {
                    AtlasWidth = w,
                    AtlasHeight = h,
                    UVs = uvs,
                    Xrefs = xrefs,
                    Indices = outIndices,
                    VertexCount = outVertCount,
                    IndexCount = outIdxCount
                };
            }
            finally
            {
                xatlas_destroy(atlas);
            }
        }
    }
}
