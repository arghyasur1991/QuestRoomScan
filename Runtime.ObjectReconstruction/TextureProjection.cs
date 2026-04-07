#if HAS_ONNXRUNTIME
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Projects the preprocessed source image onto visible mesh faces as UV texture.
    /// Uses TripoSR's canonical viewing direction (camera looks along -X in canonical space)
    /// with an orthographic projection scaled to the mesh's actual Y/Z bounds.
    /// Front-facing vertices get projected UVs; back-facing vertices fall back to vertex color.
    /// </summary>
    internal static class TextureProjection
    {
        private const float ForegroundRatio = 0.85f;
        private const float ForegroundPad = (1f - ForegroundRatio) * 0.5f;

        /// <summary>
        /// Computes UV projection data and applies it to the mesh.
        /// Sets UV0 = projected texture coordinates, UV1.x = blend factor (1=texture, 0=vertex color).
        /// The caller must assign the preprocessed source image to the material's _MainTex.
        /// </summary>
        internal static async Task ApplyProjectionAsync(Mesh mesh)
        {
            var vertices = mesh.vertices;
            var normals = mesh.normals;
            int count = vertices.Length;

            if (count == 0 || normals == null || normals.Length != count) return;

            // Compute mesh bounds in canonical Y/Z plane (the view plane)
            float minY = float.MaxValue, maxY = float.MinValue;
            float minZ = float.MaxValue, maxZ = float.MinValue;
            for (int i = 0; i < count; i++)
            {
                var v = vertices[i];
                if (v.y < minY) minY = v.y;
                if (v.y > maxY) maxY = v.y;
                if (v.z < minZ) minZ = v.z;
                if (v.z > maxZ) maxZ = v.z;
            }

            float rangeY = maxY - minY;
            float rangeZ = maxZ - minZ;

            // Use the larger of Y/Z range for uniform scaling (preserve aspect ratio)
            float maxRange = Mathf.Max(rangeY, rangeZ);
            if (maxRange < 1e-6f) return;

            float centerY = (minY + maxY) * 0.5f;
            float centerZ = (minZ + maxZ) * 0.5f;

            var uvs = new Vector2[count];
            var blendUvs = new Vector2[count];

            await Task.Run(() =>
            {
                float invRange = 1f / maxRange;

                for (int i = 0; i < count; i++)
                {
                    var v = vertices[i];
                    var n = normals[i];

                    // Canonical camera is along +X, looking toward -X.
                    // Front-facing vertices have normals pointing toward camera (n.x < 0
                    // in mesh-local space after RecalculateNormals).
                    float facing = -n.x;
                    float blend = Mathf.Clamp01(facing * 3f);

                    // Project onto Y/Z plane, normalized to [-0.5, 0.5] relative to mesh center
                    float py = (v.y - centerY) * invRange;
                    float pz = (v.z - centerZ) * invRange;

                    // Map to image UV space: centered in the foreground region
                    float u = py * ForegroundRatio + 0.5f;
                    float vCoord = pz * ForegroundRatio + 0.5f;

                    // Clamp UVs and zero-out blend for out-of-range
                    if (u < 0f || u > 1f || vCoord < 0f || vCoord > 1f)
                        blend = 0f;

                    uvs[i] = new Vector2(u, vCoord);
                    blendUvs[i] = new Vector2(blend, 0f);
                }
            });

            mesh.SetUVs(0, uvs);
            mesh.SetUVs(1, blendUvs);
        }
    }
}
#endif
