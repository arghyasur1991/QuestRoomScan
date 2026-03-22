using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public struct RefinedTextureResult
    {
        public Vector3[] Positions;
        public Vector3[] Normals;
        public Vector2[] UVs;
        public int[] Indices;
        public byte[] AtlasPixels; // RGBA32
        public int AtlasWidth;
        public int AtlasHeight;
    }

    /// <summary>
    /// On-device texture refinement pipeline: reads back the GPU mesh,
    /// UV-unwraps via xatlas, and bakes a sharp texture atlas from saved keyframes.
    /// All CPU-heavy work runs on background threads.
    /// </summary>
    public static class TextureRefinement
    {
        private const int GpuVertexStride = 32; // float3 pos + float3 norm + uint color + uint voxelIdx

        public static event Action<string> StatusChanged;

        // ═══════════════════════════════════════════════════════════════
        //  MAIN PIPELINE
        // ═══════════════════════════════════════════════════════════════

        public static async Task<RefinedTextureResult> RefineAsync(
            string keyframeDir, int atlasResolution = 2048)
        {
            // Step 1: Readback mesh from GPU
            ReportStatus("Reading mesh from GPU...");
            var (positions, normals, colors, indices) = await ReadbackMeshAsync();
            if (positions == null || positions.Length == 0)
                throw new InvalidOperationException("Mesh readback returned no vertices");

            Debug.Log($"[TextureRefine] Readback: {positions.Length} verts, {indices.Length / 3} tris");

            // Step 2: xatlas UV unwrap on background thread
            ReportStatus("UV unwrapping...");
            XAtlasWrapper.Result uvResult = default;
            Vector3[] inPos = positions;
            Vector3[] inNorm = normals;
            int[] inIdx = indices;

            await Task.Run(() =>
            {
                float[] flatPos = new float[inPos.Length * 3];
                float[] flatNorm = new float[inNorm.Length * 3];
                for (int i = 0; i < inPos.Length; i++)
                {
                    flatPos[i * 3] = inPos[i].x;
                    flatPos[i * 3 + 1] = inPos[i].y;
                    flatPos[i * 3 + 2] = inPos[i].z;
                    flatNorm[i * 3] = inNorm[i].x;
                    flatNorm[i * 3 + 1] = inNorm[i].y;
                    flatNorm[i * 3 + 2] = inNorm[i].z;
                }
                uvResult = XAtlasWrapper.Unwrap(flatPos, flatNorm, inPos.Length,
                    inIdx, inIdx.Length, atlasResolution);
            });

            if (uvResult.VertexCount == 0)
                throw new InvalidOperationException("xatlas produced no output vertices");

            Debug.Log($"[TextureRefine] xatlas: {uvResult.VertexCount} verts, " +
                      $"{uvResult.IndexCount / 3} tris, atlas {uvResult.AtlasWidth}x{uvResult.AtlasHeight}");

            // Remap: xatlas produces new vertices (splits at seams). Build final arrays.
            int atlasW = uvResult.AtlasWidth;
            int atlasH = uvResult.AtlasHeight;
            Vector3[] outPos = new Vector3[uvResult.VertexCount];
            Vector3[] outNorm = new Vector3[uvResult.VertexCount];
            Vector2[] outUVs = new Vector2[uvResult.VertexCount];

            for (int i = 0; i < uvResult.VertexCount; i++)
            {
                int src = uvResult.Xrefs[i];
                outPos[i] = inPos[src];
                outNorm[i] = inNorm[src];
                outUVs[i] = new Vector2(
                    uvResult.UVs[i * 2] / atlasW,
                    uvResult.UVs[i * 2 + 1] / atlasH);
            }

            // Step 3: Texture bake from keyframes
            ReportStatus("Baking textures...");
            byte[] atlasPixels = null;
            float[] rawUVs = uvResult.UVs;
            int[] outIndices = uvResult.Indices;

            await Task.Run(() =>
            {
                atlasPixels = BakeAtlas(inPos, inNorm, outPos, outNorm,
                    rawUVs, outIndices, uvResult.VertexCount,
                    atlasW, atlasH, keyframeDir);
            });

            // Step 4: Denoise
            ReportStatus("Denoising...");
            await Task.Run(() => DenoiseAtlas(atlasPixels, atlasW, atlasH));

            // Step 5: Dilation
            ReportStatus("Filling gaps...");
            await Task.Run(() => DilateAtlas(atlasPixels, atlasW, atlasH, 8));

            ReportStatus("Done");
            Debug.Log($"[TextureRefine] Complete: {atlasW}x{atlasH} atlas, " +
                      $"{outPos.Length} verts, {outIndices.Length / 3} tris");

            return new RefinedTextureResult
            {
                Positions = outPos,
                Normals = outNorm,
                UVs = outUVs,
                Indices = outIndices,
                AtlasPixels = atlasPixels,
                AtlasWidth = atlasW,
                AtlasHeight = atlasH
            };
        }

        // ═══════════════════════════════════════════════════════════════
        //  MESH READBACK
        // ═══════════════════════════════════════════════════════════════

        static Task<byte[]> ReadbackBytesAsync(GraphicsBuffer buffer)
        {
            var tcs = new System.Threading.Tasks.TaskCompletionSource<byte[]>();
            AsyncGPUReadback.Request(buffer, request =>
            {
                if (request.hasError) { tcs.SetResult(null); return; }
                var native = request.GetData<byte>();
                byte[] managed = new byte[native.Length];
                NativeArray<byte>.Copy(native, managed, native.Length);
                tcs.SetResult(managed);
            });
            return tcs.Task;
        }

        static async Task<(Vector3[], Vector3[], Color32[], int[])> ReadbackMeshAsync()
        {
            var gpuSN = MeshExtractor.Instance?.GpuSurfaceNets;
            if (gpuSN == null || gpuSN.VertexBuffer == null || gpuSN.IndexBuffer == null)
            {
                Debug.LogError("[TextureRefine] GpuSurfaceNets or its buffers are null");
                return (null, null, null, null);
            }

            Debug.Log("[TextureRefine] Starting GPU readback...");

            // Read all three buffers using callback-based readback
            // (copies NativeArray to managed array immediately in callback frame)
            byte[] counterBytes = await ReadbackBytesAsync(gpuSN.CountersBuffer);
            if (counterBytes == null)
            {
                Debug.LogError("[TextureRefine] Counter readback failed");
                return (null, null, null, null);
            }

            int vertCount = BitConverter.ToInt32(counterBytes, 0);
            int idxCount = counterBytes.Length >= 8 ? BitConverter.ToInt32(counterBytes, 4) : 0;

            Debug.Log($"[TextureRefine] Counters: verts={vertCount}, idx={idxCount}");

            if (vertCount <= 0 || idxCount <= 0)
            {
                Debug.LogWarning($"[TextureRefine] No mesh data: verts={vertCount}, idx={idxCount}");
                return (null, null, null, null);
            }

            byte[] vertData = await ReadbackBytesAsync(gpuSN.VertexBuffer);
            if (vertData == null)
            {
                Debug.LogError("[TextureRefine] Vertex readback failed");
                return (null, null, null, null);
            }

            byte[] idxData = await ReadbackBytesAsync(gpuSN.IndexBuffer);
            if (idxData == null)
            {
                Debug.LogError("[TextureRefine] Index readback failed");
                return (null, null, null, null);
            }

            int bufferCap = vertData.Length / GpuVertexStride;
            if (vertCount > bufferCap) vertCount = bufferCap;

            int idxCap = idxData.Length / 4;
            if (idxCount > idxCap) idxCount = idxCap;

            // Parse indices
            int[] indices = new int[idxCount];
            Buffer.BlockCopy(idxData, 0, indices, 0, idxCount * 4);

            // Parse GPU vertices
            var positions = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var colors = new Color32[vertCount];

            for (int i = 0; i < vertCount; i++)
            {
                int off = i * GpuVertexStride;
                positions[i] = new Vector3(
                    BitConverter.ToSingle(vertData, off),
                    BitConverter.ToSingle(vertData, off + 4),
                    BitConverter.ToSingle(vertData, off + 8));
                normals[i] = new Vector3(
                    BitConverter.ToSingle(vertData, off + 12),
                    BitConverter.ToSingle(vertData, off + 16),
                    BitConverter.ToSingle(vertData, off + 20));
                uint packed = BitConverter.ToUInt32(vertData, off + 24);
                colors[i] = new Color32(
                    (byte)(packed & 0xFF),
                    (byte)((packed >> 8) & 0xFF),
                    (byte)((packed >> 16) & 0xFF),
                    255);
            }

            Debug.Log($"[TextureRefine] Readback complete: {vertCount} verts, {idxCount / 3} tris");
            return (positions, normals, colors, indices);
        }

        // ═══════════════════════════════════════════════════════════════
        //  TEXTURE BAKE
        // ═══════════════════════════════════════════════════════════════

        struct Keyframe
        {
            public byte[] Pixels; // RGBA32, row-major (or compressed JPEG before decode)
            public int Width, Height;
            public int SensorWidth, SensorHeight;
            public Vector3 Position;
            public Quaternion Rotation;
            public float Fx, Fy, Cx, Cy;
            public string JpgPath; // deferred: path to JPEG, read on demand to avoid OOM
        }

        struct TriData
        {
            public int I0, I1, I2;
            public Vector3 FaceNormal;
            public Vector3 Centroid;
            public float U0, V0, U1, V1, U2, V2;
        }

        static TriData[] PrecomputeTriData(
            Vector3[] outPos, Vector3[] outNorm, float[] rawUVs, int[] indices, int outVertCount)
        {
            int triCount = indices.Length / 3;
            var data = new TriData[triCount];
            for (int t = 0; t < triCount; t++)
            {
                int i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
                if (i0 >= outVertCount || i1 >= outVertCount || i2 >= outVertCount)
                {
                    data[t].I0 = -1;
                    continue;
                }

                Vector3 p0 = outPos[i0], p1 = outPos[i1], p2 = outPos[i2];
                Vector3 fn = Vector3.Cross(p1 - p0, p2 - p0).normalized;
                if (fn.sqrMagnitude < 0.001f) fn = outNorm[i0];

                data[t].I0 = i0; data[t].I1 = i1; data[t].I2 = i2;
                data[t].FaceNormal = fn;
                data[t].Centroid = (p0 + p1 + p2) / 3f;
                data[t].U0 = rawUVs[i0 * 2]; data[t].V0 = rawUVs[i0 * 2 + 1];
                data[t].U1 = rawUVs[i1 * 2]; data[t].V1 = rawUVs[i1 * 2 + 1];
                data[t].U2 = rawUVs[i2 * 2]; data[t].V2 = rawUVs[i2 * 2 + 1];
            }
            return data;
        }

        static byte[] BakeAtlas(
            Vector3[] inPos, Vector3[] inNorm,
            Vector3[] outPos, Vector3[] outNorm,
            float[] rawUVs, int[] indices, int outVertCount,
            int atlasW, int atlasH, string keyframeDir)
        {
            int texelCount = atlasW * atlasH;
            byte[] atlas = new byte[texelCount * 4]; // RGBA32
            float[] bestScore = new float[texelCount];

            string imagesDir = Path.Combine(keyframeDir, "images");
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");

            if (!File.Exists(manifestPath))
            {
                Debug.LogWarning("[TextureRefine] No frames.jsonl found");
                return atlas;
            }

            string[] lines = File.ReadAllLines(manifestPath);
            int processed = 0;

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                Keyframe kf;
                try
                {
                    kf = ParseKeyframe(line, imagesDir);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[TextureRefine] Skip keyframe: {e.Message}");
                    continue;
                }

                if (string.IsNullOrEmpty(kf.JpgPath)) continue;

                // Build per-keyframe depth buffer for occlusion
                float[] depthBuf = BuildDepthBuffer(inPos, indices, kf, kf.Width, kf.Height);

                // Build view matrix
                Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;
                Vector3 camPos = kf.Position;

                int triCount = indices.Length / 3;
                for (int t = 0; t < triCount; t++)
                {
                    int i0 = indices[t * 3];
                    int i1 = indices[t * 3 + 1];
                    int i2 = indices[t * 3 + 2];

                    if (i0 >= outVertCount || i1 >= outVertCount || i2 >= outVertCount) continue;

                    Vector3 p0 = outPos[i0], p1 = outPos[i1], p2 = outPos[i2];
                    Vector3 n0 = outNorm[i0];
                    Vector3 faceNormal = Vector3.Cross(p1 - p0, p2 - p0).normalized;
                    if (faceNormal.sqrMagnitude < 0.001f) faceNormal = n0;

                    Vector3 viewDir = (camPos - (p0 + p1 + p2) / 3f).normalized;
                    float dot = Vector3.Dot(faceNormal, viewDir);
                    if (dot <= 0.05f) continue; // backface

                    // Project vertices to keyframe image
                    Vector2 uv0Screen = ProjectToScreen(p0, viewMat, kf);
                    Vector2 uv1Screen = ProjectToScreen(p1, viewMat, kf);
                    Vector2 uv2Screen = ProjectToScreen(p2, viewMat, kf);

                    bool inFrustum =
                        IsInFrustum(uv0Screen, kf.Width, kf.Height) ||
                        IsInFrustum(uv1Screen, kf.Width, kf.Height) ||
                        IsInFrustum(uv2Screen, kf.Width, kf.Height);
                    if (!inFrustum) continue;

                    float dist = Vector3.Distance(camPos, (p0 + p1 + p2) / 3f);
                    float score = dot / Mathf.Max(dist, 0.1f);

                    // UV coords in atlas pixel space
                    float u0 = rawUVs[i0 * 2], v0 = rawUVs[i0 * 2 + 1];
                    float u1 = rawUVs[i1 * 2], v1 = rawUVs[i1 * 2 + 1];
                    float u2 = rawUVs[i2 * 2], v2 = rawUVs[i2 * 2 + 1];

                    RasterizeTriangle(atlas, bestScore, atlasW, atlasH,
                        u0, v0, u1, v1, u2, v2,
                        uv0Screen, uv1Screen, uv2Screen,
                        score, kf, depthBuf, p0, p1, p2, viewMat);
                }

                processed++;
                if (processed % 20 == 0 || processed <= 3)
                    ReportStatus($"Baking... {processed}/{lines.Length}");
            }

            Debug.Log($"[TextureRefine] Baked {processed} keyframes");
            return atlas;
        }

        static Keyframe ParseKeyframe(string jsonLine, string imagesDir)
        {
            var kf = new Keyframe();
            // Minimal JSON parsing without dependency
            float px = 0, py = 0, pz = 0, qx = 0, qy = 0, qz = 0, qw = 1;
            int id = 0;
            float fx = 0, fy = 0, cx = 0, cy = 0;
            int sw = 0, sh = 0;

            foreach (string token in jsonLine.Trim('{', '}', ' ').Split(','))
            {
                string[] kv = token.Split(':');
                if (kv.Length < 2) continue;
                string key = kv[0].Trim('"', ' ');
                string val = kv[1].Trim('"', ' ');
                switch (key)
                {
                    case "id": id = int.Parse(val); break;
                    case "px": px = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "py": py = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "pz": pz = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qx": qx = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qy": qy = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qz": qz = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "qw": qw = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "fx": fx = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "fy": fy = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "cx": cx = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "cy": cy = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture); break;
                    case "sw": sw = int.Parse(val); break;
                    case "sh": sh = int.Parse(val); break;
                }
            }

            kf.Position = new Vector3(px, py, pz);
            kf.Rotation = new Quaternion(qx, qy, qz, qw);
            kf.Fx = fx; kf.Fy = fy;
            kf.Cx = cx; kf.Cy = cy;
            kf.SensorWidth = sw;
            kf.SensorHeight = sh;

            string imgPath = Path.Combine(imagesDir, $"{id:D6}.jpg");
            if (!File.Exists(imgPath)) return kf;

            // Store path only — JPEG bytes read on-demand during bake to avoid OOM with many keyframes
            kf.JpgPath = imgPath;
            kf.Pixels = new byte[0]; // non-null signals "has image"
            return kf;
        }

        /// <summary>
        /// Pre-decodes JPEG keyframes on the main thread into RGBA pixel arrays.
        /// Must be called before background bake.
        /// </summary>
        static (byte[][] pixels, int[] widths, int[] heights, Keyframe[] keyframes)
            PreDecodeKeyframes(string keyframeDir)
        {
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");
            if (!File.Exists(manifestPath))
                return (null, null, null, null);

            string[] lines = File.ReadAllLines(manifestPath);
            string imagesDir = Path.Combine(keyframeDir, "images");
            var pixelsList = new System.Collections.Generic.List<byte[]>();
            var widthsList = new System.Collections.Generic.List<int>();
            var heightsList = new System.Collections.Generic.List<int>();
            var kfList = new System.Collections.Generic.List<Keyframe>();

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var kf = ParseKeyframe(line, imagesDir);
                    if (string.IsNullOrEmpty(kf.JpgPath)) continue;

                    byte[] jpgBytes = File.ReadAllBytes(kf.JpgPath);
                    var tex = new Texture2D(2, 2);
                    if (!ImageConversion.LoadImage(tex, jpgBytes))
                    {
                        UnityEngine.Object.Destroy(tex);
                        continue;
                    }

                    int w = tex.width, h = tex.height;
                    Color32[] c32 = tex.GetPixels32();
                    UnityEngine.Object.Destroy(tex);
                    byte[] rgba = new byte[c32.Length * 4];
                    for (int ci = 0; ci < c32.Length; ci++)
                    {
                        rgba[ci * 4] = c32[ci].r;
                        rgba[ci * 4 + 1] = c32[ci].g;
                        rgba[ci * 4 + 2] = c32[ci].b;
                        rgba[ci * 4 + 3] = 255;
                    }

                    kf.Pixels = rgba;
                    kf.Width = w;
                    kf.Height = h;

                    pixelsList.Add(rgba);
                    widthsList.Add(w);
                    heightsList.Add(h);
                    kfList.Add(kf);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[TextureRefine] Skip keyframe decode: {e.Message}");
                }
            }

            return (pixelsList.ToArray(), widthsList.ToArray(), heightsList.ToArray(), kfList.ToArray());
        }

        /// <summary>
        /// Full pipeline with pre-decoded keyframes (main thread decodes, BG thread bakes).
        /// </summary>
        /// <param name="keyframeDir">Path to GSExport directory containing frames.jsonl + images/</param>
        /// <param name="keyframeRelocation">Matrix to transform keyframe poses from old session to current world space.
        /// Pass Matrix4x4.identity for live (non-reloaded) scans.</param>
        /// <param name="atlasResolution">Max atlas dimension for xatlas</param>
        public static async Task<RefinedTextureResult> RefineWithPreDecodedAsync(
            string keyframeDir, Matrix4x4 keyframeRelocation, int atlasResolution = 2048)
        {
            // Step 1: Readback mesh from GPU (must be on main thread for AsyncGPUReadback)
            ReportStatus("Reading mesh from GPU...");
            var (positions, normals, colors, indices) = await ReadbackMeshAsync();
            if (positions == null || positions.Length == 0)
                throw new InvalidOperationException("Mesh readback returned no vertices");

            Debug.Log($"[TextureRefine] Readback: {positions.Length} verts, {indices.Length / 3} tris");

            // Step 2: Parse keyframe metadata (poses + intrinsics) without decoding images
            ReportStatus("Loading keyframe metadata...");
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");
            string imagesDir = Path.Combine(keyframeDir, "images");
            if (!File.Exists(manifestPath))
                throw new InvalidOperationException("No frames.jsonl found");

            string[] lines = File.ReadAllLines(manifestPath);
            var metaList = new System.Collections.Generic.List<Keyframe>();
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var kf = ParseKeyframe(line, imagesDir);
                    if (string.IsNullOrEmpty(kf.JpgPath)) continue; // no image file
                    // Apply relocation to keyframe poses if scan was reloaded
                    if (keyframeRelocation != Matrix4x4.identity)
                    {
                        kf.Position = keyframeRelocation.MultiplyPoint3x4(kf.Position);
                        kf.Rotation = keyframeRelocation.rotation * kf.Rotation;
                    }
                    metaList.Add(kf);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[TextureRefine] Skip keyframe: {e.Message}");
                }
            }
            if (metaList.Count == 0)
                throw new InvalidOperationException("No keyframes available for baking");

            Debug.Log($"[TextureRefine] Found {metaList.Count} keyframes" +
                (keyframeRelocation != Matrix4x4.identity ? " (relocated)" : ""));

            // Step 3: xatlas UV unwrap on background thread
            ReportStatus("UV unwrapping...");
            XAtlasWrapper.Result uvResult = default;
            Vector3[] inPos = positions;
            Vector3[] inNorm = normals;
            int[] inIdx = indices;

            await Task.Run(() =>
            {
                float[] flatPos = new float[inPos.Length * 3];
                float[] flatNorm = new float[inNorm.Length * 3];
                for (int i = 0; i < inPos.Length; i++)
                {
                    flatPos[i * 3] = inPos[i].x;
                    flatPos[i * 3 + 1] = inPos[i].y;
                    flatPos[i * 3 + 2] = inPos[i].z;
                    flatNorm[i * 3] = inNorm[i].x;
                    flatNorm[i * 3 + 1] = inNorm[i].y;
                    flatNorm[i * 3 + 2] = inNorm[i].z;
                }
                uvResult = XAtlasWrapper.Unwrap(flatPos, flatNorm, inPos.Length,
                    inIdx, inIdx.Length, atlasResolution);
            });

            if (uvResult.VertexCount == 0)
                throw new InvalidOperationException("xatlas produced no output vertices");

            int atlasW = uvResult.AtlasWidth;
            int atlasH = uvResult.AtlasHeight;
            Debug.Log($"[TextureRefine] xatlas: {uvResult.VertexCount} verts, " +
                      $"{uvResult.IndexCount / 3} tris, atlas {atlasW}x{atlasH}");

            // Remap vertices
            Vector3[] outPos = new Vector3[uvResult.VertexCount];
            Vector3[] outNorm = new Vector3[uvResult.VertexCount];
            Vector2[] outUVs = new Vector2[uvResult.VertexCount];
            for (int i = 0; i < uvResult.VertexCount; i++)
            {
                int src = uvResult.Xrefs[i];
                outPos[i] = inPos[src];
                outNorm[i] = inNorm[src];
                outUVs[i] = new Vector2(
                    uvResult.UVs[i * 2] / atlasW,
                    uvResult.UVs[i * 2 + 1] / atlasH);
            }

            // Step 4: Bake atlas — decode one keyframe at a time to avoid OOM.
            // JPEG decode requires main thread (ImageConversion.LoadImage), so we
            // interleave: decode on main thread, bake that frame on BG thread.
            ReportStatus("Baking textures...");
            float[] rawUVs = uvResult.UVs;
            int[] outIndices = uvResult.Indices;
            int outVertCount = uvResult.VertexCount;
            int texelCount = atlasW * atlasH;
            byte[] atlasPixels = new byte[texelCount * 4];
            float[] bestScore = new float[texelCount];

            // Pre-compute per-triangle data once (invariant across keyframes)
            TriData[] triData = null;
            await Task.Run(() => triData = PrecomputeTriData(outPos, outNorm, rawUVs, outIndices, outVertCount));

            for (int ki = 0; ki < metaList.Count; ki++)
            {
                var kf = metaList[ki];
                if (string.IsNullOrEmpty(kf.JpgPath)) continue;

                // Read JPEG from disk on demand, decode on main thread (one at a time)
                byte[] jpgBytes;
                try { jpgBytes = File.ReadAllBytes(kf.JpgPath); }
                catch { continue; }

                var tex = new Texture2D(2, 2);
                if (!ImageConversion.LoadImage(tex, jpgBytes))
                {
                    UnityEngine.Object.Destroy(tex);
                    continue;
                }
                kf.Width = tex.width;
                kf.Height = tex.height;
                // LoadImage may decode JPEG as RGB24 (no alpha). Force RGBA32 layout
                // using GetPixels32 which handles any source format.
                Color32[] pxColors = tex.GetPixels32();
                UnityEngine.Object.Destroy(tex);
                jpgBytes = null;
                byte[] rgba = new byte[pxColors.Length * 4];
                for (int ci = 0; ci < pxColors.Length; ci++)
                {
                    rgba[ci * 4] = pxColors[ci].r;
                    rgba[ci * 4 + 1] = pxColors[ci].g;
                    rgba[ci * 4 + 2] = pxColors[ci].b;
                    rgba[ci * 4 + 3] = 255;
                }
                kf.Pixels = rgba;

                // Debug: log first keyframe details
                if (ki == 0)
                {
                    Debug.Log($"[TextureRefine] KF0: pos={kf.Position}, rot={kf.Rotation}, " +
                        $"fx={kf.Fx}, fy={kf.Fy}, cx={kf.Cx}, cy={kf.Cy}, " +
                        $"imgSize={kf.Width}x{kf.Height}, pixLen={kf.Pixels.Length}");
                    Debug.Log($"[TextureRefine] V0: pos={outPos[0]}, norm={outNorm[0]}, " +
                        $"uv=({rawUVs[0]},{rawUVs[1]})");
                    // Check first pixel values
                    if (kf.Pixels.Length >= 4)
                        Debug.Log($"[TextureRefine] KF0 pixel[0]: R={kf.Pixels[0]} G={kf.Pixels[1]} " +
                            $"B={kf.Pixels[2]} A={kf.Pixels[3]}");
                    // Project first vertex
                    var vm = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;
                    var s = ProjectToScreen(outPos[0], vm, kf);
                    Debug.Log($"[TextureRefine] V0 screen={s}");
                    // Check center pixel
                    int midIdx = (kf.Height / 2 * kf.Width + kf.Width / 2) * 4;
                    if (midIdx + 3 < kf.Pixels.Length)
                        Debug.Log($"[TextureRefine] KF0 center pixel: R={kf.Pixels[midIdx]} " +
                            $"G={kf.Pixels[midIdx + 1]} B={kf.Pixels[midIdx + 2]} A={kf.Pixels[midIdx + 3]}");
                }

                // Bake this single keyframe on BG thread
                Keyframe capturedKf = kf;
                await Task.Run(() =>
                {
                    BakeSingleKeyframe(atlasPixels, bestScore, atlasW, atlasH,
                        inPos, inNorm, inIdx, outPos, outNorm,
                        rawUVs, outIndices, outVertCount, capturedKf, triData);
                });

                if (ki % 20 == 0 || ki < 3 || ki == metaList.Count - 1)
                {
                    ReportStatus($"Baking... {ki + 1}/{metaList.Count}");
                    Debug.Log($"[TextureRefine] Baked keyframe {ki + 1}/{metaList.Count}");
                }
            }

            // Debug: count filled texels and sample stats before dilation
            {
                int filled = 0, totalR = 0, totalG = 0, totalB = 0;
                for (int i = 0; i < texelCount; i++)
                {
                    if (atlasPixels[i * 4 + 3] != 0)
                    {
                        filled++;
                        totalR += atlasPixels[i * 4];
                        totalG += atlasPixels[i * 4 + 1];
                        totalB += atlasPixels[i * 4 + 2];
                    }
                }
                float avgR = filled > 0 ? totalR / (float)filled : 0;
                float avgG = filled > 0 ? totalG / (float)filled : 0;
                float avgB = filled > 0 ? totalB / (float)filled : 0;
                Debug.Log($"[TextureRefine] Pre-dilation: {filled}/{texelCount} texels filled " +
                    $"({100f * filled / texelCount:F1}%), avgRGB=({avgR:F0},{avgG:F0},{avgB:F0})");
            }

            // Save debug atlas PNG
            try
            {
                var debugTex = new Texture2D(atlasW, atlasH, TextureFormat.RGBA32, false);
                debugTex.SetPixelData(atlasPixels, 0);
                debugTex.Apply();
                byte[] png = ImageConversion.EncodeToPNG(debugTex);
                UnityEngine.Object.Destroy(debugTex);
                string debugPath = Path.Combine(Application.persistentDataPath, "debug_atlas.png");
                File.WriteAllBytes(debugPath, png);
                Debug.Log($"[TextureRefine] Debug atlas saved: {debugPath} ({png.Length / 1024}KB)");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[TextureRefine] Failed to save debug atlas: {e.Message}");
            }

            // Step 5: Denoise — remove speckle outliers from baked atlas
            ReportStatus("Denoising...");
            await Task.Run(() => DenoiseAtlas(atlasPixels, atlasW, atlasH));

            // Step 6: Dilation
            ReportStatus("Filling gaps...");
            await Task.Run(() => DilateAtlas(atlasPixels, atlasW, atlasH, 8));

            ReportStatus("Done");
            Debug.Log($"[TextureRefine] Complete: {atlasW}x{atlasH} atlas");

            return new RefinedTextureResult
            {
                Positions = outPos,
                Normals = outNorm,
                UVs = outUVs,
                Indices = outIndices,
                AtlasPixels = atlasPixels,
                AtlasWidth = atlasW,
                AtlasHeight = atlasH
            };
        }

        static void BakeSingleKeyframe(
            byte[] atlas, float[] bestScore, int atlasW, int atlasH,
            Vector3[] inPos, Vector3[] inNorm, int[] origIndices,
            Vector3[] outPos, Vector3[] outNorm,
            float[] rawUVs, int[] indices, int outVertCount,
            Keyframe kf, TriData[] triData = null)
        {
            if (kf.Pixels == null || kf.Width == 0) return;

            float[] depthBuf = BuildDepthBuffer(inPos, origIndices, kf, kf.Width, kf.Height);

            Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;
            Vector3 camPos = kf.Position;

            int triCount = triData != null ? triData.Length : indices.Length / 3;

            Parallel.For(0, triCount, t =>
            {
                int i0, i1, i2;
                Vector3 faceNormal, centroid;
                float u0, v0, u1, v1, u2, v2;

                if (triData != null)
                {
                    ref readonly TriData td = ref triData[t];
                    if (td.I0 < 0) return;
                    i0 = td.I0; i1 = td.I1; i2 = td.I2;
                    faceNormal = td.FaceNormal;
                    centroid = td.Centroid;
                    u0 = td.U0; v0 = td.V0;
                    u1 = td.U1; v1 = td.V1;
                    u2 = td.U2; v2 = td.V2;
                }
                else
                {
                    i0 = indices[t * 3]; i1 = indices[t * 3 + 1]; i2 = indices[t * 3 + 2];
                    if (i0 >= outVertCount || i1 >= outVertCount || i2 >= outVertCount) return;
                    Vector3 p = outPos[i0], q = outPos[i1], r = outPos[i2];
                    faceNormal = Vector3.Cross(q - p, r - p).normalized;
                    if (faceNormal.sqrMagnitude < 0.001f) faceNormal = outNorm[i0];
                    centroid = (p + q + r) / 3f;
                    u0 = rawUVs[i0 * 2]; v0 = rawUVs[i0 * 2 + 1];
                    u1 = rawUVs[i1 * 2]; v1 = rawUVs[i1 * 2 + 1];
                    u2 = rawUVs[i2 * 2]; v2 = rawUVs[i2 * 2 + 1];
                }

                Vector3 viewDir = (camPos - centroid).normalized;
                float dot = Vector3.Dot(faceNormal, viewDir);
                if (dot <= 0.05f) return;

                Vector3 p0 = outPos[i0], p1 = outPos[i1], p2 = outPos[i2];
                Vector2 s0 = ProjectToScreen(p0, viewMat, kf);
                Vector2 s1 = ProjectToScreen(p1, viewMat, kf);
                Vector2 s2 = ProjectToScreen(p2, viewMat, kf);

                if (!IsInFrustum(s0, kf.Width, kf.Height) &&
                    !IsInFrustum(s1, kf.Width, kf.Height) &&
                    !IsInFrustum(s2, kf.Width, kf.Height))
                    return;

                float dist = Vector3.Distance(camPos, centroid);
                float score = dot / Mathf.Max(dist, 0.1f);

                RasterizeTriangle(atlas, bestScore, atlasW, atlasH,
                    u0, v0, u1, v1, u2, v2,
                    s0, s1, s2,
                    score, kf, depthBuf, p0, p1, p2, viewMat);
            });
        }

        // ═══════════════════════════════════════════════════════════════
        //  PROJECTION HELPERS
        // ═══════════════════════════════════════════════════════════════

        static Vector2 ProjectToScreen(Vector3 worldPos, Matrix4x4 viewMat, Keyframe kf)
        {
            Vector3 cam = viewMat.MultiplyPoint3x4(worldPos);
            if (cam.z <= 0.001f) return new Vector2(-1, -1);
            float sensorX = kf.Fx * (cam.x / cam.z) + kf.Cx;
            float sensorY = kf.Fy * (cam.y / cam.z) + kf.Cy;
            int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
            int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
            float cropX = (sw - kf.Width) * 0.5f;
            float cropY = (sh - kf.Height) * 0.5f;
            return new Vector2(sensorX - cropX, sensorY - cropY);
        }

        static bool IsInFrustum(Vector2 screen, int w, int h)
        {
            return screen.x >= -w * 0.1f && screen.x < w * 1.1f &&
                   screen.y >= -h * 0.1f && screen.y < h * 1.1f;
        }

        static float[] BuildDepthBuffer(Vector3[] positions, int[] indices,
            Keyframe kf, int w, int h)
        {
            float[] depth = new float[w * h];
            for (int i = 0; i < depth.Length; i++) depth[i] = float.MaxValue;

            Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;
            int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
            int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
            float cropX = (sw - kf.Width) * 0.5f;
            float cropY = (sh - kf.Height) * 0.5f;
            int triCount = indices.Length / 3;

            for (int t = 0; t < triCount; t++)
            {
                int i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
                if (i0 >= positions.Length || i1 >= positions.Length || i2 >= positions.Length) continue;

                Vector3 c0 = viewMat.MultiplyPoint3x4(positions[i0]);
                Vector3 c1 = viewMat.MultiplyPoint3x4(positions[i1]);
                Vector3 c2 = viewMat.MultiplyPoint3x4(positions[i2]);

                if (c0.z <= 0 && c1.z <= 0 && c2.z <= 0) continue;

                Vector2 s0 = new Vector2(kf.Fx * c0.x / Mathf.Max(c0.z, 0.001f) + kf.Cx - cropX,
                                         kf.Fy * c0.y / Mathf.Max(c0.z, 0.001f) + kf.Cy - cropY);
                Vector2 s1 = new Vector2(kf.Fx * c1.x / Mathf.Max(c1.z, 0.001f) + kf.Cx - cropX,
                                         kf.Fy * c1.y / Mathf.Max(c1.z, 0.001f) + kf.Cy - cropY);
                Vector2 s2 = new Vector2(kf.Fx * c2.x / Mathf.Max(c2.z, 0.001f) + kf.Cx - cropX,
                                         kf.Fy * c2.y / Mathf.Max(c2.z, 0.001f) + kf.Cy - cropY);

                RasterizeDepthTriangle(depth, w, h, s0, s1, s2, c0.z, c1.z, c2.z);
            }

            return depth;
        }

        static unsafe void RasterizeDepthTriangle(float[] depth, int w, int h,
            Vector2 s0, Vector2 s1, Vector2 s2,
            float z0, float z1, float z2)
        {
            int minX = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(s0.x, Mathf.Min(s1.x, s2.x))));
            int maxX = Mathf.Min(w - 1, Mathf.CeilToInt(Mathf.Max(s0.x, Mathf.Max(s1.x, s2.x))));
            int minY = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(s0.y, Mathf.Min(s1.y, s2.y))));
            int maxY = Mathf.Min(h - 1, Mathf.CeilToInt(Mathf.Max(s0.y, Mathf.Max(s1.y, s2.y))));

            float denom = (s1.y - s2.y) * (s0.x - s2.x) + (s2.x - s1.x) * (s0.y - s2.y);
            if (Mathf.Abs(denom) < 1e-8f) return;
            float invDenom = 1f / denom;

            float a0x = s1.y - s2.y, a0y = s2.x - s1.x;
            float a1x = s2.y - s0.y, a1y = s0.x - s2.x;
            float ox = -s2.x, oy = -s2.y;

            fixed (float* pDepth = depth)
            {
                for (int y = minY; y <= maxY; y++)
                {
                    float dy = y + oy;
                    float* row = pDepth + y * w;
                    for (int x = minX; x <= maxX; x++)
                    {
                        float dx = x + ox;
                        float bw0 = (a0x * dx + a0y * dy) * invDenom;
                        float bw1 = (a1x * dx + a1y * dy) * invDenom;
                        float bw2 = 1f - bw0 - bw1;

                        if (bw0 < -0.001f || bw1 < -0.001f || bw2 < -0.001f) continue;

                        float z = bw0 * z0 + bw1 * z1 + bw2 * z2;
                        if (z < row[x]) row[x] = z;
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        //  UV-SPACE TRIANGLE RASTERIZATION
        // ═══════════════════════════════════════════════════════════════

        static unsafe void RasterizeTriangle(
            byte[] atlas, float[] bestScore, int atlasW, int atlasH,
            float u0, float v0, float u1, float v1, float u2, float v2,
            Vector2 s0, Vector2 s1, Vector2 s2,
            float score, Keyframe kf, float[] depthBuf,
            Vector3 p0, Vector3 p1, Vector3 p2, Matrix4x4 viewMat)
        {
            int minX = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(u0, Mathf.Min(u1, u2))));
            int maxX = Mathf.Min(atlasW - 1, Mathf.CeilToInt(Mathf.Max(u0, Mathf.Max(u1, u2))));
            int minY = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(v0, Mathf.Min(v1, v2))));
            int maxY = Mathf.Min(atlasH - 1, Mathf.CeilToInt(Mathf.Max(v0, Mathf.Max(v1, v2))));

            float denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
            if (Mathf.Abs(denom) < 1e-8f) return;
            float invDenom = 1f / denom;

            float a0x = v1 - v2, a0y = u2 - u1;
            float a1x = v2 - v0, a1y = u0 - u2;
            float ox = -u2, oy = -v2;

            int kfW = kf.Width, kfH = kf.Height;
            int pixelLen = kf.Pixels.Length;

            fixed (byte* pAtlas = atlas, pPixels = kf.Pixels)
            fixed (float* pScore = bestScore, pDepth = depthBuf)
            {
                for (int y = minY; y <= maxY; y++)
                {
                    float dy = y + oy;
                    int atlasRowOff = y * atlasW;
                    for (int x = minX; x <= maxX; x++)
                    {
                        float dx = x + ox;
                        float bw0 = (a0x * dx + a0y * dy) * invDenom;
                        float bw1 = (a1x * dx + a1y * dy) * invDenom;
                        float bw2 = 1f - bw0 - bw1;

                        if (bw0 < -0.001f || bw1 < -0.001f || bw2 < -0.001f) continue;

                        int texelIdx = atlasRowOff + x;
                        if (score <= pScore[texelIdx]) continue;

                        float sx = bw0 * s0.x + bw1 * s1.x + bw2 * s2.x;
                        float sy = bw0 * s0.y + bw1 * s1.y + bw2 * s2.y;

                        int px = (int)(sx + 0.5f);
                        int screenY = (int)(sy + 0.5f);
                        if ((uint)px >= (uint)kfW || (uint)screenY >= (uint)kfH) continue;

                        Vector3 worldPt = bw0 * p0 + bw1 * p1 + bw2 * p2;
                        Vector3 camPt = viewMat.MultiplyPoint3x4(worldPt);
                        int depthIdx = screenY * kfW + px;
                        if (camPt.z > pDepth[depthIdx] + 0.05f) continue;

                        int pixelIdx = depthIdx * 4;
                        if (pixelIdx + 3 >= pixelLen) continue;

                        int atlasOff = texelIdx * 4;
                        pAtlas[atlasOff] = pPixels[pixelIdx];
                        pAtlas[atlasOff + 1] = pPixels[pixelIdx + 1];
                        pAtlas[atlasOff + 2] = pPixels[pixelIdx + 2];
                        pAtlas[atlasOff + 3] = 255;
                        pScore[texelIdx] = score;
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        //  DENOISE (SPECKLE REMOVAL)
        // ═══════════════════════════════════════════════════════════════

        static void DenoiseAtlas(byte[] atlas, int w, int h)
        {
            const int threshold = 60;
            byte[] clean = new byte[atlas.Length];
            Buffer.BlockCopy(atlas, 0, clean, 0, atlas.Length);

            int replaced = 0;
            for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                int idx = (y * w + x) * 4;
                if (atlas[idx + 3] == 0) continue;

                int cr = atlas[idx], cg = atlas[idx + 1], cb = atlas[idx + 2];

                int sumR = 0, sumG = 0, sumB = 0, count = 0;
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nIdx = ((y + dy) * w + (x + dx)) * 4;
                    if (atlas[nIdx + 3] == 0) continue;
                    sumR += atlas[nIdx];
                    sumG += atlas[nIdx + 1];
                    sumB += atlas[nIdx + 2];
                    count++;
                }

                if (count < 3) continue;

                int avgR = sumR / count, avgG = sumG / count, avgB = sumB / count;
                int diff = Mathf.Abs(cr - avgR) + Mathf.Abs(cg - avgG) + Mathf.Abs(cb - avgB);

                if (diff > threshold)
                {
                    clean[idx] = (byte)avgR;
                    clean[idx + 1] = (byte)avgG;
                    clean[idx + 2] = (byte)avgB;
                    replaced++;
                }
            }

            Buffer.BlockCopy(clean, 0, atlas, 0, atlas.Length);
            Debug.Log($"[TextureRefine] Denoise: replaced {replaced} outlier texels (threshold={threshold})");
        }

        // ═══════════════════════════════════════════════════════════════
        //  DILATION (GAP FILL)
        // ═══════════════════════════════════════════════════════════════

        static void DilateAtlas(byte[] atlas, int w, int h, int passes)
        {
            byte[] temp = new byte[atlas.Length];

            for (int pass = 0; pass < passes; pass++)
            {
                Buffer.BlockCopy(atlas, 0, temp, 0, atlas.Length);
                bool changed = false;

                for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int idx = (y * w + x) * 4;
                    if (atlas[idx + 3] != 0) continue; // already filled

                    int r = 0, g = 0, b = 0, count = 0;
                    for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        int nIdx = (ny * w + nx) * 4;
                        if (atlas[nIdx + 3] == 0) continue;
                        r += atlas[nIdx];
                        g += atlas[nIdx + 1];
                        b += atlas[nIdx + 2];
                        count++;
                    }

                    if (count > 0)
                    {
                        temp[idx] = (byte)(r / count);
                        temp[idx + 1] = (byte)(g / count);
                        temp[idx + 2] = (byte)(b / count);
                        temp[idx + 3] = 255;
                        changed = true;
                    }
                }

                Buffer.BlockCopy(temp, 0, atlas, 0, atlas.Length);
                if (!changed) break;
            }
        }

        static void ReportStatus(string status)
        {
            StatusChanged?.Invoke(status);
        }
    }
}
