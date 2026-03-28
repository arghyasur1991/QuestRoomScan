using System;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    internal struct UnwrappedMeshResult
    {
        public Vector3[] Positions;
        public Vector3[] Normals;
        public Vector2[] UVs;
        public float[] RawUVs;
        public int[] Indices;
        public int AtlasWidth;
        public int AtlasHeight;
        internal Vector3[] OrigPositions;
        internal Vector3[] OrigNormals;
        internal int[] OrigIndices;
    }

    internal struct RefinedTextureResult
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
    /// Attach as an optional module on the same GameObject as <see cref="RoomScanner"/>.
    /// </summary>
    [RequireComponent(typeof(KeyframeCollector))]
    public class TextureRefinement : MonoBehaviour, IRoomScanModule
    {
        public string ModuleName => "Texture Refinement";

        [Header("Bake Pipeline")]
        [SerializeField] internal Shader refinedMeshShader;
        [SerializeField] internal ComputeShader atlasBakeCompute;
        [Tooltip("Force CPU bake path instead of GPU compute")]
        [SerializeField] internal bool forceCpuBake = false;
        [Tooltip("Skip denoise pass after baking")]
        [SerializeField] internal bool skipDenoise = true;
        [Tooltip("Multi-view blend: 2-pass GPU bake that blends top views per texel")]
        [SerializeField] internal bool multiViewBlend = true;
        [Tooltip("Unsharp mask strength (0 = off)")]
        [Range(0f, 2f)]
        [SerializeField] internal float sharpenStrength = 0.8f;
        [Tooltip("Sharpening kernel radius")]
        [Range(1, 4)]
        [SerializeField] internal int sharpenRadius = 2;
        [Tooltip("Blend seams between UV charts")]
        [SerializeField] internal bool enableSeamBlending = true;
        [Tooltip("Seam blending pixel radius")]
        [Range(1, 8)]
        [SerializeField] internal int seamBlendRadius = 3;
        [Tooltip("Minimum score fraction for multi-view blend inclusion")]
        [Range(0.1f, 0.9f)]
        [SerializeField] internal float blendMinFraction = 0.3f;

        [Header("HQ Server Refinement")]
        [Tooltip("Server-side atlas super-resolution scale")]
        [Range(1, 4)]
        [SerializeField] internal int hqRefineScale = 2;

        [Header("Unwrap")]
        [Tooltip("Simplify mesh before UV unwrap (1.0 = no simplification)")]
        [Range(0.1f, 1f)]
        [SerializeField] internal float decimationRatio = 1f;
        [Tooltip("Align charts to 4x4 blocks for faster packing")]
        [SerializeField] internal bool useBlockAlign = true;
        [Tooltip("Chart growth cost limit")]
        [Range(0.5f, 4f)]
        [SerializeField] internal float xatlasMaxCost = 1.5f;

        private RoomScanner _scanner;

        public void OnModuleInitialize(RoomScanner scanner)
        {
            _scanner = scanner;
        }

        private const int GpuVertexStride = 32;

        internal event Action<string> StatusChanged;

        // ═══════════════════════════════════════════════════════════════
        //  UV UNWRAP (shared prerequisite for both on-device and HQ refine)
        // ═══════════════════════════════════════════════════════════════

        internal Task<UnwrappedMeshResult> UnwrapMeshAsync(
            string keyframeDir, Matrix4x4 keyframeRelocation, int atlasResolution = 2048)
        {
            var opts = XAtlasWrapper.UnwrapOptions.Default;
            opts.Resolution = (uint)atlasResolution;
            opts.MaxCost = xatlasMaxCost;
            opts.BlockAlign = useBlockAlign;
            return UnwrapMeshAsync(keyframeDir, keyframeRelocation, opts);
        }

        internal async Task<UnwrappedMeshResult> UnwrapMeshAsync(
            string keyframeDir, Matrix4x4 keyframeRelocation,
            XAtlasWrapper.UnwrapOptions opts)
        {
            ReportStatus("Reading mesh from GPU...");
            var (positions, normals, colors, indices) = await ReadbackMeshAsync();
            if (positions == null || positions.Length == 0)
                throw new InvalidOperationException("Mesh readback returned no vertices");

            Logger.Info($"[TextureRefine] Readback: {positions.Length} verts, {indices.Length / 3} tris");

            Vector3[] inPos = positions;
            Vector3[] inNorm = normals;
            int[] inIdx = indices;

            float decRatio = decimationRatio;
            if (decRatio < 1f && decRatio > 0f)
            {
                ReportStatus($"Simplifying mesh ({decRatio:P0})...");
                await Task.Run(() =>
                {
                    float[] flatPos = new float[inPos.Length * 3];
                    for (int i = 0; i < inPos.Length; i++)
                    {
                        flatPos[i * 3] = inPos[i].x;
                        flatPos[i * 3 + 1] = inPos[i].y;
                        flatPos[i * 3 + 2] = inPos[i].z;
                    }
                    var sr = XAtlasWrapper.Simplify(flatPos, inPos.Length,
                        inIdx, inIdx.Length, decRatio);
                    inIdx = new int[sr.IndexCount];
                    for (int i = 0; i < sr.IndexCount; i++)
                        inIdx[i] = (int)sr.Indices[i];
                });
                Logger.Info($"[TextureRefine] Simplified: {inIdx.Length / 3} tris (was {indices.Length / 3})");
            }

            ReportStatus("UV unwrapping...");
            XAtlasWrapper.Result uvResult = default;

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
                    inIdx, inIdx.Length, opts);
            });

            if (uvResult.VertexCount == 0)
                throw new InvalidOperationException("xatlas produced no output vertices");

            int atlasW = uvResult.AtlasWidth;
            int atlasH = uvResult.AtlasHeight;
            Logger.Info($"[TextureRefine] xatlas: {uvResult.VertexCount} verts, " +
                      $"{uvResult.IndexCount / 3} tris, atlas {atlasW}x{atlasH}");

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

            return new UnwrappedMeshResult
            {
                Positions = outPos,
                Normals = outNorm,
                UVs = outUVs,
                RawUVs = uvResult.UVs,
                Indices = uvResult.Indices,
                AtlasWidth = atlasW,
                AtlasHeight = atlasH,
                OrigPositions = inPos,
                OrigNormals = inNorm,
                OrigIndices = inIdx,
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
                Logger.Error("[TextureRefine] GpuSurfaceNets or its buffers are null");
                return (null, null, null, null);
            }

            Logger.Info("[TextureRefine] Starting GPU readback...");

            // Read all three buffers using callback-based readback
            // (copies NativeArray to managed array immediately in callback frame)
            byte[] counterBytes = await ReadbackBytesAsync(gpuSN.CountersBuffer);
            if (counterBytes == null)
            {
                Logger.Error("[TextureRefine] Counter readback failed");
                return (null, null, null, null);
            }

            int vertCount = BitConverter.ToInt32(counterBytes, 0);
            int idxCount = counterBytes.Length >= 8 ? BitConverter.ToInt32(counterBytes, 4) : 0;

            Logger.Info($"[TextureRefine] Counters: verts={vertCount}, idx={idxCount}");

            if (vertCount <= 0 || idxCount <= 0)
            {
                Logger.Warning($"[TextureRefine] No mesh data: verts={vertCount}, idx={idxCount}");
                return (null, null, null, null);
            }

            byte[] vertData = await ReadbackBytesAsync(gpuSN.VertexBuffer);
            if (vertData == null)
            {
                Logger.Error("[TextureRefine] Vertex readback failed");
                return (null, null, null, null);
            }

            byte[] idxData = await ReadbackBytesAsync(gpuSN.IndexBuffer);
            if (idxData == null)
            {
                Logger.Error("[TextureRefine] Index readback failed");
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

            Logger.Info($"[TextureRefine] Readback complete: {vertCount} verts, {idxCount / 3} tris");
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

        /// <summary>
        /// Returns the contents of frames.jsonl with poses transformed by the relocation matrix.
        /// Used by GS training and HQ refine upload to send corrected poses to the server.
        /// </summary>
        public static byte[] RelocateFramesJsonl(string keyframeDir, Matrix4x4 relocation)
        {
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");
            if (!File.Exists(manifestPath)) return null;

            if (relocation == Matrix4x4.identity)
                return File.ReadAllBytes(manifestPath);

            var ci = System.Globalization.CultureInfo.InvariantCulture;
            string[] lines = File.ReadAllLines(manifestPath);
            var sb = new System.Text.StringBuilder(lines.Length * 256);
            var rot = relocation.rotation;

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var kf = ParseKeyframe(line, "");
                    var pos = relocation.MultiplyPoint3x4(kf.Position);
                    var q = rot * kf.Rotation;

                    string relocated = System.Text.RegularExpressions.Regex.Replace(line,
                        "\"px\":[^,}]+", $"\"px\":{pos.x.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"py\":[^,}]+", $"\"py\":{pos.y.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"pz\":[^,}]+", $"\"pz\":{pos.z.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qx\":[^,}]+", $"\"qx\":{q.x.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qy\":[^,}]+", $"\"qy\":{q.y.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qz\":[^,}]+", $"\"qz\":{q.z.ToString("F6", ci)}");
                    relocated = System.Text.RegularExpressions.Regex.Replace(relocated,
                        "\"qw\":[^,}]+", $"\"qw\":{q.w.ToString("F6", ci)}");

                    sb.AppendLine(relocated);
                }
                catch { sb.AppendLine(line); }
            }
            return System.Text.Encoding.UTF8.GetBytes(sb.ToString());
        }

        static System.Collections.Generic.List<Keyframe> ParseKeyframeManifest(
            string keyframeDir, Matrix4x4 keyframeRelocation)
        {
            string manifestPath = Path.Combine(keyframeDir, "frames.jsonl");
            string imagesDir = Path.Combine(keyframeDir, "images");
            var metaList = new System.Collections.Generic.List<Keyframe>();

            if (!File.Exists(manifestPath)) return metaList;

            string[] lines = File.ReadAllLines(manifestPath);
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var kf = ParseKeyframe(line, imagesDir);
                    if (string.IsNullOrEmpty(kf.JpgPath)) continue;
                    if (keyframeRelocation != Matrix4x4.identity)
                    {
                        kf.Position = keyframeRelocation.MultiplyPoint3x4(kf.Position);
                        kf.Rotation = keyframeRelocation.rotation * kf.Rotation;
                    }
                    metaList.Add(kf);
                }
                catch (Exception e)
                {
                    Logger.Warning($"[TextureRefine] Skip keyframe: {e.Message}");
                }
            }
            return metaList;
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
        /// GPU-accelerated atlas baking via compute shader.
        /// Mirrors the CPU rasterization logic 1:1 using integer pixel indexing
        /// on StructuredBuffers — no texture UV ambiguity, no framebuffer orientation issues.
        /// Falls back to CPU path if the compute shader is unavailable.
        /// </summary>
        internal async Task<byte[]> BakeAtlasAsync(
            UnwrappedMeshResult mesh, string keyframeDir, Matrix4x4 keyframeRelocation)
        {
            ComputeShader compute = forceCpuBake ? null : atlasBakeCompute;
            if (compute == null)
            {
                Logger.Warning("[TextureRefine] No compute shader, falling back to CPU bake");
                return await BakeAtlasCPUAsync(mesh, keyframeDir, keyframeRelocation);
            }

            ReportStatus("Loading keyframe metadata...");
            var metaList = ParseKeyframeManifest(keyframeDir, keyframeRelocation);
            if (metaList.Count == 0)
                throw new InvalidOperationException("No keyframes available for baking");

            int total = metaList.Count;
            Logger.Info($"[TextureRefine] GPU compute bake: {total} keyframes" +
                (keyframeRelocation != Matrix4x4.identity ? " (relocated)" : ""));

            int atlasW = mesh.AtlasWidth;
            int atlasH = mesh.AtlasHeight;
            int texelCount = atlasW * atlasH;
            int origTriCount = mesh.OrigIndices.Length / 3;
            int outTriCount = mesh.Indices.Length / 3;

            // Find kernels
            int kClear = compute.FindKernel("ClearDepth");
            int kDepth = compute.FindKernel("BuildDepth");
            int kBake  = compute.FindKernel("BakeAtlas");

            // ── Create persistent GPU buffers (mesh data + atlas) ──
            var origPosBuf = new ComputeBuffer(mesh.OrigPositions.Length, 12);
            origPosBuf.SetData(mesh.OrigPositions);
            var origIdxBuf = new ComputeBuffer(mesh.OrigIndices.Length, 4);
            origIdxBuf.SetData(mesh.OrigIndices);

            var outPosBuf = new ComputeBuffer(mesh.Positions.Length, 12);
            outPosBuf.SetData(mesh.Positions);
            var outNormBuf = new ComputeBuffer(mesh.Normals.Length, 12);
            outNormBuf.SetData(mesh.Normals);
            var outIdxBuf = new ComputeBuffer(mesh.Indices.Length, 4);
            outIdxBuf.SetData(mesh.Indices);

            var rawUV2 = new Vector2[mesh.Positions.Length];
            for (int i = 0; i < rawUV2.Length; i++)
                rawUV2[i] = new Vector2(mesh.RawUVs[i * 2], mesh.RawUVs[i * 2 + 1]);
            var rawUVBuf = new ComputeBuffer(rawUV2.Length, 8);
            rawUVBuf.SetData(rawUV2);

            var scoreBuf = new ComputeBuffer(texelCount, 4);
            scoreBuf.SetData(new uint[texelCount]);
            var atlasBuf = new ComputeBuffer(texelCount, 4);
            atlasBuf.SetData(new uint[texelCount]);

            // Bind static buffers to kernels
            compute.SetBuffer(kDepth, "_OrigPos", origPosBuf);
            compute.SetBuffer(kDepth, "_OrigIdx", origIdxBuf);
            compute.SetBuffer(kBake, "_OutPos", outPosBuf);
            compute.SetBuffer(kBake, "_OutNorm", outNormBuf);
            compute.SetBuffer(kBake, "_OutIdx", outIdxBuf);
            compute.SetBuffer(kBake, "_RawUV", rawUVBuf);
            compute.SetBuffer(kBake, "_ScoreBuf", scoreBuf);
            compute.SetBuffer(kBake, "_AtlasBuf", atlasBuf);
            compute.SetInt("_AtlasW", atlasW);
            compute.SetInt("_AtlasH", atlasH);
            compute.SetInt("_OrigTriCount", origTriCount);
            compute.SetInt("_OutTriCount", outTriCount);

            // Per-keyframe buffers (created on first use, resized as needed)
            ComputeBuffer depthBuf = null;
            ComputeBuffer kfPixelBuf = null;

            ReportStatus("Baking textures (GPU compute)...");
            int bakeCount = 0;

            for (int ki = 0; ki < total; ki++)
            {
                var kf = metaList[ki];
                if (string.IsNullOrEmpty(kf.JpgPath)) continue;

                byte[] jpgBytes;
                try { jpgBytes = await ReadFileAsync(kf.JpgPath); }
                catch { continue; }

                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (!ImageConversion.LoadImage(tex, jpgBytes))
                {
                    UnityEngine.Object.Destroy(tex);
                    continue;
                }
                kf.Width = tex.width;
                kf.Height = tex.height;

                // GetPixels32: same byte layout as CPU kf.Pixels — no Y ambiguity
                Color32[] colors = tex.GetPixels32();
                UnityEngine.Object.Destroy(tex);

                int imgW = kf.Width, imgH = kf.Height;
                int imgPixels = imgW * imgH;

                // Create/resize per-keyframe buffers
                if (depthBuf == null || depthBuf.count != imgPixels)
                {
                    depthBuf?.Release();
                    depthBuf = new ComputeBuffer(imgPixels, 4);
                    kfPixelBuf?.Release();
                    kfPixelBuf = new ComputeBuffer(imgPixels, 4);
                }
                kfPixelBuf.SetData(colors);

                // Per-keyframe uniforms
                int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
                int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
                float cropX = (sw - kf.Width) * 0.5f;
                float cropY = (sh - kf.Height) * 0.5f;
                Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;

                compute.SetMatrix("_ViewMat", viewMat);
                compute.SetVector("_CamPos", new Vector4(kf.Position.x, kf.Position.y, kf.Position.z, 1f));
                compute.SetFloat("_Fx", kf.Fx);
                compute.SetFloat("_Fy", kf.Fy);
                compute.SetFloat("_Cx", kf.Cx);
                compute.SetFloat("_Cy", kf.Cy);
                compute.SetFloat("_CropX", cropX);
                compute.SetFloat("_CropY", cropY);
                compute.SetInt("_ImgW", imgW);
                compute.SetInt("_ImgH", imgH);

                // Bind per-keyframe buffers
                compute.SetBuffer(kClear, "_DepthBuf", depthBuf);
                compute.SetBuffer(kDepth, "_DepthBuf", depthBuf);
                compute.SetBuffer(kBake, "_DepthBuf", depthBuf);
                compute.SetBuffer(kBake, "_KfPixels", kfPixelBuf);

                // Dispatch: clear depth → build depth → bake atlas
                compute.Dispatch(kClear, (imgPixels + 255) / 256, 1, 1);
                compute.Dispatch(kDepth, (origTriCount + 63) / 64, 1, 1);
                compute.Dispatch(kBake, (outTriCount + 63) / 64, 1, 1);

                bakeCount++;
                if (bakeCount % 20 == 0 || bakeCount < 3)
                {
                    ReportStatus($"Baking (GPU)... {bakeCount}/{total}");
                    Logger.Info($"[TextureRefine] GPU baked keyframe {bakeCount}/{total}");
                }

                await Task.Yield();
            }

            Logger.Info($"[TextureRefine] GPU baked {bakeCount} keyframes total (pass 1)");

            // ── Pass 2: Multi-view blend accumulation (optional) ──
            // Re-iterates keyframes, accumulating score-weighted colors from all
            // qualifying views into fixed-point buffers, then resolves to final atlas.
            if (multiViewBlend)
            {
                int kAccum = compute.FindKernel("BlendAccum");
                int kResolve = compute.FindKernel("ResolveBlend");

                var accumR = new ComputeBuffer(texelCount, 4);
                var accumG = new ComputeBuffer(texelCount, 4);
                var accumB = new ComputeBuffer(texelCount, 4);
                var accumW = new ComputeBuffer(texelCount, 4);
                accumR.SetData(new uint[texelCount]);
                accumG.SetData(new uint[texelCount]);
                accumB.SetData(new uint[texelCount]);
                accumW.SetData(new uint[texelCount]);

                compute.SetBuffer(kAccum, "_OutPos", outPosBuf);
                compute.SetBuffer(kAccum, "_OutNorm", outNormBuf);
                compute.SetBuffer(kAccum, "_OutIdx", outIdxBuf);
                compute.SetBuffer(kAccum, "_RawUV", rawUVBuf);
                compute.SetBuffer(kAccum, "_AccumR", accumR);
                compute.SetBuffer(kAccum, "_AccumG", accumG);
                compute.SetBuffer(kAccum, "_AccumB", accumB);
                compute.SetBuffer(kAccum, "_AccumW", accumW);
                compute.SetBuffer(kAccum, "_BestScore", scoreBuf);
                compute.SetFloat("_BlendMinFraction", blendMinFraction);

                ReportStatus("Multi-view blending (pass 2)...");
                int blendCount = 0;

                for (int ki = 0; ki < total; ki++)
                {
                    var kf = metaList[ki];
                    if (string.IsNullOrEmpty(kf.JpgPath)) continue;

                    byte[] jpgBytes;
                    try { jpgBytes = await ReadFileAsync(kf.JpgPath); }
                    catch { continue; }

                    var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                    if (!ImageConversion.LoadImage(tex, jpgBytes))
                    {
                        UnityEngine.Object.Destroy(tex);
                        continue;
                    }
                    kf.Width = tex.width;
                    kf.Height = tex.height;

                    Color32[] colors = tex.GetPixels32();
                    UnityEngine.Object.Destroy(tex);

                    int imgW = kf.Width, imgH = kf.Height;
                    int imgPixels = imgW * imgH;

                    if (depthBuf == null || depthBuf.count != imgPixels)
                    {
                        depthBuf?.Release();
                        depthBuf = new ComputeBuffer(imgPixels, 4);
                        kfPixelBuf?.Release();
                        kfPixelBuf = new ComputeBuffer(imgPixels, 4);
                    }
                    kfPixelBuf.SetData(colors);

                    int sw = kf.SensorWidth > 0 ? kf.SensorWidth : kf.Width;
                    int sh = kf.SensorHeight > 0 ? kf.SensorHeight : kf.Height;
                    float cropX = (sw - kf.Width) * 0.5f;
                    float cropY = (sh - kf.Height) * 0.5f;
                    Matrix4x4 viewMat = Matrix4x4.TRS(kf.Position, kf.Rotation, Vector3.one).inverse;

                    compute.SetMatrix("_ViewMat", viewMat);
                    compute.SetVector("_CamPos", new Vector4(kf.Position.x, kf.Position.y, kf.Position.z, 1f));
                    compute.SetFloat("_Fx", kf.Fx);
                    compute.SetFloat("_Fy", kf.Fy);
                    compute.SetFloat("_Cx", kf.Cx);
                    compute.SetFloat("_Cy", kf.Cy);
                    compute.SetFloat("_CropX", cropX);
                    compute.SetFloat("_CropY", cropY);
                    compute.SetInt("_ImgW", imgW);
                    compute.SetInt("_ImgH", imgH);

                    compute.SetBuffer(kClear, "_DepthBuf", depthBuf);
                    compute.SetBuffer(kDepth, "_DepthBuf", depthBuf);
                    compute.SetBuffer(kAccum, "_DepthBuf", depthBuf);
                    compute.SetBuffer(kAccum, "_KfPixels", kfPixelBuf);

                    compute.Dispatch(kClear, (imgPixels + 255) / 256, 1, 1);
                    compute.Dispatch(kDepth, (origTriCount + 63) / 64, 1, 1);
                    compute.Dispatch(kAccum, (outTriCount + 63) / 64, 1, 1);

                    blendCount++;
                    if (blendCount % 20 == 0 || blendCount < 3)
                    {
                        ReportStatus($"Multi-view blend... {blendCount}/{total}");
                        Logger.Info($"[TextureRefine] Blend pass keyframe {blendCount}/{total}");
                    }

                    await Task.Yield();
                }

                Logger.Info($"[TextureRefine] Blend pass 2 complete: {blendCount} keyframes");

                // Resolve: divide accumulated colors by weights → final atlas
                compute.SetBuffer(kResolve, "_AccumRIn", accumR);
                compute.SetBuffer(kResolve, "_AccumGIn", accumG);
                compute.SetBuffer(kResolve, "_AccumBIn", accumB);
                compute.SetBuffer(kResolve, "_AccumWIn", accumW);
                compute.SetBuffer(kResolve, "_AtlasBuf", atlasBuf);
                compute.Dispatch(kResolve, (texelCount + 255) / 256, 1, 1);

                accumR.Release(); accumG.Release();
                accumB.Release(); accumW.Release();
                Logger.Info("[TextureRefine] Multi-view blend resolved");
            }

            // ── Sharpening pass (GPU unsharp mask) ──
            if (sharpenStrength > 0.01f)
            {
                ReportStatus("Sharpening...");
                int kSharpen = compute.FindKernel("SharpenAtlas");
                var sharpenSrcBuf = new ComputeBuffer(texelCount, 4);
                var tmp = new uint[texelCount];
                atlasBuf.GetData(tmp);
                sharpenSrcBuf.SetData(tmp);

                compute.SetFloat("_SharpenStrength", sharpenStrength);
                compute.SetInt("_SharpenRadius", sharpenRadius);
                compute.SetInt("_AtlasW", atlasW);
                compute.SetInt("_AtlasH", atlasH);
                compute.SetBuffer(kSharpen, "_AtlasBufSrc", sharpenSrcBuf);
                compute.SetBuffer(kSharpen, "_AtlasBuf", atlasBuf);

                int groupsX = (atlasW + 7) / 8;
                int groupsY = (atlasH + 7) / 8;
                compute.Dispatch(kSharpen, groupsX, groupsY, 1);

                sharpenSrcBuf.Release();
                Logger.Info($"[TextureRefine] Sharpening complete (strength={sharpenStrength}, radius={sharpenRadius})");
            }

            // ── Seam blending pass (GPU) ──
            int kSeam = -1;
            try { kSeam = compute.FindKernel("BlendSeams"); }
            catch { /* kernel not available in older shader variants */ }

            if (kSeam >= 0 && enableSeamBlending)
            {
                ReportStatus("Blending seams...");
                var seamSrcBuf = new ComputeBuffer(texelCount, 4);
                var tmp = new uint[texelCount];
                atlasBuf.GetData(tmp);
                seamSrcBuf.SetData(tmp);

                compute.SetInt("_AtlasW", atlasW);
                compute.SetInt("_AtlasH", atlasH);
                compute.SetInt("_BlendRadius", seamBlendRadius);
                compute.SetBuffer(kSeam, "_AtlasBufSrc", seamSrcBuf);
                compute.SetBuffer(kSeam, "_AtlasBuf", atlasBuf);

                int groupsX = (atlasW + 7) / 8;
                int groupsY = (atlasH + 7) / 8;
                compute.Dispatch(kSeam, groupsX, groupsY, 1);

                seamSrcBuf.Release();
                Logger.Info("[TextureRefine] Seam blending complete");
            }

            // Readback atlas buffer
            ReportStatus("Reading back atlas...");
            byte[] atlasPixels = await ReadbackComputeBufferAsync(atlasBuf, texelCount);

            // Log fill stats
            {
                int filled = 0;
                for (int i = 0; i < texelCount; i++)
                    if (atlasPixels[i * 4 + 3] != 0) filled++;
                Logger.Info($"[TextureRefine] GPU bake pre-dilation: {filled}/{texelCount} texels filled " +
                    $"({100f * filled / texelCount:F1}%)");
            }

            // Post-process on background thread
            if (!skipDenoise)
            {
                ReportStatus("Denoising...");
                await Task.Run(() => DenoiseAtlas(atlasPixels, atlasW, atlasH));
            }

            ReportStatus("Filling gaps...");
            await Task.Run(() => DilateAtlas(atlasPixels, atlasW, atlasH, 8));

            // Cleanup
            origPosBuf.Release(); origIdxBuf.Release();
            outPosBuf.Release(); outNormBuf.Release(); outIdxBuf.Release();
            rawUVBuf.Release(); scoreBuf.Release(); atlasBuf.Release();
            depthBuf?.Release(); kfPixelBuf?.Release();

            ReportStatus("Done");
            Logger.Info($"[TextureRefine] GPU compute bake complete: {atlasW}x{atlasH} atlas");

            return atlasPixels;
        }

        // ═══════════════════════════════════════════════════════════════
        //  GPU COMPUTE HELPERS
        // ═══════════════════════════════════════════════════════════════

        static async Task<byte[]> ReadFileAsync(string path)
        {
#if UNITY_2021_3_OR_NEWER
            return await File.ReadAllBytesAsync(path);
#else
            return await Task.Run(() => File.ReadAllBytes(path));
#endif
        }

        static Task<byte[]> ReadbackComputeBufferAsync(ComputeBuffer buffer, int elementCount)
        {
            var tcs = new TaskCompletionSource<byte[]>();
            AsyncGPUReadback.Request(buffer, elementCount * 4, 0, request =>
            {
                if (request.hasError)
                {
                    Logger.Error("[TextureRefine] Compute buffer readback failed");
                    tcs.SetResult(new byte[elementCount * 4]);
                    return;
                }
                var native = request.GetData<byte>();
                byte[] managed = new byte[native.Length];
                NativeArray<byte>.Copy(native, managed, native.Length);
                tcs.SetResult(managed);
            });
            return tcs.Task;
        }

        // ═══════════════════════════════════════════════════════════════
        //  CPU BAKE FALLBACK
        // ═══════════════════════════════════════════════════════════════

        /// <summary>
        /// CPU-based atlas baking fallback.
        /// Decodes one JPEG at a time on the main thread to avoid OOM, bakes on BG thread.
        /// </summary>
        async Task<byte[]> BakeAtlasCPUAsync(
            UnwrappedMeshResult mesh, string keyframeDir, Matrix4x4 keyframeRelocation)
        {
            ReportStatus("Loading keyframe metadata...");
            var metaList = ParseKeyframeManifest(keyframeDir, keyframeRelocation);
            if (metaList.Count == 0)
                throw new InvalidOperationException("No keyframes available for baking");

            Logger.Info($"[TextureRefine] Found {metaList.Count} keyframes" +
                (keyframeRelocation != Matrix4x4.identity ? " (relocated)" : ""));

            ReportStatus("Baking textures...");
            Vector3[] inPos = mesh.OrigPositions;
            Vector3[] inNorm = mesh.OrigNormals;
            int[] inIdx = mesh.OrigIndices;
            Vector3[] outPos = mesh.Positions;
            Vector3[] outNorm = mesh.Normals;
            float[] rawUVs = mesh.RawUVs;
            int[] outIndices = mesh.Indices;
            int outVertCount = mesh.Positions.Length;
            int atlasW = mesh.AtlasWidth;
            int atlasH = mesh.AtlasHeight;
            int texelCount = atlasW * atlasH;
            byte[] atlasPixels = new byte[texelCount * 4];
            float[] bestScore = new float[texelCount];

            TriData[] triData = null;
            await Task.Run(() => triData = PrecomputeTriData(outPos, outNorm, rawUVs, outIndices, outVertCount));

            for (int ki = 0; ki < metaList.Count; ki++)
            {
                var kf = metaList[ki];
                if (string.IsNullOrEmpty(kf.JpgPath)) continue;

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

                if (ki == 0)
                {
                    Logger.Info($"[TextureRefine] KF0: pos={kf.Position}, rot={kf.Rotation}, " +
                        $"fx={kf.Fx}, fy={kf.Fy}, cx={kf.Cx}, cy={kf.Cy}, " +
                        $"imgSize={kf.Width}x{kf.Height}, pixLen={kf.Pixels.Length}");
                }

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
                    Logger.Info($"[TextureRefine] Baked keyframe {ki + 1}/{metaList.Count}");
                }
            }

            {
                int filled = 0;
                for (int i = 0; i < texelCount; i++)
                    if (atlasPixels[i * 4 + 3] != 0) filled++;
                Logger.Info($"[TextureRefine] Pre-dilation: {filled}/{texelCount} texels filled " +
                    $"({100f * filled / texelCount:F1}%)");
            }

            if (!skipDenoise)
            {
                ReportStatus("Denoising...");
                await Task.Run(() => DenoiseAtlas(atlasPixels, atlasW, atlasH));
            }

            ReportStatus("Filling gaps...");
            await Task.Run(() => DilateAtlas(atlasPixels, atlasW, atlasH, 8));

            ReportStatus("Done");
            Logger.Info($"[TextureRefine] Complete: {atlasW}x{atlasH} atlas");

            return atlasPixels;
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

            for (int t = 0; t < triCount; t++)
            {
                int i0, i1, i2;
                Vector3 faceNormal, centroid;
                float u0, v0, u1, v1, u2, v2;

                if (triData != null)
                {
                    ref readonly TriData td = ref triData[t];
                    if (td.I0 < 0) continue;
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
                    if (i0 >= outVertCount || i1 >= outVertCount || i2 >= outVertCount) continue;
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
                if (dot <= 0.05f) continue;

                Vector3 p0 = outPos[i0], p1 = outPos[i1], p2 = outPos[i2];
                Vector2 s0 = ProjectToScreen(p0, viewMat, kf);
                Vector2 s1 = ProjectToScreen(p1, viewMat, kf);
                Vector2 s2 = ProjectToScreen(p2, viewMat, kf);

                if (!IsInFrustum(s0, kf.Width, kf.Height) &&
                    !IsInFrustum(s1, kf.Width, kf.Height) &&
                    !IsInFrustum(s2, kf.Width, kf.Height))
                    continue;

                float dist = Vector3.Distance(camPos, centroid);
                float score = dot / Mathf.Max(dist, 0.1f);

                RasterizeTriangle(atlas, bestScore, atlasW, atlasH,
                    u0, v0, u1, v1, u2, v2,
                    s0, s1, s2,
                    score, kf, depthBuf, p0, p1, p2, viewMat);
            }
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

                        int px = Mathf.RoundToInt(sx);
                        int screenY = Mathf.RoundToInt(sy);
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
            Logger.Info($"[TextureRefine] Denoise: replaced {replaced} outlier texels (threshold={threshold})");
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

        void ReportStatus(string status)
        {
            StatusChanged?.Invoke(status);
        }
    }
}
