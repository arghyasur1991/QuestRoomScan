using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class TriplanarCache : MonoBehaviour
    {
        public static TriplanarCache Instance { get; private set; }

        [SerializeField] private ComputeShader bakeCompute;
        [SerializeField, Range(512, 8192)] private int textureResolution = 4096;
        [SerializeField, Tooltip("Auto-calculated if 0. Higher = faster fill but more GPU work per pixel")]
        private int splatRadiusOverride = 0;

        private RenderTexture _triXZ, _triXY, _triYZ;
        private RenderTexture _depthXZ, _depthXY, _depthYZ;
        private RenderTexture _camFrameCopy;
        private ComputeKernelHelper _bakeKernel;
        private ComputeKernelHelper _clearKernel;
        private bool _kernelsReady;

        public RenderTexture TriXZ => _triXZ;
        public RenderTexture TriXY => _triXY;
        public RenderTexture TriYZ => _triYZ;
        public RenderTexture DepthXZ => _depthXZ;
        public RenderTexture DepthXY => _depthXY;
        public RenderTexture DepthYZ => _depthYZ;

        static readonly int TriXZID = Shader.PropertyToID("_RSTriXZ");
        static readonly int TriXYID = Shader.PropertyToID("_RSTriXY");
        static readonly int TriYZID = Shader.PropertyToID("_RSTriYZ");
        static readonly int TriAvailableID = Shader.PropertyToID("_RSTriAvailable");

        static readonly int TriXZRWID = Shader.PropertyToID("gsTriXZ_RW");
        static readonly int TriXYRWID = Shader.PropertyToID("gsTriXY_RW");
        static readonly int TriYZRWID = Shader.PropertyToID("gsTriYZ_RW");
        static readonly int DepthXZRWID = Shader.PropertyToID("gsTriDepthXZ_RW");
        static readonly int DepthXYRWID = Shader.PropertyToID("gsTriDepthXY_RW");
        static readonly int DepthYZRWID = Shader.PropertyToID("gsTriDepthYZ_RW");
        static readonly int TriSizeID = Shader.PropertyToID("gsTriSize");
        static readonly int CamRGBID = Shader.PropertyToID("gsCamRGB");
        static readonly int CamPosID = Shader.PropertyToID("gsCamPos");
        static readonly int CamInvRotID = Shader.PropertyToID("gsCamInvRot");
        static readonly int CamFocalLenID = Shader.PropertyToID("gsCamFocalLen");
        static readonly int CamPrincipalPtID = Shader.PropertyToID("gsCamPrincipalPt");
        static readonly int CamSensorResID = Shader.PropertyToID("gsCamSensorRes");
        static readonly int CamCurrentResID = Shader.PropertyToID("gsCamCurrentRes");
        static readonly int CamExposureID = Shader.PropertyToID("gsCamExposure");
        static readonly int MaxUpdateDistID = Shader.PropertyToID("gsMaxUpdateDist");

        private void Awake() => Instance = this;

        private void Start()
        {
            CreateTextures();
            if (bakeCompute != null)
            {
                _bakeKernel = new ComputeKernelHelper(bakeCompute, "BakeTriplanar");
                _clearKernel = new ComputeKernelHelper(bakeCompute, "ClearTriplanar");
                _kernelsReady = true;
            }
            Clear();
            UpdateShaderGlobals();

            long bytes = (long)textureResolution * textureResolution * 4 * 3;
            long mb = bytes / (1024 * 1024);
            int splatR = splatRadiusOverride > 0 ? splatRadiusOverride : Mathf.Max(1, textureResolution / 2048);
            Debug.Log($"[RoomScan] Triplanar cache: 3x {textureResolution}x{textureResolution} RGBA8 = {mb}MB, splatR={splatR}");
            if (mb > 300)
                Debug.LogWarning($"[RoomScan] Triplanar memory {mb}MB is very high! Consider reducing textureResolution (4096 = 192MB).");
        }

        private void OnDestroy()
        {
            if (_triXZ) Destroy(_triXZ);
            if (_triXY) Destroy(_triXY);
            if (_triYZ) Destroy(_triYZ);
            if (_depthXZ) Destroy(_depthXZ);
            if (_depthXY) Destroy(_depthXY);
            if (_depthYZ) Destroy(_depthYZ);
            if (_camFrameCopy) Destroy(_camFrameCopy);
        }

        private void CreateTextures()
        {
            _triXZ = CreateTriplanarRT("TriXZ");
            _triXY = CreateTriplanarRT("TriXY");
            _triYZ = CreateTriplanarRT("TriYZ");
            _depthXZ = CreateDepthRT("DepthXZ");
            _depthXY = CreateDepthRT("DepthXY");
            _depthYZ = CreateDepthRT("DepthYZ");
        }

        private RenderTexture CreateTriplanarRT(string rtName)
        {
            var rt = new RenderTexture(textureResolution, textureResolution, 0, GraphicsFormat.R8G8B8A8_UNorm)
            {
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                name = rtName
            };
            rt.Create();
            return rt;
        }

        private RenderTexture CreateDepthRT(string rtName)
        {
            var rt = new RenderTexture(textureResolution, textureResolution, 0, GraphicsFormat.R8_UNorm)
            {
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = rtName
            };
            rt.Create();
            return rt;
        }

        public void Clear()
        {
            if (!_kernelsReady) return;
            _clearKernel.Set(TriXZRWID, _triXZ);
            _clearKernel.Set(TriXYRWID, _triXY);
            _clearKernel.Set(TriYZRWID, _triYZ);
            _clearKernel.Set(DepthXZRWID, _depthXZ);
            _clearKernel.Set(DepthXYRWID, _depthXY);
            _clearKernel.Set(DepthYZRWID, _depthYZ);
            bakeCompute.SetInts(TriSizeID, textureResolution, textureResolution);
            _clearKernel.DispatchFit(textureResolution, textureResolution);
        }

        public void UpdateShaderGlobals()
        {
            if (_triXZ) Shader.SetGlobalTexture(TriXZID, _triXZ);
            if (_triXY) Shader.SetGlobalTexture(TriXYID, _triXY);
            if (_triYZ) Shader.SetGlobalTexture(TriYZID, _triYZ);
            Shader.SetGlobalFloat(TriAvailableID, _triXZ != null ? 1f : 0f);
        }

        /// <summary>
        /// Resample triplanar textures from old coordinate frame into new frame using
        /// forward-splat compute dispatch: each old texel has its exact 3D position
        /// (from stored depth), gets relocated by R, and scatter-written at the correct
        /// new triplanar UV. 3 compute dispatches (one per src face), each writing to
        /// all 3 dst faces simultaneously. Followed by compute dilation for coverage gaps.
        /// </summary>
        public void BakeRelocation(Matrix4x4 relocationMatrix, string triplanarDir)
        {
            var vi = VolumeIntegrator.Instance;
            if (vi == null || bakeCompute == null) return;

            var vc = vi.VoxelCount;
            int res = textureResolution;

            // Load old textures as Texture2D (SRV-safe, no Blit→RT Vulkan issue).
            var srcTriFiles = new[] { "tri_xz.raw", "tri_xy.raw", "tri_yz.raw" };
            var srcDepthFiles = new[] { "depth_xz.raw", "depth_xy.raw", "depth_yz.raw" };
            var srcTris = new Texture2D[3];
            var srcDepths = new Texture2D[3];
            for (int i = 0; i < 3; i++)
            {
                srcTris[i] = LoadTex2D(Path.Combine(triplanarDir, srcTriFiles[i]),
                                       res, res, TextureFormat.RGBA32);
                srcDepths[i] = LoadTex2D(Path.Combine(triplanarDir, srcDepthFiles[i]),
                                         res, res, TextureFormat.R8);
            }

            var dsts = new RenderTexture[] {
                CreateTriplanarRT("RelocXZ"),
                CreateTriplanarRT("RelocXY"),
                CreateTriplanarRT("RelocYZ")
            };

            // Forward-splat compute: 3 dispatches (one per src face).
            var relocKernel = new ComputeKernelHelper(bakeCompute, "ForwardSplatRelocation");
            int srcTriID = Shader.PropertyToID("gsRelocSrcTri");
            int srcDepthID = Shader.PropertyToID("gsRelocSrcDepth");
            int dstXZID = Shader.PropertyToID("gsRelocDstXZ");
            int dstXYID = Shader.PropertyToID("gsRelocDstXY");
            int dstYZID = Shader.PropertyToID("gsRelocDstYZ");

            bakeCompute.SetMatrix(Shader.PropertyToID("gsRelocMatrix"), relocationMatrix);
            bakeCompute.SetVector(Shader.PropertyToID("gsRelocVoxCount"),
                new Vector4(vc.x, vc.y, vc.z, 0));
            bakeCompute.SetFloat(Shader.PropertyToID("gsRelocVoxSize"), vi.VoxelSize);
            bakeCompute.SetInts(TriSizeID, res, res);

            relocKernel.Set(dstXZID, dsts[0]);
            relocKernel.Set(dstXYID, dsts[1]);
            relocKernel.Set(dstYZID, dsts[2]);

            for (int srcFace = 0; srcFace < 3; srcFace++)
            {
                bakeCompute.SetTexture(relocKernel.KernelIndex, srcTriID, srcTris[srcFace]);
                bakeCompute.SetTexture(relocKernel.KernelIndex, srcDepthID, srcDepths[srcFace]);
                bakeCompute.SetInt(Shader.PropertyToID("gsRelocSrcFace"), srcFace);
                relocKernel.DispatchFit(res, res);
            }

            // Clean up source Texture2Ds
            for (int i = 0; i < 3; i++)
            {
                Destroy(srcTris[i]);
                Destroy(srcDepths[i]);
            }

            // Compute dilation: ping-pong between dsts and tmps.
            var dilateKernel = new ComputeKernelHelper(bakeCompute, "DilateTriplanar");
            int dilateSrcID = Shader.PropertyToID("gsDilateSrc");
            int dilateDstID = Shader.PropertyToID("gsDilateDst");

            var tmps = new RenderTexture[] {
                CreateTriplanarRT("DilateXZ"),
                CreateTriplanarRT("DilateXY"),
                CreateTriplanarRT("DilateYZ")
            };

            const int dilationPasses = 4;
            for (int d = 0; d < dilationPasses; d++)
            {
                var src = (d % 2 == 0) ? dsts : tmps;
                var dst = (d % 2 == 0) ? tmps : dsts;
                for (int face = 0; face < 3; face++)
                {
                    dilateKernel.Set(dilateSrcID, src[face]);
                    dilateKernel.Set(dilateDstID, dst[face]);
                    dilateKernel.DispatchFit(res, res);
                }
            }

            // After even number of passes, result is back in dsts. Destroy tmps.
            for (int i = 0; i < 3; i++) Destroy(tmps[i]);

            // Swap old → new
            Destroy(_triXZ); Destroy(_triXY); Destroy(_triYZ);
            if (_depthXZ) Destroy(_depthXZ);
            if (_depthXY) Destroy(_depthXY);
            if (_depthYZ) Destroy(_depthYZ);

            _triXZ = dsts[0]; _triXY = dsts[1]; _triYZ = dsts[2];
            _depthXZ = CreateDepthRT("DepthXZ");
            _depthXY = CreateDepthRT("DepthXY");
            _depthYZ = CreateDepthRT("DepthYZ");

            Shader.SetGlobalTexture(TriXZID, _triXZ);
            Shader.SetGlobalTexture(TriXYID, _triXY);
            Shader.SetGlobalTexture(TriYZID, _triYZ);
            Shader.SetGlobalFloat(TriAvailableID, 1f);

            if (_clearKernel.Shader != null)
            {
                _clearKernel.Set(TriXZRWID, _triXZ);
                _clearKernel.Set(TriXYRWID, _triXY);
                _clearKernel.Set(TriYZRWID, _triYZ);
            }

            Debug.Log($"[RoomScan] Triplanar relocation complete (all compute) — " +
                      $"3x {res}x{res}, 3 splat + {dilationPasses}x3 dilate dispatches");
        }

        private static Texture2D LoadTex2D(string path, int w, int h, TextureFormat format)
        {
            var tex = new Texture2D(w, h, format, false, true);
            if (File.Exists(path))
            {
                byte[] data = File.ReadAllBytes(path);
                tex.LoadRawTextureData(data);
            }
            tex.Apply(false, false);
            return tex;
        }

        private int _bakeCount;
        public void DispatchBake(Texture camFrame, Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes,
            float exposure, List<Transform> exclusionZones)
        {
            if (!_kernelsReady || camFrame == null)
            {
                if (_bakeCount < 3) Debug.Log($"[RoomScan] TriBake skip: kernels={_kernelsReady}, frame={camFrame != null}");
                return;
            }

            var dc = DepthCapture.Instance;
            if (dc == null || !DepthCapture.DepthAvailable || dc.DepthTex == null)
            {
                if (_bakeCount < 3) Debug.Log($"[RoomScan] TriBake skip: dc={dc != null}, depthAvail={DepthCapture.DepthAvailable}, depthTex={dc?.DepthTex != null}");
                return;
            }

            EnsureCamCopy(camFrame.width, camFrame.height);
            Graphics.Blit(camFrame, _camFrameCopy);

            bakeCompute.SetMatrixArray(DepthCapture.ViewID, dc.View);
            bakeCompute.SetMatrixArray(DepthCapture.ProjID, dc.Proj);
            bakeCompute.SetMatrixArray(DepthCapture.ViewInvID, dc.ViewInv);
            bakeCompute.SetMatrixArray(DepthCapture.ProjInvID, dc.ProjInv);
            bakeCompute.SetVector(DepthCapture.TexSizeID,
                new Vector2(dc.DepthTex.width, dc.DepthTex.height));

            bakeCompute.SetTexture(_bakeKernel.KernelIndex, CamRGBID, _camFrameCopy);
            bakeCompute.SetVector(CamPosID, camPos);
            bakeCompute.SetMatrix(CamInvRotID, Matrix4x4.Rotate(Quaternion.Inverse(camRot)));
            bakeCompute.SetVector(CamFocalLenID, focalLen);
            bakeCompute.SetVector(CamPrincipalPtID, principalPt);
            bakeCompute.SetVector(CamSensorResID, sensorRes);
            bakeCompute.SetVector(CamCurrentResID, currentRes);
            bakeCompute.SetFloat(CamExposureID, exposure);

            var vi = VolumeIntegrator.Instance;
            if (vi != null)
            {
                bakeCompute.SetInts(Shader.PropertyToID("gsVoxCount"),
                    vi.VoxelCount.x, vi.VoxelCount.y, vi.VoxelCount.z);
                bakeCompute.SetFloat(Shader.PropertyToID("gsVoxSize"), vi.VoxelSize);
                bakeCompute.SetFloat(MaxUpdateDistID, 5f);
            }

            var excPositions = new Vector4[64];
            int numExc = exclusionZones != null ? Mathf.Min(exclusionZones.Count, 64) : 0;
            for (int i = 0; i < numExc; i++)
                if (exclusionZones[i] != null)
                    excPositions[i] = exclusionZones[i].position;
            bakeCompute.SetInt(Shader.PropertyToID("gsNumExclusions"), numExc);
            bakeCompute.SetVectorArray(Shader.PropertyToID("gsExclusionHeads"), excPositions);

            bakeCompute.SetInts(TriSizeID, textureResolution, textureResolution);
            int splatR = splatRadiusOverride > 0 ? splatRadiusOverride : Mathf.Max(1, textureResolution / 2048);
            bakeCompute.SetInt(Shader.PropertyToID("gsSplatRadius"), splatR);
            _bakeKernel.Set(TriXZRWID, _triXZ);
            _bakeKernel.Set(TriXYRWID, _triXY);
            _bakeKernel.Set(TriYZRWID, _triYZ);
            _bakeKernel.Set(DepthXZRWID, _depthXZ);
            _bakeKernel.Set(DepthXYRWID, _depthXY);
            _bakeKernel.Set(DepthYZRWID, _depthYZ);

            _bakeKernel.Set(DepthCapture.DepthTexID, dc.DepthTex);
            _bakeKernel.Set(DepthCapture.NormTexID, dc.NormTex);

            _bakeKernel.DispatchFit(dc.DepthTex.width, dc.DepthTex.height);

            Shader.SetGlobalFloat(TriAvailableID, 1f);

            _bakeCount++;
            if (_bakeCount <= 3 || _bakeCount % 100 == 0)
                Debug.Log($"[RoomScan] TriBake #{_bakeCount}: depth={dc.DepthTex.width}x{dc.DepthTex.height}, " +
                    $"cam={camFrame.width}x{camFrame.height}, triAvail=1");
        }

        private void EnsureCamCopy(int w, int h)
        {
            if (_camFrameCopy != null && _camFrameCopy.width == w && _camFrameCopy.height == h) return;
            if (_camFrameCopy) Destroy(_camFrameCopy);
            _camFrameCopy = new RenderTexture(w, h, 0, GraphicsFormat.R8G8B8A8_SRGB)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _camFrameCopy.Create();
        }

        public void Save(string directory)
        {
            if (!Directory.Exists(directory)) Directory.CreateDirectory(directory);
            SaveRT(_triXZ, Path.Combine(directory, "tri_xz.raw"));
            SaveRT(_triXY, Path.Combine(directory, "tri_xy.raw"));
            SaveRT(_triYZ, Path.Combine(directory, "tri_yz.raw"));
            SaveDepthRT(_depthXZ, Path.Combine(directory, "depth_xz.raw"));
            SaveDepthRT(_depthXY, Path.Combine(directory, "depth_xy.raw"));
            SaveDepthRT(_depthYZ, Path.Combine(directory, "depth_yz.raw"));
            Debug.Log($"[RoomScan] Triplanar textures + depth saved to {directory}");
        }

        /// <summary>
        /// Reads triplanar pixel data on main thread (required for ReadPixels),
        /// returns raw bytes for background file I/O.
        /// </summary>
        public (byte[] xz, byte[] xy, byte[] yz,
                byte[] dxz, byte[] dxy, byte[] dyz) ReadRawBytes()
        {
            return (ReadRTBytes(_triXZ), ReadRTBytes(_triXY), ReadRTBytes(_triYZ),
                    ReadDepthRTBytes(_depthXZ), ReadDepthRTBytes(_depthXY), ReadDepthRTBytes(_depthYZ));
        }

        public static byte[] ReadRTBytes(RenderTexture rt)
        {
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false, true);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = prev;
            byte[] data = tex.GetRawTextureData();
            Destroy(tex);
            return data;
        }

        public static byte[] ReadDepthRTBytes(RenderTexture rt)
        {
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.R8, false, true);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = prev;
            byte[] data = tex.GetRawTextureData();
            Destroy(tex);
            return data;
        }

        public void Load(string directory)
        {
            LoadRT(_triXZ, Path.Combine(directory, "tri_xz.raw"));
            LoadRT(_triXY, Path.Combine(directory, "tri_xy.raw"));
            LoadRT(_triYZ, Path.Combine(directory, "tri_yz.raw"));
            UpdateShaderGlobals();
            Debug.Log($"[RoomScan] Triplanar textures loaded from {directory}");
        }

        public void LoadDepth(string directory)
        {
            LoadDepthRT(_depthXZ, Path.Combine(directory, "depth_xz.raw"));
            LoadDepthRT(_depthXY, Path.Combine(directory, "depth_xy.raw"));
            LoadDepthRT(_depthYZ, Path.Combine(directory, "depth_yz.raw"));
            Debug.Log($"[RoomScan] Triplanar depth textures loaded from {directory}");
        }

        private static void SaveRT(RenderTexture rt, string path)
        {
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = prev;

            byte[] data = tex.GetRawTextureData();
            File.WriteAllBytes(path, data);
            Destroy(tex);
        }

        private static void SaveDepthRT(RenderTexture rt, string path)
        {
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.R8, false, true);
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = prev;

            byte[] data = tex.GetRawTextureData();
            File.WriteAllBytes(path, data);
            Destroy(tex);
        }

        private static void LoadRT(RenderTexture rt, string path)
        {
            if (!File.Exists(path)) return;
            byte[] data = File.ReadAllBytes(path);
            int expected = rt.width * rt.height * 4;
            if (data.Length != expected)
            {
                Debug.LogWarning($"[RoomScan] Triplanar {path}: size {data.Length} != expected {expected} " +
                    $"(RT {rt.width}x{rt.height}). Skipping plane.");
                return;
            }

            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false, true);
            tex.LoadRawTextureData(data);
            tex.Apply(false, false);
            Graphics.Blit(tex, rt);
            Destroy(tex);
        }

        private static void LoadDepthRT(RenderTexture rt, string path)
        {
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[RoomScan] Depth texture not found: {path}");
                return;
            }
            byte[] data = File.ReadAllBytes(path);
            int expected = rt.width * rt.height * 1;
            if (data.Length != expected)
            {
                Debug.LogWarning($"[RoomScan] Depth {path}: size {data.Length} != expected {expected}. Skipping.");
                return;
            }

            var tex = new Texture2D(rt.width, rt.height, TextureFormat.R8, false, true);
            tex.LoadRawTextureData(data);
            tex.Apply(false, false);
            Graphics.Blit(tex, rt);
            Destroy(tex);
        }
    }
}
