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
        private RenderTexture _camFrameCopy;
        private ComputeKernelHelper _bakeKernel;
        private ComputeKernelHelper _clearKernel;
        private bool _kernelsReady;

        public RenderTexture TriXZ => _triXZ;
        public RenderTexture TriXY => _triXY;
        public RenderTexture TriYZ => _triYZ;

        static readonly int TriXZID = Shader.PropertyToID("_RSTriXZ");
        static readonly int TriXYID = Shader.PropertyToID("_RSTriXY");
        static readonly int TriYZID = Shader.PropertyToID("_RSTriYZ");
        static readonly int TriAvailableID = Shader.PropertyToID("_RSTriAvailable");
        static readonly int OldTriXZID = Shader.PropertyToID("gsOldTriXZ");
        static readonly int OldTriXYID = Shader.PropertyToID("gsOldTriXY");
        static readonly int OldTriYZID = Shader.PropertyToID("gsOldTriYZ");
        static readonly int DstTriRWID = Shader.PropertyToID("gsDstTri_RW");
        static readonly int BakeTriInvRelocID = Shader.PropertyToID("gsBakeTriInvReloc");
        static readonly int BakeTriFaceID = Shader.PropertyToID("gsBakeTriFace");
        static readonly int BakeVoxCountFID = Shader.PropertyToID("gsBakeVoxCountF");
        static readonly int BakeVoxSizeID = Shader.PropertyToID("gsBakeVoxSize");

        static readonly int TriXZRWID = Shader.PropertyToID("gsTriXZ_RW");
        static readonly int TriXYRWID = Shader.PropertyToID("gsTriXY_RW");
        static readonly int TriYZRWID = Shader.PropertyToID("gsTriYZ_RW");
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
            if (_camFrameCopy) Destroy(_camFrameCopy);
        }

        private void CreateTextures()
        {
            _triXZ = CreateTriplanarRT("TriXZ");
            _triXY = CreateTriplanarRT("TriXY");
            _triYZ = CreateTriplanarRT("TriYZ");
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

        public void Clear()
        {
            if (!_kernelsReady) return;
            _clearKernel.Set(TriXZRWID, _triXZ);
            _clearKernel.Set(TriXYRWID, _triXY);
            _clearKernel.Set(TriYZRWID, _triYZ);
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
        /// Resample triplanar textures from old coordinate frame into new (identity) frame.
        /// Loads raw triplanar files from disk as Texture2D (always SRV-readable on Vulkan)
        /// to avoid the Blit→compute SRV layout transition bug on Quest.
        /// </summary>
        public void BakeRelocation(Matrix4x4 relocationMatrix, string triplanarDir)
        {
            if (!_kernelsReady || bakeCompute == null) return;

            var vi = VolumeIntegrator.Instance;
            if (vi == null) return;

            Matrix4x4 invReloc = relocationMatrix.inverse;
            int res = textureResolution;

            var srcXZ = LoadTex2D(Path.Combine(triplanarDir, "tri_xz.raw"), res);
            var srcXY = LoadTex2D(Path.Combine(triplanarDir, "tri_xy.raw"), res);
            var srcYZ = LoadTex2D(Path.Combine(triplanarDir, "tri_yz.raw"), res);

            var dstXZ = CreateTriplanarRT("DstXZ");
            var dstXY = CreateTriplanarRT("DstXY");
            var dstYZ = CreateTriplanarRT("DstYZ");
            var dstArr = new[] { dstXZ, dstXY, dstYZ };

            int kernel = bakeCompute.FindKernel("BakeTriplanarRelocation");

            var vc = vi.VoxelCount;
            bakeCompute.SetVector(BakeVoxCountFID, new Vector4(vc.x, vc.y, vc.z, 0));
            bakeCompute.SetFloat(BakeVoxSizeID, vi.VoxelSize);
            bakeCompute.SetInts(TriSizeID, res, res);
            bakeCompute.SetMatrix(BakeTriInvRelocID, invReloc);

            bakeCompute.SetTexture(kernel, OldTriXZID, srcXZ);
            bakeCompute.SetTexture(kernel, OldTriXYID, srcXY);
            bakeCompute.SetTexture(kernel, OldTriYZID, srcYZ);

            int groupsX = Mathf.CeilToInt(res / 8f);
            int groupsY = Mathf.CeilToInt(res / 8f);

            for (int face = 0; face < 3; face++)
            {
                bakeCompute.SetInt(BakeTriFaceID, face);
                bakeCompute.SetTexture(kernel, DstTriRWID, dstArr[face]);
                bakeCompute.Dispatch(kernel, groupsX, groupsY, 1);
            }
            GL.Flush();

            Destroy(srcXZ);
            Destroy(srcXY);
            Destroy(srcYZ);

            Destroy(_triXZ);
            Destroy(_triXY);
            Destroy(_triYZ);
            _triXZ = dstXZ;
            _triXY = dstXY;
            _triYZ = dstYZ;

            Shader.SetGlobalTexture(TriXZID, _triXZ);
            Shader.SetGlobalTexture(TriXYID, _triXY);
            Shader.SetGlobalTexture(TriYZID, _triYZ);
            Shader.SetGlobalFloat(TriAvailableID, 1f);

            _clearKernel.Set(TriXZRWID, _triXZ);
            _clearKernel.Set(TriXYRWID, _triXY);
            _clearKernel.Set(TriYZRWID, _triYZ);

            Debug.Log($"[RoomScan] Triplanar bake relocation complete (from disk) — " +
                      $"3x {res}x{res} texels, invReloc col3={invReloc.GetColumn(3)}");
        }

        private static Texture2D LoadTex2D(string path, int res)
        {
            byte[] data = File.ReadAllBytes(path);
            var tex = new Texture2D(res, res, TextureFormat.RGBA32, false, true);
            tex.LoadRawTextureData(data);
            tex.Apply(false, true);
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
            Debug.Log($"[RoomScan] Triplanar textures saved to {directory}");
        }

        /// <summary>
        /// Reads triplanar pixel data on main thread (required for ReadPixels),
        /// returns raw bytes for background file I/O.
        /// </summary>
        public (byte[] xz, byte[] xy, byte[] yz) ReadRawBytes()
        {
            return (ReadRTBytes(_triXZ), ReadRTBytes(_triXY), ReadRTBytes(_triYZ));
        }

        public static byte[] ReadRTBytes(RenderTexture rt)
        {
            // linear=true: bytes match UNorm RT / LoadRaw round-trip (sRGB Texture2D darkens on Blit).
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

        public void Load(string directory)
        {
            LoadRT(_triXZ, Path.Combine(directory, "tri_xz.raw"));
            LoadRT(_triXY, Path.Combine(directory, "tri_xy.raw"));
            LoadRT(_triYZ, Path.Combine(directory, "tri_yz.raw"));
            UpdateShaderGlobals();
            Debug.Log($"[RoomScan] Triplanar textures loaded from {directory}");
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
    }
}
