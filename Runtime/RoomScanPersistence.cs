using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    // ─────────────────────────────────────────────────────────────
    //  Package data model (serialized to manifest.json / anchor.json)
    // ─────────────────────────────────────────────────────────────

    [Serializable]
    public class ScanPackageEntry
    {
        public string id;
        public string displayName;
        public long timestamp;
        public string anchorUuid;
        public bool hasSplat;
        public bool hasRefined;
        public bool hasHQRefined;
        public bool hasKeyframes;
    }

    [Serializable]
    public class ScanPackageManifest
    {
        public int version = 1;
        public List<ScanPackageEntry> packages = new();
    }

    public enum ArtifactType { Splat, Refined, HQRefined }

    /// <summary>
    /// Per-artifact anchor matrices. All matrices are from localizing the same
    /// OVRSpatialAnchor in different sessions. Each artifact tracks the anchor's
    /// localToWorldMatrix at the time it was created / saved to disk.
    /// </summary>
    [Serializable]
    public class PackageAnchorData
    {
        public string anchorUuid;
        public float[] baseMatrixAtSave;
        public float[] splatMatrixAtCreate;
        public float[] refinedMatrixAtCreate;
        public float[] hqMatrixAtCreate;
    }

    // ─────────────────────────────────────────────────────────────

    public class RoomScanPersistence : MonoBehaviour
    {
        public static RoomScanPersistence Instance { get; private set; }

        private const uint Magic = 0x48534D52; // "RMSH"
        private const int FormatVersion = 1;
        private const uint RefinedMeshMagic = 0x46524D52; // "RMRF"
        private const int RefinedMeshVersion = 1;

        public bool IsSaving { get; private set; }
        public bool IsLoading { get; private set; }

        public event Action SaveCompleted;
        public event Action LoadCompleted;

        /// <summary>Currently active package ID (set after save or load).</summary>
        public string ActivePackageId { get; private set; }

        /// <summary>Whether there is an active package that artifacts can be saved to.</summary>
        public bool HasActivePackage => !string.IsNullOrEmpty(ActivePackageId);

        private string RoomScansRoot => Path.Combine(Application.persistentDataPath, "RoomScans");
        private string ManifestPath => Path.Combine(RoomScansRoot, "manifest.json");

        public string ActivePackageDirectory =>
            HasActivePackage ? Path.Combine(RoomScansRoot, ActivePackageId) : null;

        // Paths relative to active package
        private string PkgScanBin => Path.Combine(ActivePackageDirectory, "scan.bin");
        private string PkgAnchorJson => Path.Combine(ActivePackageDirectory, "anchor.json");
        private string PkgTriplanarDir => Path.Combine(ActivePackageDirectory, "triplanar");
        private string PkgKeyframesDir => Path.Combine(ActivePackageDirectory, "keyframes");
        private string PkgSplatPath => Path.Combine(ActivePackageDirectory, "splat.ply");
        private string PkgRefinedMeshPath => Path.Combine(ActivePackageDirectory, "refined_mesh.bin");
        private string PkgRefinedAtlasPath => Path.Combine(ActivePackageDirectory, "refined_atlas.raw");
        private string PkgHQAtlasPath => Path.Combine(ActivePackageDirectory, "hq_atlas.raw");

        // Legacy single-slot paths (for backward compat references in RoomScanner.ClearAllDataAsync)
        [Obsolete("Use package-based paths")] public string SaveFilePath => PkgScanBin;
        [Obsolete("Use package-based paths")] public string TriplanarDirectory => PkgTriplanarDir;
        [Obsolete("Use package-based paths")] public string SplatFilePath => PkgSplatPath;

        private PackageAnchorData _activeAnchorData;

        private void Awake() => Instance = this;

        // ─────────────────────────────────────────────────────────────
        //  Manifest I/O
        // ─────────────────────────────────────────────────────────────

        public ScanPackageManifest ReadManifest()
        {
            if (!File.Exists(ManifestPath))
                return new ScanPackageManifest();
            try
            {
                string json = File.ReadAllText(ManifestPath);
                return JsonUtility.FromJson<ScanPackageManifest>(json) ?? new ScanPackageManifest();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Persistence] Failed to read manifest: {e.Message}");
                return new ScanPackageManifest();
            }
        }

        private void WriteManifest(ScanPackageManifest manifest)
        {
            if (!Directory.Exists(RoomScansRoot))
                Directory.CreateDirectory(RoomScansRoot);
            File.WriteAllText(ManifestPath, JsonUtility.ToJson(manifest, true));
        }

        public List<ScanPackageEntry> ListPackages()
        {
            var manifest = ReadManifest();
            manifest.packages.Sort((a, b) => b.timestamp.CompareTo(a.timestamp));
            return manifest.packages;
        }

        public bool HasAnyPackage()
        {
            var manifest = ReadManifest();
            return manifest.packages.Count > 0;
        }

        // ─────────────────────────────────────────────────────────────
        //  anchor.json I/O
        // ─────────────────────────────────────────────────────────────

        private static PackageAnchorData ReadAnchorData(string anchorJsonPath)
        {
            if (!File.Exists(anchorJsonPath)) return null;
            string json = File.ReadAllText(anchorJsonPath);
            return JsonUtility.FromJson<PackageAnchorData>(json);
        }

        private static void WriteAnchorData(string anchorJsonPath, PackageAnchorData data)
        {
            File.WriteAllText(anchorJsonPath, JsonUtility.ToJson(data, true));
        }

        private static float[] MatrixToFloats(Matrix4x4 m)
        {
            return new[]
            {
                m.m00, m.m01, m.m02, m.m03,
                m.m10, m.m11, m.m12, m.m13,
                m.m20, m.m21, m.m22, m.m23,
                m.m30, m.m31, m.m32, m.m33
            };
        }

        private static Matrix4x4 FloatsToMatrix(float[] f)
        {
            if (f == null || f.Length < 16) return Matrix4x4.identity;
            return new Matrix4x4(
                new Vector4(f[0], f[4], f[8], f[12]),
                new Vector4(f[1], f[5], f[9], f[13]),
                new Vector4(f[2], f[6], f[10], f[14]),
                new Vector4(f[3], f[7], f[11], f[15]));
        }

        // ─────────────────────────────────────────────────────────────
        //  Save — creates a new package with base scan data
        // ─────────────────────────────────────────────────────────────

        public async Task<bool> SaveToNewPackageAsync()
        {
            var vi = VolumeIntegrator.Instance;
            if (vi == null || vi.Volume == null)
            {
                Debug.LogWarning("[Persistence] Cannot save, no volume");
                return false;
            }
            if (IsSaving) { Debug.LogWarning("[Persistence] Save already in progress"); return false; }

            IsSaving = true;
            try
            {
                string pkgId = $"pkg_{DateTimeOffset.UtcNow:yyyyMMdd_HHmmss}";
                string pkgDir = Path.Combine(RoomScansRoot, pkgId);
                Directory.CreateDirectory(pkgDir);

                int3 s = vi.VoxelCount;

                // GPU readback TSDF
                var tsdfReq = await AsyncGPUReadback.RequestAsync(vi.Volume, 0);
                if (tsdfReq.hasError) { Debug.LogError("[Persistence] TSDF readback failed"); return false; }
                byte[] tsdfBytes = new byte[s.x * s.y * s.z * 2];
                for (int z = 0; z < s.z; z++)
                {
                    var slice = tsdfReq.GetData<byte>(z);
                    Unity.Collections.NativeArray<byte>.Copy(slice, 0, tsdfBytes, z * s.x * s.y * 2, s.x * s.y * 2);
                }

                // GPU readback color
                var colorReq = await AsyncGPUReadback.RequestAsync(vi.ColorVolume, 0);
                if (colorReq.hasError) { Debug.LogError("[Persistence] Color readback failed"); return false; }
                byte[] colorBytes = new byte[s.x * s.y * s.z * 4];
                for (int z = 0; z < s.z; z++)
                {
                    var slice = colorReq.GetData<byte>(z);
                    Unity.Collections.NativeArray<byte>.Copy(slice, 0, colorBytes, z * s.x * s.y * 4, s.x * s.y * 4);
                }

                var tc = TriplanarCache.Instance;
                int triRes = tc != null && tc.TriXZ != null ? tc.TriXZ.width : 0;

                // Create spatial anchor at MRUK floor position
                Matrix4x4 anchorMatrix = Matrix4x4.identity;
                string anchorUuidStr = "";
                var roomAnchor = RoomAnchorManager.Instance;

                if (roomAnchor != null && roomAnchor.enabled && roomAnchor.IsRoomLoaded)
                {
                    var result = await roomAnchor.CreateAndSaveSpatialAnchorAsync(
                        default, Quaternion.identity);
                    if (result.HasValue)
                    {
                        anchorUuidStr = result.Value.uuid.ToString();
                        anchorMatrix = result.Value.matrix;
                        Debug.Log($"[Persistence] Spatial anchor created for package: {anchorUuidStr}");
                    }
                    else
                    {
                        anchorMatrix = roomAnchor.GetRoomLocalToWorldForPersistence();
                        Debug.LogWarning("[Persistence] Spatial anchor creation failed, using MRUK fallback matrix");
                    }
                }

                // Write scan.bin (same v1 format for backward compat)
                string scanBinPath = Path.Combine(pkgDir, "scan.bin");
                await Task.Run(() => WriteBinary(scanBinPath, s, vi.VoxelSize,
                    vi.IntegrationCount, tsdfBytes, colorBytes, triRes, anchorMatrix));
                tsdfBytes = null;
                colorBytes = null;

                // Write anchor.json
                var anchorData = new PackageAnchorData
                {
                    anchorUuid = anchorUuidStr,
                    baseMatrixAtSave = MatrixToFloats(anchorMatrix)
                };
                string anchorJsonPath = Path.Combine(pkgDir, "anchor.json");
                await Task.Run(() => WriteAnchorData(anchorJsonPath, anchorData));

                // Save triplanar
                if (tc != null && triRes > 0)
                {
                    string triDir = Path.Combine(pkgDir, "triplanar");
                    Directory.CreateDirectory(triDir);
                    await SaveTriplanarOneAtATime(tc, triDir);
                }

                // Copy keyframes from GSExport/
                string gsExportDir = Path.Combine(Application.persistentDataPath, "GSExport");
                string kfDir = Path.Combine(pkgDir, "keyframes");
                if (Directory.Exists(gsExportDir))
                {
                    await Task.Run(() => CopyDirectoryContents(gsExportDir, kfDir));
                    Debug.Log("[Persistence] Keyframes copied to package");
                }

                // Update manifest
                var manifest = ReadManifest();
                bool hasKf = Directory.Exists(kfDir) &&
                    Directory.GetFiles(kfDir, "*.jpg", SearchOption.AllDirectories).Length > 0;
                manifest.packages.Add(new ScanPackageEntry
                {
                    id = pkgId,
                    displayName = $"Scan {DateTime.Now:MMM dd HH:mm}",
                    timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    anchorUuid = anchorUuidStr,
                    hasKeyframes = hasKf
                });
                WriteManifest(manifest);

                ActivePackageId = pkgId;
                _activeAnchorData = anchorData;

                float sizeMB = new FileInfo(scanBinPath).Length / (1024f * 1024f);
                Debug.Log($"[Persistence] Package saved: {pkgId} ({sizeMB:F1}MB), " +
                          $"triplanar={triRes > 0}, keyframes={hasKf}, anchor={anchorUuidStr}");
                SaveCompleted?.Invoke();
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[Persistence] Save failed: {e.Message}\n{e.StackTrace}");
                return false;
            }
            finally { IsSaving = false; }
        }

        // ─────────────────────────────────────────────────────────────
        //  Save artifact to active package
        // ─────────────────────────────────────────────────────────────

        public async Task<bool> SaveArtifactAsync(ArtifactType type, byte[] data,
            RefinedTextureResult? refinedResult = null)
        {
            if (!HasActivePackage)
            {
                Debug.LogWarning("[Persistence] No active package — artifact stays in memory only");
                return false;
            }

            try
            {
                string pkgDir = ActivePackageDirectory;
                Matrix4x4 currentMatrix = GetCurrentAnchorMatrix();

                switch (type)
                {
                    case ArtifactType.Splat:
                        string splatPath = Path.Combine(pkgDir, "splat.ply");
                        await Task.Run(() => File.WriteAllBytes(splatPath, data));
                        if (_activeAnchorData != null)
                            _activeAnchorData.splatMatrixAtCreate = MatrixToFloats(currentMatrix);
                        Debug.Log($"[Persistence] Splat auto-saved ({data.Length / (1024f * 1024f):F1}MB)");
                        break;

                    case ArtifactType.Refined:
                        if (refinedResult.HasValue)
                        {
                            var r = refinedResult.Value;
                            string meshPath = Path.Combine(pkgDir, "refined_mesh.bin");
                            await Task.Run(() => WriteRefinedMesh(meshPath, r));
                            if (r.AtlasPixels != null)
                            {
                                string atlasPath = Path.Combine(pkgDir, "refined_atlas.raw");
                                await Task.Run(() => File.WriteAllBytes(atlasPath, r.AtlasPixels));
                            }
                        }
                        if (_activeAnchorData != null)
                            _activeAnchorData.refinedMatrixAtCreate = MatrixToFloats(currentMatrix);
                        Debug.Log("[Persistence] Refined texture auto-saved");
                        break;

                    case ArtifactType.HQRefined:
                        string hqPath = Path.Combine(pkgDir, "hq_atlas.raw");
                        await Task.Run(() => File.WriteAllBytes(hqPath, data));
                        if (_activeAnchorData != null)
                            _activeAnchorData.hqMatrixAtCreate = MatrixToFloats(currentMatrix);
                        Debug.Log("[Persistence] HQ atlas auto-saved");
                        break;
                }

                // Persist updated anchor.json with the new artifact matrix
                if (_activeAnchorData != null)
                    await Task.Run(() => WriteAnchorData(
                        Path.Combine(pkgDir, "anchor.json"), _activeAnchorData));

                // Update manifest flags
                UpdateManifestFlags(ActivePackageId, type, true);
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[Persistence] Artifact save failed: {e.Message}\n{e.StackTrace}");
                return false;
            }
        }

        /// <summary>
        /// Deletes a specific artifact from the active package on disk and in memory.
        /// </summary>
        public void DeleteArtifactFromPackage(ArtifactType type)
        {
            if (!HasActivePackage) return;
            string pkgDir = ActivePackageDirectory;

            switch (type)
            {
                case ArtifactType.Splat:
                    TryDeleteFile(Path.Combine(pkgDir, "splat.ply"));
                    if (_activeAnchorData != null) _activeAnchorData.splatMatrixAtCreate = null;
                    break;
                case ArtifactType.Refined:
                    TryDeleteFile(Path.Combine(pkgDir, "refined_mesh.bin"));
                    TryDeleteFile(Path.Combine(pkgDir, "refined_atlas.raw"));
                    if (_activeAnchorData != null) _activeAnchorData.refinedMatrixAtCreate = null;
                    break;
                case ArtifactType.HQRefined:
                    TryDeleteFile(Path.Combine(pkgDir, "hq_atlas.raw"));
                    if (_activeAnchorData != null) _activeAnchorData.hqMatrixAtCreate = null;
                    break;
            }

            if (_activeAnchorData != null)
                WriteAnchorData(Path.Combine(pkgDir, "anchor.json"), _activeAnchorData);
            UpdateManifestFlags(ActivePackageId, type, false);
            Debug.Log($"[Persistence] Artifact {type} deleted from {ActivePackageId}");
        }

        // ─────────────────────────────────────────────────────────────
        //  Load package
        // ─────────────────────────────────────────────────────────────

        public async Task<bool> LoadPackageAsync(string pkgId)
        {
            string pkgDir = Path.Combine(RoomScansRoot, pkgId);
            string scanBinPath = Path.Combine(pkgDir, "scan.bin");
            if (!File.Exists(scanBinPath))
            {
                Debug.LogWarning($"[Persistence] Package {pkgId} has no scan.bin");
                return false;
            }

            var vi = VolumeIntegrator.Instance;
            var cm = MeshExtractor.Instance;
            if (vi == null || vi.Volume == null)
            {
                Debug.LogWarning("[Persistence] Cannot load, no volume");
                return false;
            }
            if (IsLoading) { Debug.LogWarning("[Persistence] Load already in progress"); return false; }

            IsLoading = true;
            try
            {
                var unitySync = SynchronizationContext.Current;
                string triplanarDir = Path.Combine(pkgDir, "triplanar");
                string anchorJsonPath = Path.Combine(pkgDir, "anchor.json");

                // Read scan.bin + anchor.json in background
                byte[] tsdfBytes = null, colorBytes = null;
                int3 savedVoxCount = default;
                float savedVoxSize = 0;
                int savedIntCount = 0, triRes = 0;
                Matrix4x4 anchorAtSave = Matrix4x4.identity;
                PackageAnchorData anchorData = null;

                await Task.Run(() =>
                {
                    ReadBinary(scanBinPath, out savedVoxCount, out savedVoxSize, out savedIntCount,
                        out tsdfBytes, out colorBytes, out triRes, out anchorAtSave);
                    anchorData = ReadAnchorData(anchorJsonPath);
                });

                await SwitchToUnityMainThreadAsync(unitySync);

                // Validate voxel grid compatibility
                int3 currentVox = vi.VoxelCount;
                if (math.any(savedVoxCount != currentVox) ||
                    Mathf.Abs(savedVoxSize - vi.VoxelSize) > 0.001f)
                {
                    Debug.LogWarning($"[Persistence] Voxel mismatch saved={savedVoxCount}/{savedVoxSize} " +
                                     $"current={currentVox}/{vi.VoxelSize}");
                    return false;
                }

                if (!vi.LoadVolumes(tsdfBytes, colorBytes, savedIntCount))
                    return false;

                // Resolve base matrix — prefer anchor.json, fall back to scan.bin matrix
                Matrix4x4 baseMatrix = anchorData?.baseMatrixAtSave != null
                    ? FloatsToMatrix(anchorData.baseMatrixAtSave)
                    : anchorAtSave;

                // Try to load spatial anchor for relocation
                Matrix4x4 anchorNow = Matrix4x4.identity;
                bool spatialAnchorOk = false;
                var roomAnchor = RoomAnchorManager.Instance;

                if (anchorData != null && !string.IsNullOrEmpty(anchorData.anchorUuid) &&
                    Guid.TryParse(anchorData.anchorUuid, out Guid uuid))
                {
                    var loadedMatrix = await roomAnchor.LoadSpatialAnchorAsync(uuid);
                    await SwitchToUnityMainThreadAsync(unitySync);
                    if (loadedMatrix.HasValue)
                    {
                        anchorNow = loadedMatrix.Value;
                        spatialAnchorOk = true;
                        Debug.Log("[Persistence] Spatial anchor localized for relocation");
                    }
                }

                // Fallback to MRUK if spatial anchor failed
                if (!spatialAnchorOk && roomAnchor != null && roomAnchor.enabled)
                {
                    if (!roomAnchor.IsRoomLoaded)
                    {
                        Debug.Log("[Persistence] Waiting for MRUK room...");
                        for (int i = 0; i < 300 && !roomAnchor.IsRoomLoaded; i++)
                        {
                            await Task.Delay(16);
                            await SwitchToUnityMainThreadAsync(unitySync);
                        }
                    }
                    if (roomAnchor.IsRoomLoaded)
                    {
                        // Stabilize MRUK anchor
                        Matrix4x4 prev = roomAnchor.GetRoomLocalToWorldForPersistence();
                        int stable = 0;
                        for (int i = 0; i < 60 && stable < 5; i++)
                        {
                            await Task.Delay(16);
                            await SwitchToUnityMainThreadAsync(unitySync);
                            Matrix4x4 cur = roomAnchor.GetRoomLocalToWorldForPersistence();
                            float d = Vector3.Distance(
                                new Vector3(prev.m03, prev.m13, prev.m23),
                                new Vector3(cur.m03, cur.m13, cur.m23));
                            stable = d < 0.001f ? stable + 1 : 0;
                            prev = cur;
                        }
                        anchorNow = roomAnchor.GetRoomLocalToWorldForPersistence();
                        Debug.LogWarning("[Persistence] Using MRUK fallback for relocation");
                    }
                }

                // Compute per-artifact relocation matrices
                Matrix4x4 relocVolume = RoomAnchorManager.ComputeRelocationMatrix(anchorNow, baseMatrix);

                Matrix4x4 splatMatrix = anchorData?.splatMatrixAtCreate != null
                    ? FloatsToMatrix(anchorData.splatMatrixAtCreate) : baseMatrix;
                Matrix4x4 relocSplat = RoomAnchorManager.ComputeRelocationMatrix(anchorNow, splatMatrix);

                Matrix4x4 refinedMatrix = anchorData?.refinedMatrixAtCreate != null
                    ? FloatsToMatrix(anchorData.refinedMatrixAtCreate) : baseMatrix;
                Matrix4x4 relocRefined = RoomAnchorManager.ComputeRelocationMatrix(anchorNow, refinedMatrix);

                Matrix4x4 hqMatrix = anchorData?.hqMatrixAtCreate != null
                    ? FloatsToMatrix(anchorData.hqMatrixAtCreate) : refinedMatrix;
                Matrix4x4 relocHQ = RoomAnchorManager.ComputeRelocationMatrix(anchorNow, hqMatrix);

                var scanner = RoomScanner.Instance;
                if (scanner != null)
                    scanner.KeyframeRelocation = relocVolume;

                // BakeRelocation on TSDF volume
                if (relocVolume != Matrix4x4.identity)
                {
                    vi.BakeRelocation(relocVolume);
                    Debug.Log("[Persistence] TSDF bake-relocated");
                }

                // Triplanar
                var tc = TriplanarCache.Instance;
                if (tc != null && triRes > 0 && Directory.Exists(triplanarDir))
                {
                    try
                    {
                        if (relocVolume != Matrix4x4.identity)
                            tc.BakeRelocation(relocVolume, triplanarDir);
                        else
                        {
                            tc.Load(triplanarDir);
                            tc.LoadDepth(triplanarDir);
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[Persistence] Triplanar load skipped ({e.Message})");
                    }
                }

                // Mesh extraction
                if (cm != null)
                {
                    cm.Reinitialize();
                    cm.Extract();
                }

                // Load splat
                string splatPath = Path.Combine(pkgDir, "splat.ply");
                if (File.Exists(splatPath))
                {
                    try
                    {
                        byte[] plyBytes = null;
                        await Task.Run(() => plyBytes = File.ReadAllBytes(splatPath));
                        await SwitchToUnityMainThreadAsync(unitySync);

                        var gm = scanner?.GetComponent<GSplat.GSplatManager>();
                        if (gm != null && plyBytes != null && plyBytes.Length > 0)
                        {
                            gm.LoadTrainedPly(plyBytes);
                            gm.RenderVisible = scanner.CurrentRenderMode == ScanRenderMode.Splat;
                            scanner.DownloadedPlyData = plyBytes;

                            if (relocSplat != Matrix4x4.identity && gm.SplatHolder != null)
                            {
                                gm.SplatHolder.SetPositionAndRotation(
                                    new Vector3(relocSplat.m03, relocSplat.m13, relocSplat.m23),
                                    relocSplat.rotation);
                            }
                            else
                                gm.ResetSplatTransform();

                            Debug.Log($"[Persistence] Splat loaded ({plyBytes.Length / (1024f * 1024f):F1}MB)");
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[Persistence] Splat load skipped ({e.Message})");
                    }
                }

                // Load refined mesh + atlas
                string refinedMeshPath = Path.Combine(pkgDir, "refined_mesh.bin");
                string refinedAtlasPath = Path.Combine(pkgDir, "refined_atlas.raw");
                string hqAtlasPath = Path.Combine(pkgDir, "hq_atlas.raw");
                bool hasMesh = File.Exists(refinedMeshPath);
                bool hasAtlas = File.Exists(refinedAtlasPath);
                bool hasHQ = File.Exists(hqAtlasPath);

                if (hasMesh)
                {
                    try
                    {
                        RefinedTextureResult meshData = default;
                        byte[] atlasBytes = null;
                        await Task.Run(() =>
                        {
                            meshData = ReadRefinedMesh(refinedMeshPath);
                            if (hasAtlas) atlasBytes = File.ReadAllBytes(refinedAtlasPath);
                        });
                        await SwitchToUnityMainThreadAsync(unitySync);

                        // Per-artifact relocation on refined mesh vertices
                        if (relocRefined != Matrix4x4.identity)
                        {
                            for (int i = 0; i < meshData.Positions.Length; i++)
                            {
                                meshData.Positions[i] = relocRefined.MultiplyPoint3x4(meshData.Positions[i]);
                                meshData.Normals[i] = relocRefined.MultiplyVector(meshData.Normals[i]).normalized;
                            }
                        }

                        if (scanner != null)
                        {
                            scanner.LastRefinedResult = meshData;

                            if (atlasBytes != null)
                            {
                                var atlasTex = new Texture2D(meshData.AtlasWidth, meshData.AtlasHeight,
                                    TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear };
                                atlasTex.SetPixelData(atlasBytes, 0);
                                atlasTex.Apply();

                                var mesh = new Mesh
                                {
                                    name = "RefinedScanMesh",
                                    indexFormat = IndexFormat.UInt32
                                };
                                mesh.SetVertices(meshData.Positions);
                                mesh.SetNormals(meshData.Normals);
                                mesh.SetUVs(0, meshData.UVs);
                                mesh.SetTriangles(meshData.Indices, 0);

                                scanner.ApplyRefinedTexture(atlasTex, mesh);
                                Debug.Log($"[Persistence] Refined atlas loaded ({meshData.AtlasWidth}x{meshData.AtlasHeight})");
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[Persistence] Refined load skipped ({e.Message})");
                    }
                }

                if (hasHQ)
                {
                    try
                    {
                        byte[] hqBytes = null;
                        await Task.Run(() => hqBytes = File.ReadAllBytes(hqAtlasPath));
                        await SwitchToUnityMainThreadAsync(unitySync);

                        int hqW = 2048, hqH = 2048;
                        if (scanner?.LastRefinedResult != null)
                        {
                            hqW = scanner.LastRefinedResult.Value.AtlasWidth;
                            hqH = scanner.LastRefinedResult.Value.AtlasHeight;
                        }

                        var hqTex = new Texture2D(hqW, hqH, TextureFormat.RGBA32, false)
                            { filterMode = FilterMode.Bilinear };
                        hqTex.SetPixelData(hqBytes, 0);
                        hqTex.Apply();

                        scanner?.ApplyHQTexture(hqTex);
                        Debug.Log("[Persistence] HQ atlas loaded");
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[Persistence] HQ load skipped ({e.Message})");
                    }
                }

                ActivePackageId = pkgId;
                _activeAnchorData = anchorData ?? new PackageAnchorData
                {
                    baseMatrixAtSave = MatrixToFloats(anchorAtSave)
                };

                Debug.Log($"[Persistence] Package loaded: {pkgId} (integ={savedIntCount}, " +
                          $"splat={File.Exists(splatPath)}, refined={hasMesh}, hq={hasHQ})");
                LoadCompleted?.Invoke();
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[Persistence] Load failed: {e.Message}\n{e.StackTrace}");
                return false;
            }
            finally { IsLoading = false; }
        }

        // ─────────────────────────────────────────────────────────────
        //  Delete package
        // ─────────────────────────────────────────────────────────────

        public async Task DeletePackageAsync(string pkgId)
        {
            string pkgDir = Path.Combine(RoomScansRoot, pkgId);

            // Erase spatial anchor if present
            string anchorJsonPath = Path.Combine(pkgDir, "anchor.json");
            if (File.Exists(anchorJsonPath))
            {
                var ad = ReadAnchorData(anchorJsonPath);
                if (ad != null && !string.IsNullOrEmpty(ad.anchorUuid) &&
                    Guid.TryParse(ad.anchorUuid, out Guid uuid))
                {
                    var mgr = RoomAnchorManager.Instance;
                    if (mgr != null)
                        await mgr.EraseSpatialAnchorAsync(uuid);
                }
            }

            // Delete directory
            if (Directory.Exists(pkgDir))
                Directory.Delete(pkgDir, true);

            // Update manifest
            var manifest = ReadManifest();
            manifest.packages.RemoveAll(p => p.id == pkgId);
            WriteManifest(manifest);

            if (ActivePackageId == pkgId)
            {
                ActivePackageId = null;
                _activeAnchorData = null;
            }

            Debug.Log($"[Persistence] Package deleted: {pkgId}");
        }

        public void ClearActivePackage()
        {
            ActivePackageId = null;
            _activeAnchorData = null;
        }

        // ─────────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────────

        private Matrix4x4 GetCurrentAnchorMatrix()
        {
            var mgr = RoomAnchorManager.Instance;
            if (mgr != null && mgr.HasSpatialAnchor)
                return mgr.SpatialAnchorMatrix;
            if (mgr != null && mgr.enabled && mgr.IsRoomLoaded)
                return mgr.GetRoomLocalToWorldForPersistence();
            return Matrix4x4.identity;
        }

        private void UpdateManifestFlags(string pkgId, ArtifactType type, bool present)
        {
            var manifest = ReadManifest();
            var entry = manifest.packages.Find(p => p.id == pkgId);
            if (entry == null) return;

            switch (type)
            {
                case ArtifactType.Splat: entry.hasSplat = present; break;
                case ArtifactType.Refined: entry.hasRefined = present; break;
                case ArtifactType.HQRefined: entry.hasHQRefined = present; break;
            }
            WriteManifest(manifest);
        }

        private static void TryDeleteFile(string path)
        {
            try { if (File.Exists(path)) File.Delete(path); }
            catch (Exception e) { Debug.LogWarning($"[Persistence] Delete failed: {path} — {e.Message}"); }
        }

        private static void CopyDirectoryContents(string src, string dst)
        {
            Directory.CreateDirectory(dst);
            foreach (var file in Directory.GetFiles(src))
                File.Copy(file, Path.Combine(dst, Path.GetFileName(file)), true);
            foreach (var dir in Directory.GetDirectories(src))
                CopyDirectoryContents(dir, Path.Combine(dst, Path.GetFileName(dir)));
        }

        // ─────────────────────────────────────────────────────────────
        //  scan.bin binary I/O (unchanged from v1)
        // ─────────────────────────────────────────────────────────────

        private static void WriteBinary(string path, int3 voxCount, float voxSize,
            int integrationCount, byte[] tsdfBytes, byte[] colorBytes, int triplanarRes,
            Matrix4x4 anchorAtSave)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536);
            using var w = new BinaryWriter(fs);

            w.Write(Magic);
            w.Write(FormatVersion);
            w.Write(DateTimeOffset.UtcNow.ToUnixTimeSeconds());

            w.Write(voxCount.x); w.Write(voxCount.y); w.Write(voxCount.z);
            w.Write(voxSize);
            w.Write(integrationCount);
            w.Write(triplanarRes);
            WriteMatrix4(w, anchorAtSave);

            w.Write(tsdfBytes.Length);
            w.Write(tsdfBytes);
            w.Write(colorBytes.Length);
            w.Write(colorBytes);
        }

        private static void WriteMatrix4(BinaryWriter w, Matrix4x4 m)
        {
            w.Write(m.m00); w.Write(m.m01); w.Write(m.m02); w.Write(m.m03);
            w.Write(m.m10); w.Write(m.m11); w.Write(m.m12); w.Write(m.m13);
            w.Write(m.m20); w.Write(m.m21); w.Write(m.m22); w.Write(m.m23);
            w.Write(m.m30); w.Write(m.m31); w.Write(m.m32); w.Write(m.m33);
        }

        private static Matrix4x4 ReadMatrix4(BinaryReader r)
        {
            float m00 = r.ReadSingle(), m01 = r.ReadSingle(), m02 = r.ReadSingle(), m03 = r.ReadSingle();
            float m10 = r.ReadSingle(), m11 = r.ReadSingle(), m12 = r.ReadSingle(), m13 = r.ReadSingle();
            float m20 = r.ReadSingle(), m21 = r.ReadSingle(), m22 = r.ReadSingle(), m23 = r.ReadSingle();
            float m30 = r.ReadSingle(), m31 = r.ReadSingle(), m32 = r.ReadSingle(), m33 = r.ReadSingle();
            return new Matrix4x4(
                new Vector4(m00, m10, m20, m30),
                new Vector4(m01, m11, m21, m31),
                new Vector4(m02, m12, m22, m32),
                new Vector4(m03, m13, m23, m33));
        }

        private static void ReadBinary(string path,
            out int3 voxCount, out float voxSize, out int integrationCount,
            out byte[] tsdfBytes, out byte[] colorBytes, out int triplanarRes,
            out Matrix4x4 anchorAtSave)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
            using var r = new BinaryReader(fs);

            uint magic = r.ReadUInt32();
            if (magic != Magic)
                throw new InvalidDataException($"Bad magic: 0x{magic:X8}, expected 0x{Magic:X8}");

            int version = r.ReadInt32();
            if (version != FormatVersion)
                throw new InvalidDataException(
                    $"scan.bin format v{version} not supported (need v{FormatVersion}).");

            r.ReadInt64(); // timestamp

            voxCount = new int3(r.ReadInt32(), r.ReadInt32(), r.ReadInt32());
            voxSize = r.ReadSingle();
            integrationCount = r.ReadInt32();
            triplanarRes = r.ReadInt32();
            anchorAtSave = ReadMatrix4(r);

            int tsdfLen = r.ReadInt32();
            tsdfBytes = r.ReadBytes(tsdfLen);
            int colorLen = r.ReadInt32();
            colorBytes = r.ReadBytes(colorLen);
        }

        // ─────────────────────────────────────────────────────────────
        //  Triplanar save helper
        // ─────────────────────────────────────────────────────────────

        private static async Task SaveTriplanarOneAtATime(TriplanarCache tc, string dir)
        {
            foreach (var (rt, fn) in new[] {
                (tc.TriXZ, "tri_xz.raw"), (tc.TriXY, "tri_xy.raw"), (tc.TriYZ, "tri_yz.raw") })
            {
                byte[] data = TriplanarCache.ReadRTBytes(rt);
                await Task.Run(() => File.WriteAllBytes(Path.Combine(dir, fn), data));
            }
            foreach (var (rt, fn) in new[] {
                (tc.DepthXZ, "depth_xz.raw"), (tc.DepthXY, "depth_xy.raw"), (tc.DepthYZ, "depth_yz.raw") })
            {
                byte[] data = TriplanarCache.ReadDepthRTBytes(rt);
                await Task.Run(() => File.WriteAllBytes(Path.Combine(dir, fn), data));
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  Refined mesh binary I/O (unchanged)
        // ─────────────────────────────────────────────────────────────

        internal static void WriteRefinedMesh(string path, RefinedTextureResult r)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536);
            using var w = new BinaryWriter(fs);

            w.Write(RefinedMeshMagic); w.Write(RefinedMeshVersion);
            w.Write(r.Positions.Length); w.Write(r.Indices.Length);
            w.Write(r.AtlasWidth); w.Write(r.AtlasHeight);

            for (int i = 0; i < r.Positions.Length; i++)
            {
                w.Write(r.Positions[i].x); w.Write(r.Positions[i].y); w.Write(r.Positions[i].z);
                w.Write(r.Normals[i].x); w.Write(r.Normals[i].y); w.Write(r.Normals[i].z);
                w.Write(r.UVs[i].x); w.Write(r.UVs[i].y);
            }
            foreach (int idx in r.Indices) w.Write(idx);
        }

        private static RefinedTextureResult ReadRefinedMesh(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
            using var r = new BinaryReader(fs);

            uint magic = r.ReadUInt32();
            if (magic != RefinedMeshMagic)
                throw new InvalidDataException($"Bad refined mesh magic: 0x{magic:X8}");
            int version = r.ReadInt32();
            if (version != RefinedMeshVersion)
                throw new InvalidDataException($"Unsupported refined mesh version: {version}");

            int vertCount = r.ReadInt32(), idxCount = r.ReadInt32();
            int atlasW = r.ReadInt32(), atlasH = r.ReadInt32();

            var positions = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];

            for (int i = 0; i < vertCount; i++)
            {
                positions[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                normals[i] = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
                uvs[i] = new Vector2(r.ReadSingle(), r.ReadSingle());
            }

            int[] indices = new int[idxCount];
            for (int i = 0; i < idxCount; i++) indices[i] = r.ReadInt32();

            return new RefinedTextureResult
            {
                Positions = positions, Normals = normals, UVs = uvs,
                Indices = indices, AtlasWidth = atlasW, AtlasHeight = atlasH
            };
        }

        // ─────────────────────────────────────────────────────────────
        //  Thread marshalling (unchanged)
        // ─────────────────────────────────────────────────────────────

        private static Task SwitchToUnityMainThreadAsync(SynchronizationContext captured)
        {
            if (captured == null)
            {
                Debug.LogWarning("[Persistence] No SynchronizationContext; MainThreadAsync fallback");
                return MainThreadAsyncAsTask();
            }
            if (ReferenceEquals(captured, SynchronizationContext.Current))
                return Task.CompletedTask;

            var tcs = new TaskCompletionSource<bool>();
            captured.Post(_ => tcs.TrySetResult(true), null);
            return tcs.Task;
        }

        private static async Task MainThreadAsyncAsTask()
        {
            await Awaitable.MainThreadAsync();
        }
    }
}
