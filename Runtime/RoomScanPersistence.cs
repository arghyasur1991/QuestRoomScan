using System;
using System.IO;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class RoomScanPersistence : MonoBehaviour
    {
        public static RoomScanPersistence Instance { get; private set; }

        [Header("Settings")]
#pragma warning disable CS0414
        [SerializeField] private bool autoSaveOnPause = true;
        [SerializeField] private bool autoLoadOnStart = true;
#pragma warning restore CS0414

        private const uint Magic = 0x48534D52; // "RMSH"
        private const int FormatVersion = 1;

        private string SaveDirectory => Path.Combine(Application.persistentDataPath, "RoomScans");
        private string SaveFilePath => Path.Combine(SaveDirectory, "scan.bin");
        private string TriplanarDirectory => Path.Combine(SaveDirectory, "triplanar");

        public bool IsSaving { get; private set; }
        public bool IsLoading { get; private set; }

        public event Action SaveCompleted;
        public event Action LoadCompleted;

        private void Awake() => Instance = this;

        public bool HasSavedScan()
        {
            bool exists = File.Exists(SaveFilePath);
            Debug.Log($"[RoomScan] Persistence: HasSavedScan={exists}, path={SaveFilePath}");
            return exists;
        }

        public async Task<bool> SaveAsync()
        {
            var vi = VolumeIntegrator.Instance;
            if (vi == null || vi.Volume == null)
            {
                Debug.LogWarning("[RoomScan] Persistence: cannot save, no volume");
                return false;
            }

            if (IsSaving)
            {
                Debug.LogWarning("[RoomScan] Persistence: save already in progress");
                return false;
            }

            IsSaving = true;
            try
            {
                if (!Directory.Exists(SaveDirectory))
                    Directory.CreateDirectory(SaveDirectory);

                int3 s = vi.VoxelCount;

                Vector3 volumeOrigin = Vector3.zero;
                var anchor = RoomAnchorManager.Instance;
                if (anchor != null)
                    volumeOrigin = anchor.OriginInRoomSpace;

                var tsdfReq = await AsyncGPUReadback.RequestAsync(vi.Volume, 0);
                if (tsdfReq.hasError)
                {
                    Debug.LogError("[RoomScan] Persistence: TSDF readback failed");
                    return false;
                }

                byte[] tsdfBytes = new byte[s.x * s.y * s.z * 2];
                for (int z = 0; z < s.z; z++)
                {
                    var slice = tsdfReq.GetData<byte>(z);
                    int sliceBytes = s.x * s.y * 2;
                    Unity.Collections.NativeArray<byte>.Copy(slice, 0, tsdfBytes, z * sliceBytes, sliceBytes);
                }

                var colorReq = await AsyncGPUReadback.RequestAsync(vi.ColorVolume, 0);
                if (colorReq.hasError)
                {
                    Debug.LogError("[RoomScan] Persistence: Color readback failed");
                    return false;
                }

                byte[] colorBytes = new byte[s.x * s.y * s.z * 4];
                for (int z = 0; z < s.z; z++)
                {
                    var slice = colorReq.GetData<byte>(z);
                    int sliceBytes = s.x * s.y * 4;
                    Unity.Collections.NativeArray<byte>.Copy(slice, 0, colorBytes, z * sliceBytes, sliceBytes);
                }

                var tc = TriplanarCache.Instance;
                int triRes = 0;
                if (tc != null && tc.TriXZ != null)
                    triRes = tc.TriXZ.width;

                string savePath = SaveFilePath;
                string triDir = TriplanarDirectory;
                await Task.Run(() => WriteBinary(savePath, s, vi.VoxelSize,
                    vi.IntegrationCount, tsdfBytes, colorBytes, triRes, volumeOrigin));

                tsdfBytes = null;
                colorBytes = null;

                if (tc != null && triRes > 0)
                {
                    if (!Directory.Exists(triDir)) Directory.CreateDirectory(triDir);
                    await SaveTriplanarOneAtATime(tc, triDir);
                }

                float sizeMB = new FileInfo(savePath).Length / (1024f * 1024f);
                Debug.Log($"[RoomScan] Persistence: saved to {savePath} ({sizeMB:F1}MB), " +
                          $"triplanar={triRes > 0}, volumeOrigin={volumeOrigin}");
                SaveCompleted?.Invoke();
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] Persistence: save failed: {e.Message}\n{e.StackTrace}");
                return false;
            }
            finally
            {
                IsSaving = false;
            }
        }

        public async Task<bool> LoadAsync()
        {
            if (!HasSavedScan())
            {
                Debug.Log("[RoomScan] Persistence: no saved scan found");
                return false;
            }

            var vi = VolumeIntegrator.Instance;
            var cm = MeshExtractor.Instance;
            if (vi == null || vi.Volume == null)
            {
                Debug.LogWarning("[RoomScan] Persistence: cannot load, no volume");
                return false;
            }

            if (IsLoading)
            {
                Debug.LogWarning("[RoomScan] Persistence: load already in progress");
                return false;
            }

            IsLoading = true;
            try
            {
                byte[] tsdfBytes = null;
                byte[] colorBytes = null;
                int3 savedVoxCount = default;
                float savedVoxSize = 0;
                int savedIntCount = 0;
                int triRes = 0;
                Vector3 volumeOrigin = Vector3.zero;

                await Task.Run(() => ReadBinary(SaveFilePath,
                    out savedVoxCount, out savedVoxSize, out savedIntCount,
                    out tsdfBytes, out colorBytes, out triRes, out volumeOrigin));

                int3 currentVox = vi.VoxelCount;
                if (math.any(savedVoxCount != currentVox))
                {
                    Debug.LogWarning($"[RoomScan] Persistence: voxel count mismatch " +
                        $"saved={savedVoxCount} current={currentVox}, deleting stale save");
                    DeleteSavedScan();
                    return false;
                }
                if (Mathf.Abs(savedVoxSize - vi.VoxelSize) > 0.001f)
                {
                    Debug.LogWarning($"[RoomScan] Persistence: voxel size mismatch " +
                        $"saved={savedVoxSize} current={vi.VoxelSize}, deleting stale save");
                    DeleteSavedScan();
                    return false;
                }

                var anchor = RoomAnchorManager.Instance;
                if (anchor != null && volumeOrigin != Vector3.zero)
                    anchor.SetOriginInRoomSpace(volumeOrigin);

                vi.LoadVolumes(tsdfBytes, colorBytes, savedIntCount);

                var tc = TriplanarCache.Instance;
                if (tc != null && triRes > 0 && Directory.Exists(TriplanarDirectory))
                    tc.Load(TriplanarDirectory);

                if (cm != null)
                    cm.Reinitialize();

                Debug.Log($"[RoomScan] Persistence: loaded scan (integrations={savedIntCount}, " +
                          $"volumeOrigin={volumeOrigin})");
                LoadCompleted?.Invoke();
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] Persistence: load failed: {e.Message}\n{e.StackTrace}");
                return false;
            }
            finally
            {
                IsLoading = false;
            }
        }

        public void DeleteSavedScan()
        {
            if (File.Exists(SaveFilePath)) File.Delete(SaveFilePath);
            if (Directory.Exists(TriplanarDirectory))
                Directory.Delete(TriplanarDirectory, true);
            Debug.Log("[RoomScan] Persistence: saved scan deleted");
        }

        private static async Task SaveTriplanarOneAtATime(TriplanarCache tc, string dir)
        {
            var planes = new[] {
                (tc.TriXZ, "tri_xz.raw"),
                (tc.TriXY, "tri_xy.raw"),
                (tc.TriYZ, "tri_yz.raw")
            };
            foreach (var (rt, filename) in planes)
            {
                byte[] data = TriplanarCache.ReadRTBytes(rt);
                string path = Path.Combine(dir, filename);
                await Task.Run(() => File.WriteAllBytes(path, data));
                data = null;
            }
        }

        private static void WriteBinary(string path, int3 voxCount, float voxSize,
            int integrationCount, byte[] tsdfBytes, byte[] colorBytes, int triplanarRes,
            Vector3 volumeOrigin)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536);
            using var w = new BinaryWriter(fs);

            w.Write(Magic);
            w.Write(FormatVersion);
            w.Write(DateTimeOffset.UtcNow.ToUnixTimeSeconds());

            w.Write(voxCount.x);
            w.Write(voxCount.y);
            w.Write(voxCount.z);
            w.Write(voxSize);
            w.Write(integrationCount);
            w.Write(triplanarRes);

            w.Write(volumeOrigin.x);
            w.Write(volumeOrigin.y);
            w.Write(volumeOrigin.z);

            w.Write(tsdfBytes.Length);
            w.Write(tsdfBytes);

            w.Write(colorBytes.Length);
            w.Write(colorBytes);
        }

        private static void ReadBinary(string path,
            out int3 voxCount, out float voxSize, out int integrationCount,
            out byte[] tsdfBytes, out byte[] colorBytes, out int triplanarRes,
            out Vector3 volumeOrigin)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
            using var r = new BinaryReader(fs);

            uint magic = r.ReadUInt32();
            if (magic != Magic)
                throw new InvalidDataException($"Bad magic: 0x{magic:X8}, expected 0x{Magic:X8}");

            int version = r.ReadInt32();
            if (version > FormatVersion)
                throw new InvalidDataException($"Unsupported version: {version}");

            r.ReadInt64(); // timestamp

            voxCount = new int3(r.ReadInt32(), r.ReadInt32(), r.ReadInt32());
            voxSize = r.ReadSingle();
            integrationCount = r.ReadInt32();
            triplanarRes = r.ReadInt32();

            volumeOrigin = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

            int tsdfLen = r.ReadInt32();
            tsdfBytes = r.ReadBytes(tsdfLen);

            int colorLen = r.ReadInt32();
            colorBytes = r.ReadBytes(colorLen);
        }
    }
}
