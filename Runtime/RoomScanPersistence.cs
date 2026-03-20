using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class RoomScanPersistence : MonoBehaviour
    {
        public static RoomScanPersistence Instance { get; private set; }

        private const uint Magic = 0x48534D52; // "RMSH"
        /// <summary>
        /// v1: header + anchor localToWorld (16 floats) + TSDF + color blobs.
        /// </summary>
        private const int FormatVersion = 1;

        private string SaveDirectory => Path.Combine(Application.persistentDataPath, "RoomScans");
        public string SaveFilePath => Path.Combine(SaveDirectory, "scan.bin");
        public string TriplanarDirectory => Path.Combine(SaveDirectory, "triplanar");

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

                Matrix4x4 anchorAtSave = Matrix4x4.identity;
                var anchor = RoomAnchorManager.Instance;
                if (anchor != null && anchor.enabled && anchor.IsRoomLoaded)
                    anchorAtSave = anchor.GetRoomLocalToWorldForPersistence();

                string savePath = SaveFilePath;
                string triDir = TriplanarDirectory;
                await Task.Run(() => WriteBinary(savePath, s, vi.VoxelSize,
                    vi.IntegrationCount, tsdfBytes, colorBytes, triRes, anchorAtSave));

                tsdfBytes = null;
                colorBytes = null;

                if (tc != null && triRes > 0)
                {
                    if (!Directory.Exists(triDir)) Directory.CreateDirectory(triDir);
                    await SaveTriplanarOneAtATime(tc, triDir);
                }

                float sizeMB = new FileInfo(savePath).Length / (1024f * 1024f);
                Debug.Log($"[RoomScan] Persistence: saved to {savePath} ({sizeMB:F1}MB), " +
                          $"triplanar={triRes > 0}, format=v{FormatVersion}, " +
                          $"anchor col3={anchorAtSave.GetColumn(3)}, row0={anchorAtSave.GetRow(0)}");
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
                // Capture Unity's sync context on entry (UI / main thread). After Task.Run, continuations
                // can run on the thread pool on some IL2CPP builds. Do not use Awaitable.MainThreadAsync()
                // inside async Task — it can stall forever on device; marshal with captured context instead.
                var unitySync = SynchronizationContext.Current;

                // Capture paths on main thread — SaveFilePath/TriplanarDirectory use
                // Application.persistentDataPath; must not run inside Task.Run (Android/IL2CPP).
                string saveFilePath = SaveFilePath;
                string triplanarDir = TriplanarDirectory;

                byte[] tsdfBytes = null;
                byte[] colorBytes = null;
                int3 savedVoxCount = default;
                float savedVoxSize = 0;
                int savedIntCount = 0;
                int triRes = 0;
                Matrix4x4 anchorAtSave = Matrix4x4.identity;

                Debug.Log("[RoomScan] Persistence: load reading file (background)...");
                await Task.Run(() => ReadBinary(saveFilePath,
                    out savedVoxCount, out savedVoxSize, out savedIntCount,
                    out tsdfBytes, out colorBytes, out triRes, out anchorAtSave));
                Debug.Log($"[RoomScan] Persistence: load read done voxels={savedVoxCount}, tsdf={tsdfBytes?.Length ?? 0}, color={colorBytes?.Length ?? 0}");

                await SwitchToUnityMainThreadAsync(unitySync);
                Debug.Log("[RoomScan] Persistence: load continuing on main thread");

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

                Debug.Log("[RoomScan] Persistence: uploading volumes to GPU...");
                if (!vi.LoadVolumes(tsdfBytes, colorBytes, savedIntCount))
                    return false;

                // Wait for MRUK room anchor before computing relocation.
                // MRUK loads asynchronously from Start(); if the user triggers
                // "Load Scan" before the room is ready, we'd skip the bake and
                // the mesh would land at the old-session coordinates.
                // Uses Task.Delay polling (not TCS event subscription) to avoid
                // deadlocks with Unity's SynchronizationContext on IL2CPP/Quest.
                var anchor = RoomAnchorManager.Instance;
                if (anchor != null && anchor.enabled && !anchor.IsRoomLoaded)
                {
                    Debug.Log("[RoomScan] Persistence: waiting for MRUK room to load...");
                    for (int i = 0; i < 300 && !anchor.IsRoomLoaded; i++)
                    {
                        await Task.Delay(16);
                        await SwitchToUnityMainThreadAsync(unitySync);
                    }
                    if (anchor.IsRoomLoaded)
                        Debug.Log("[RoomScan] Persistence: MRUK room loaded, proceeding");
                    else
                        Debug.LogWarning("[RoomScan] Persistence: MRUK room load timed out after ~5s, proceeding without relocation");
                }

                // Bake anchor-drift relocation: R = A_now * Inv(A_save).
                Matrix4x4 reloc = Matrix4x4.identity;
                if (anchor != null && anchor.enabled && anchor.IsRoomLoaded)
                    reloc = anchor.ComputeRelocationMatrix(anchorAtSave);

                if (reloc != Matrix4x4.identity)
                {
                    vi.BakeRelocation(reloc);
                    Debug.Log("[RoomScan] Persistence: TSDF baked to identity");
                }

                var tc = TriplanarCache.Instance;
                if (tc != null && triRes > 0 && Directory.Exists(triplanarDir))
                {
                    try
                    {
                        if (reloc != Matrix4x4.identity)
                        {
                            tc.BakeRelocation(reloc, triplanarDir);
                            Debug.Log("[RoomScan] Persistence: triplanar baked to identity (from disk)");
                        }
                        else
                        {
                            tc.Load(triplanarDir);
                            tc.LoadDepth(triplanarDir);
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[RoomScan] Persistence: triplanar load skipped ({e.Message})");
                    }
                }

                if (cm != null)
                {
                    cm.Reinitialize();
                    cm.Extract();
                    Debug.Log("[RoomScan] Persistence: mesh extracted from loaded volume");
                }

                Debug.Log($"[RoomScan] Persistence: loaded scan (integrations={savedIntCount})");
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
            var planes = new (RenderTexture rt, string filename)[] {
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

            var depthPlanes = new (RenderTexture rt, string filename)[] {
                (tc.DepthXZ, "depth_xz.raw"),
                (tc.DepthXY, "depth_xy.raw"),
                (tc.DepthYZ, "depth_yz.raw")
            };
            foreach (var (rt, filename) in depthPlanes)
            {
                byte[] data = TriplanarCache.ReadDepthRTBytes(rt);
                string path = Path.Combine(dir, filename);
                await Task.Run(() => File.WriteAllBytes(path, data));
                data = null;
            }
        }

        private static void WriteBinary(string path, int3 voxCount, float voxSize,
            int integrationCount, byte[] tsdfBytes, byte[] colorBytes, int triplanarRes,
            Matrix4x4 anchorAtSave)
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
            // Written row-by-row (m00..m03, m10..); Unity ctor expects column vectors.
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
                    $"scan.bin format v{version} is not supported (need v{FormatVersion}). Delete RoomScans on device.");

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

        /// <summary>
        /// Returns to the Unity player main thread after <see cref="Task.Run"/> I/O.
        /// Prefer posting to the <see cref="SynchronizationContext"/> captured while still on the main thread;
        /// <see cref="Awaitable.MainThreadAsync"/> inside <c>async Task</c> methods has been observed to hang on Quest.
        /// </summary>
        private static Task SwitchToUnityMainThreadAsync(SynchronizationContext capturedAtLoadStart)
        {
            if (capturedAtLoadStart == null)
            {
                Debug.LogWarning("[RoomScan] Persistence: no SynchronizationContext when load started; using MainThreadAsync fallback");
                return MainThreadAsyncAsTask();
            }

            if (ReferenceEquals(capturedAtLoadStart, SynchronizationContext.Current))
                return Task.CompletedTask;

            var tcs = new TaskCompletionSource<bool>();
            capturedAtLoadStart.Post(_ => tcs.TrySetResult(true), null);
            return tcs.Task;
        }

        private static async Task MainThreadAsyncAsTask()
        {
            await Awaitable.MainThreadAsync();
        }
    }
}
