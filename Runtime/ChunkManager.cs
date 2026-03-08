using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class ChunkManager : MonoBehaviour
    {
        public static ChunkManager Instance { get; private set; }

        [Header("Chunks")]
        [SerializeField] private float3 chunkWorldSize = new(4, 4, 4);
        [SerializeField] private float overlap = 0.25f;
        [SerializeField] private int numMeshWorkers = 2;
        [SerializeField] private float updateDistance = 5f;

        [Header("Rendering")]
        [SerializeField] private Material scanMeshMaterial;

        private readonly Dictionary<int3, MeshChunkData> _chunks = new();
        private readonly ConcurrentQueue<int3> _meshQueue = new();
        private readonly HashSet<int3> _enqueuedCoords = new();
        private readonly SemaphoreSlim _mesherSemaphore = new(0);
        private CancellationTokenSource _workerCts;
        private readonly Plane[] _frustumPlanes = new Plane[6];

        private VolumeIntegrator _volume;

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            _volume = VolumeIntegrator.Instance;
            if (_volume == null)
                throw new Exception("[RoomScan] VolumeIntegrator not found");

            var rpAsset = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline;
            Debug.Log($"[RoomScan] ChunkManager Start: mat={scanMeshMaterial?.name ?? "NULL"}, " +
                $"shader={scanMeshMaterial?.shader?.name ?? "NULL"}, " +
                $"instancing={scanMeshMaterial?.enableInstancing}, " +
                $"rp={rpAsset?.name ?? "NULL"}, " +
                $"stereoMode={UnityEngine.XR.XRSettings.stereoRenderingMode}");

            StartWorkers();
        }

        private void OnDisable()
        {
            _workerCts?.Cancel();
        }

        private void OnDestroy()
        {
            _workerCts?.Cancel();
            foreach (var chunk in _chunks.Values)
                chunk.Dispose();
            _chunks.Clear();
        }

        private void StartWorkers()
        {
            _workerCts?.Cancel();
            _workerCts = new CancellationTokenSource();

            for (int i = 0; i < numMeshWorkers; i++)
                _ = RunMesherWorker(_workerCts.Token);
        }

        private async Task RunMesherWorker(CancellationToken ctkn)
        {
            try
            {
                while (!ctkn.IsCancellationRequested)
                {
                    await _mesherSemaphore.WaitAsync(ctkn);

                    if (!_meshQueue.TryDequeue(out int3 coord))
                        continue;
                    _enqueuedCoords.Remove(coord);

                    if (!_chunks.TryGetValue(coord, out MeshChunkData chunk))
                    {
                        chunk = CreateChunk(coord);
                        _chunks[coord] = chunk;
                    }

                    await MeshChunk(chunk, ctkn);
                }
            }
            catch (OperationCanceledException) { }
        }

        private int _updateCallCount;
        private int _totalEnqueued;

        /// <summary>
        /// Called by RoomScanner after each integration pass to enqueue dirty chunks.
        /// </summary>
        public void UpdateDirtyChunks()
        {
            if (!DepthCapture.DepthAvailable) return;

            _updateCallCount++;

            UpdateFrustumPlanes();

            var depthProj = DepthCapture.Instance.Proj[0];
            depthProj = WithFiniteFarPlane(depthProj, updateDistance);
            Matrix4x4 projInv = depthProj.inverse;
            Matrix4x4 view = DepthCapture.Instance.View[0];
            Matrix4x4 viewInv = view.inverse;

            Vector4[] ndcCorners =
            {
                new(-1, -1, 1, 1),
                new(1, -1, 1, 1),
                new(1, 1, 1, 1),
                new(-1, 1, 1, 1)
            };

            float3 boxMin = float.MaxValue;
            float3 boxMax = float.MinValue;

            for (int i = 0; i < 4; i++)
            {
                Vector4 localCorner = projInv * ndcCorners[i];
                Vector3 farCorner = new(
                    localCorner.x / localCorner.w,
                    localCorner.y / localCorner.w,
                    localCorner.z / localCorner.w
                );
                Vector3 worldCorner = viewInv.MultiplyPoint(farCorner);
                boxMin = math.min(boxMin, (float3)worldCorner);
                boxMax = math.max(boxMax, (float3)worldCorner);
            }

            Vector3 eyePos = viewInv.MultiplyPoint(Vector3.zero);
            boxMin = math.min(boxMin, (float3)eyePos);
            boxMax = math.max(boxMax, (float3)eyePos);

            int3 chunkMin = (int3)math.floor(boxMin / chunkWorldSize - 1);
            int3 chunkMax = (int3)math.floor(boxMax / chunkWorldSize + 1);

            int enqueued = 0;
            int tested = 0;
            int frustumFailed = 0;
            int alreadyQueued = 0;

            for (int x = chunkMin.x; x <= chunkMax.x; x++)
            for (int y = chunkMin.y; y <= chunkMax.y; y++)
            for (int z = chunkMin.z; z <= chunkMax.z; z++)
            {
                tested++;
                int3 coord = new(x, y, z);
                if (!ChunkInFrustum(coord)) { frustumFailed++; continue; }
                if (_enqueuedCoords.Contains(coord)) { alreadyQueued++; continue; }

                if (!_chunks.TryGetValue(coord, out MeshChunkData chunk))
                {
                    chunk = CreateChunk(coord);
                    _chunks[coord] = chunk;
                }

                chunk.Dirty = true;
                _meshQueue.Enqueue(coord);
                _enqueuedCoords.Add(coord);
                _mesherSemaphore.Release();
                enqueued++;
                _totalEnqueued++;
            }

            if (_updateCallCount <= 3 || _updateCallCount % 20 == 0)
            {
                Debug.Log($"[RoomScan] UpdateDirtyChunks #{_updateCallCount}: eye={eyePos:F2}, " +
                          $"box=[{boxMin:F1}→{boxMax:F1}], chunkRange=[{chunkMin}→{chunkMax}], " +
                          $"tested={tested}, frustumFail={frustumFailed}, alreadyQueued={alreadyQueued}, " +
                          $"enqueued={enqueued}, totalEnqueued={_totalEnqueued}, " +
                          $"chunks={_chunks.Count}, matAssigned={scanMeshMaterial != null}");
            }
        }

        private MeshChunkData CreateChunk(int3 coord)
        {
            var go = new GameObject($"Chunk_{coord.x}_{coord.y}_{coord.z}");
            go.transform.SetParent(transform);
            go.transform.localPosition = Vector3.zero;

            var mf = go.AddComponent<MeshFilter>();
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = scanMeshMaterial;

            var mesh = new Mesh { name = $"chunk_{coord}" };
            mesh.MarkDynamic();
            mesh.vertexBufferTarget |= GraphicsBuffer.Target.Structured;
            mf.sharedMesh = mesh;

            return new MeshChunkData
            {
                Coord = coord,
                GameObject = go,
                MeshFilter = mf,
                Mesh = mesh,
                Extents = chunkWorldSize + overlap,
                Mesher = new SurfaceNetsMesher()
            };
        }

        private int _meshAttempts;
        private int _meshSuccesses;

        private async Task MeshChunk(MeshChunkData chunk, CancellationToken ctkn)
        {
            _meshAttempts++;
            try
            {
                float3 worldPos = chunk.Coord * chunkWorldSize;
                int3 start = _volume.WorldToVoxel(worldPos);
                int3 end = start + new int3(chunk.Extents / _volume.VoxelSize);

                for (int d = 0; d < 3; d++)
                {
                    if (start[d] >= _volume.VoxelCount[d])
                    {
                        if (_meshAttempts <= 5)
                            Debug.Log($"[RoomScan] MeshChunk {chunk.Coord}: skipped, start[{d}]={start[d]} >= volCount[{d}]={_volume.VoxelCount[d]}");
                        return;
                    }
                    if (end[d] <= 0)
                    {
                        if (_meshAttempts <= 5)
                            Debug.Log($"[RoomScan] MeshChunk {chunk.Coord}: skipped, end[{d}]={end[d]} <= 0");
                        return;
                    }
                }

                start = math.max(start, 0);
                end = math.min(end, _volume.VoxelCount - 1);
                int3 size = end - start;
                if (math.any(size <= 0))
                {
                    if (_meshAttempts <= 5)
                        Debug.Log($"[RoomScan] MeshChunk {chunk.Coord}: skipped, size={size} <= 0");
                    return;
                }

                // Position the chunk GO at the world-space origin of this voxel sub-region.
                // CoordToPos in SurfaceNetsMesher produces positions starting from 0 in local space,
                // so the GO must sit at where voxel 'start' maps to in world space (minus half-voxel).
                float3 origin = ((float3)start - (float3)_volume.VoxelCount / 2f) * _volume.VoxelSize;
                chunk.GameObject.transform.position = (Vector3)origin;

                if (_meshAttempts <= 5)
                    Debug.Log($"[RoomScan] MeshChunk {chunk.Coord}: readback start={start}, size={size}, origin={origin}");

                int sliceSize = size.x * size.y;
                int totalSize = sliceSize * size.z;

                if (!chunk.VolumeData.IsCreated || chunk.VolumeData.Length < totalSize)
                {
                    if (chunk.VolumeData.IsCreated) chunk.VolumeData.Dispose();
                    chunk.VolumeData = new NativeArray<sbyte>(totalSize, Allocator.Persistent);
                }

                if (!chunk.ColorData.IsCreated || chunk.ColorData.Length < totalSize)
                {
                    if (chunk.ColorData.IsCreated) chunk.ColorData.Dispose();
                    chunk.ColorData = new NativeArray<Color32>(totalSize, Allocator.Persistent);
                }

                // TSDF readback — copy data immediately before requesting color readback,
                // because GetData NativeArrays are invalidated by subsequent readback requests.
                AsyncGPUReadbackRequest tsdfReq = await AsyncGPUReadback.RequestAsync(
                    _volume.Volume, 0,
                    start.x, size.x,
                    start.y, size.y,
                    start.z, size.z);

                if (tsdfReq.hasError)
                {
                    Debug.LogWarning($"[RoomScan] MeshChunk {chunk.Coord}: TSDF readback error");
                    return;
                }
                ctkn.ThrowIfCancellationRequested();

                for (int z = 0; z < size.z; z++)
                {
                    NativeArray<sbyte> tsdfSlice = tsdfReq.GetData<sbyte>(z);
                    int dstOffset = z * sliceSize;
                    var copier = new CopySliceRG8Job
                    {
                        Source = tsdfSlice,
                        Destination = chunk.VolumeData,
                        DestOffset = dstOffset
                    };
                    copier.ScheduleParallelByRef(sliceSize, 64, default).Complete();
                }

                // Color volume readback — separate request, copy immediately.
                // Wrapped in try-catch so color failure doesn't prevent mesh creation.
                bool hasColorData = false;
                try
                {
                    AsyncGPUReadbackRequest colorReq = await AsyncGPUReadback.RequestAsync(
                        _volume.ColorVolume, 0,
                        start.x, size.x,
                        start.y, size.y,
                        start.z, size.z);

                    if (!colorReq.hasError)
                    {
                        for (int z = 0; z < size.z; z++)
                        {
                            NativeArray<Color32> colorSlice = colorReq.GetData<Color32>(z);
                            NativeArray<Color32>.Copy(colorSlice, 0, chunk.ColorData, z * sliceSize, sliceSize);
                        }
                        hasColorData = true;
                    }
                }
                catch (Exception colorEx)
                {
                    if (_meshAttempts <= 3)
                        Debug.LogWarning($"[RoomScan] MeshChunk {chunk.Coord}: Color readback failed: {colorEx.Message}");
                }
                ctkn.ThrowIfCancellationRequested();

                bool populated = await chunk.Mesher.CreateMesh(
                    chunk.VolumeData,
                    hasColorData ? chunk.ColorData : default,
                    size, _volume.VoxelSize, chunk.Mesh, ctkn);
                chunk.IsPopulated = populated;
                if (populated) _meshSuccesses++;

                if (_meshAttempts <= 5 || _meshAttempts % 20 == 0)
                    Debug.Log($"[RoomScan] MeshChunk {chunk.Coord}: populated={populated}, " +
                              $"verts={chunk.Mesh.vertexCount}, attempts={_meshAttempts}, successes={_meshSuccesses}");

                if (populated && (_meshSuccesses <= 3 || _meshSuccesses % 50 == 0))
                {
                    var mr = chunk.GameObject.GetComponent<MeshRenderer>();
                    Debug.Log($"[RoomScan] ChunkDiag {chunk.Coord}: " +
                        $"go.active={chunk.GameObject.activeSelf}, " +
                        $"mr.enabled={mr.enabled}, " +
                        $"mat={mr.sharedMaterial?.name ?? "NULL"}, " +
                        $"shader={mr.sharedMaterial?.shader?.name ?? "NULL"}, " +
                        $"worldPos={chunk.GameObject.transform.position}, " +
                        $"localScale={chunk.GameObject.transform.localScale}, " +
                        $"meshBounds={chunk.Mesh.bounds}, " +
                        $"rendererBounds={mr.bounds}, " +
                        $"layer={chunk.GameObject.layer}");
                }
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                Debug.LogError($"[RoomScan] MeshChunk {chunk.Coord} exception: {e.Message}\n{e.StackTrace}");
            }
            finally
            {
                chunk.Dirty = false;
            }
        }

        private void UpdateFrustumPlanes()
        {
            Matrix4x4 proj = DepthCapture.Instance.Proj[0];
            proj = WithFiniteFarPlane(proj, updateDistance);
            Matrix4x4 view = DepthCapture.Instance.View[0];
            GeometryUtility.CalculateFrustumPlanes(proj * view, _frustumPlanes);
        }

        private bool ChunkInFrustum(int3 coord)
        {
            float3 min = coord * chunkWorldSize;
            float3 center = min + chunkWorldSize / 2f;
            Bounds b = new(center, chunkWorldSize);
            return GeometryUtility.TestPlanesAABB(_frustumPlanes, b);
        }

        public void ClearAllChunks()
        {
            _workerCts?.Cancel();
            foreach (var chunk in _chunks.Values)
            {
                if (chunk.GameObject) Destroy(chunk.GameObject);
                chunk.Dispose();
            }
            _chunks.Clear();
            _enqueuedCoords.Clear();
            while (_meshQueue.TryDequeue(out _)) { }
            if (enabled) StartWorkers();
        }

        /// <summary>
        /// Returns all populated chunk meshes for texture projection.
        /// </summary>
        public IEnumerable<MeshChunkData> GetPopulatedChunks()
        {
            foreach (var kvp in _chunks)
            {
                if (kvp.Value.IsPopulated)
                    yield return kvp.Value;
            }
        }

        private static Matrix4x4 WithFiniteFarPlane(Matrix4x4 infiniteProj, float far)
        {
            float near = -infiniteProj.m23 * 0.5f;
            Matrix4x4 proj = infiniteProj;
            proj.m22 = -(far + near) / (far - near);
            proj.m23 = -(2f * far * near) / (far - near);
            proj.m32 = -1f;
            proj.m33 = 0f;
            return proj;
        }

        [BurstCompile]
        private struct CopySliceRG8Job : IJobFor
        {
            [ReadOnly] public NativeArray<sbyte> Source;
            public int DestOffset;
            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<sbyte> Destination;

            public void Execute(int i)
            {
                Destination[DestOffset + i] = Source[i * 2];
            }
        }
    }

    public class MeshChunkData : IDisposable
    {
        public int3 Coord;
        public GameObject GameObject;
        public MeshFilter MeshFilter;
        public Mesh Mesh;
        public float3 Extents;
        public bool Dirty;
        public bool IsPopulated;
        public NativeArray<sbyte> VolumeData;
        public NativeArray<Color32> ColorData;
        public SurfaceNetsMesher Mesher;

        public void Dispose()
        {
            if (VolumeData.IsCreated) VolumeData.Dispose();
            if (ColorData.IsCreated) ColorData.Dispose();
            Mesher?.Dispose();
            if (Mesh) UnityEngine.Object.Destroy(Mesh);
        }
    }
}
