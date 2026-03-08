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

                AsyncGPUReadbackRequest req = await AsyncGPUReadback.RequestAsync(
                    _volume.Volume, 0,
                    start.x, size.x,
                    start.y, size.y,
                    start.z, size.z);

                if (req.hasError)
                {
                    Debug.LogWarning($"[RoomScan] MeshChunk {chunk.Coord}: GPU readback error");
                    return;
                }
                ctkn.ThrowIfCancellationRequested();

                int sliceSize = size.x * size.y;
                int totalSize = sliceSize * size.z;

                if (!chunk.VolumeData.IsCreated || chunk.VolumeData.Length < totalSize)
                {
                    if (chunk.VolumeData.IsCreated) chunk.VolumeData.Dispose();
                    chunk.VolumeData = new NativeArray<sbyte>(totalSize, Allocator.Persistent);
                }

                for (int z = 0; z < size.z; z++)
                {
                    NativeArray<sbyte> slice = req.GetData<sbyte>(z);
                    int dstOffset = z * sliceSize;

                    var copier = new CopySliceJob
                    {
                        Source = slice,
                        Destination = chunk.VolumeData,
                        DestOffset = dstOffset
                    };
                    copier.ScheduleParallelByRef(slice.Length, 64, default).Complete();
                }

                bool populated = await chunk.Mesher.CreateMesh(chunk.VolumeData, size, _volume.VoxelSize, chunk.Mesh, ctkn);
                chunk.IsPopulated = populated;
                if (populated) _meshSuccesses++;

                if (_meshAttempts <= 5 || _meshAttempts % 20 == 0)
                    Debug.Log($"[RoomScan] MeshChunk {chunk.Coord}: populated={populated}, " +
                              $"verts={chunk.Mesh.vertexCount}, attempts={_meshAttempts}, successes={_meshSuccesses}");
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
        private struct CopySliceJob : IJobFor
        {
            [ReadOnly] public NativeArray<sbyte> Source;
            public int DestOffset;
            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<sbyte> Destination;

            public void Execute(int i)
            {
                Destination[DestOffset + i] = Source[i];
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
        public SurfaceNetsMesher Mesher;

        public void Dispose()
        {
            if (VolumeData.IsCreated) VolumeData.Dispose();
            Mesher?.Dispose();
            if (Mesh) UnityEngine.Object.Destroy(Mesh);
        }
    }
}
