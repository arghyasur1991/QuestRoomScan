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
        [SerializeField] private float updateDistance = 6f;

        [Header("Freezing")]
        [SerializeField, Tooltip("Max eye-to-chunk distance for a close-range observation")]
        private float freezeDistanceThreshold = 1f;
        [SerializeField, Tooltip("Close-range observations needed before freeze can trigger")]
        private int minCloseObservations = 5;
        [SerializeField, Tooltip("Consecutive stable extractions needed to freeze")]
        private int stableCyclesRequired = 15;

        [Header("TSDF Smoothing")]
        [SerializeField, Tooltip("Bilateral filter passes on TSDF before mesh extraction. 0 = disabled.")]
        [Range(0, 3)] private int tsdfSmoothIterations = 0;
        [SerializeField, Tooltip("Bilateral range sigma for edge preservation. Lower = sharper edges.")]
        [Range(0.05f, 1f)] private float tsdfSmoothSigma = 0.3f;

        [Header("Mesh Smoothing")]
        [SerializeField, Tooltip("Post-extraction vertex smoothing iterations. 0 = disabled.")]
        [Range(0, 8)] private int meshSmoothIterations = 1;
        [SerializeField, Tooltip("Laplacian blend strength per iteration.")]
        [Range(0.1f, 1f)] private float meshSmoothLambda = 0.33f;
        [SerializeField, Tooltip("HC back-projection strength to prevent volume shrinkage.")]
        [Range(0f, 1f)] private float meshSmoothBeta = 0.5f;

        [Header("Plane Snapping")]
        [SerializeField, Tooltip("Max vertex-to-plane distance for snapping. 0 = disabled.")]
        [Range(0f, 0.1f)] private float planeSnapThreshold = 0.03f;

        [Header("Temporal Stability")]
        [SerializeField, Tooltip("Alpha for large displacements (fast convergence).")]
        [Range(0.1f, 1f)] private float temporalAlphaMax = 0.85f;
        [SerializeField, Tooltip("Alpha for long-stable vertices (strong resistance to change).")]
        [Range(0.01f, 0.5f)] private float temporalAlphaMin = 0.1f;
        [SerializeField, Tooltip("How quickly alpha decays from max to min as vertex stabilizes.")]
        [Range(0.01f, 1f)] private float temporalDecayRate = 0.15f;
        [SerializeField, Tooltip("Displacement threshold (meters) to consider a vertex still converging.")]
        [Range(0.001f, 0.02f)] private float convergenceThreshold = 0.005f;
        [SerializeField, Tooltip("Position changes below this (meters) are suppressed entirely.")]
        [Range(0f, 0.01f)] private float temporalDeadzone = 0.001f;

        [Header("Rendering")]
        [SerializeField] private Material scanMeshMaterial;

        [Header("GPU Surface Nets")]
        [SerializeField, Tooltip("Use GPU compute-based Surface Nets instead of CPU Burst jobs.")]
        private bool useGPUSurfaceNets = true;
        [SerializeField] private ComputeShader surfaceNetsCompute;
        [SerializeField, Tooltip("Max vertex fraction of total voxels (0.01-0.10).")]
        [Range(0.01f, 0.10f)] private float gpuVertexBudgetPercent = 0.05f;

        private GPUSurfaceNets _gpuSurfaceNets;
        private GPUMeshRenderer _gpuRenderer;
        private int _gpuExtractCount;

        public bool UseGPUSurfaceNets => useGPUSurfaceNets;

        private readonly Dictionary<int3, MeshChunkData> _chunks = new();
        private readonly ConcurrentQueue<int3> _meshQueue = new();
        private readonly HashSet<int3> _enqueuedCoords = new();
        private readonly SemaphoreSlim _mesherSemaphore = new(0);
        private CancellationTokenSource _workerCts;
        private readonly Plane[] _frustumPlanes = new Plane[6];
        private int _frozenChunkCount;

        public int FrozenChunkCount => _frozenChunkCount;

        private VolumeIntegrator _volume;
        private PlaneDetector _planeDetector;

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            _volume = VolumeIntegrator.Instance;
            if (_volume == null)
                throw new Exception("[RoomScan] VolumeIntegrator not found");

            _planeDetector = GetComponent<PlaneDetector>();

            var rpAsset = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline;
            Debug.Log($"[RoomScan] ChunkManager Start: mat={scanMeshMaterial?.name ?? "NULL"}, " +
                $"shader={scanMeshMaterial?.shader?.name ?? "NULL"}, " +
                $"instancing={scanMeshMaterial?.enableInstancing}, " +
                $"rp={rpAsset?.name ?? "NULL"}, " +
                $"stereoMode={UnityEngine.XR.XRSettings.stereoRenderingMode}, " +
                $"gpuSurfaceNets={useGPUSurfaceNets}");

            if (useGPUSurfaceNets && surfaceNetsCompute != null)
            {
                try
                {
                    InitGPUSurfaceNets();
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[RoomScan] GPU Surface Nets init failed, falling back to CPU: {ex.Message}");
                    _gpuSurfaceNets?.Dispose();
                    _gpuSurfaceNets = null;
                    useGPUSurfaceNets = false;
                    StartWorkers();
                }
            }
            else
            {
                StartWorkers();
            }
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
            _gpuSurfaceNets?.Dispose();
            _gpuSurfaceNets = null;
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

        private void InitGPUSurfaceNets()
        {
            _gpuSurfaceNets = new GPUSurfaceNets(surfaceNetsCompute)
            {
                MinMeshWeight = _volume.MinMeshWeight,
                SmoothIterations = meshSmoothIterations,
                SmoothLambda = meshSmoothLambda,
                SmoothBeta = meshSmoothBeta,
                PlaneSnapThreshold = planeSnapThreshold,
                TemporalAlphaMax = temporalAlphaMax,
                TemporalAlphaMin = temporalAlphaMin,
                TemporalDecayRate = temporalDecayRate,
                ConvergenceThreshold = convergenceThreshold,
                TemporalDeadzone = temporalDeadzone
            };
            _gpuSurfaceNets.EnsureBuffers(_volume.VoxelCount, gpuVertexBudgetPercent);

            _gpuRenderer = gameObject.AddComponent<GPUMeshRenderer>();
            _gpuRenderer.GpuMeshMaterial = scanMeshMaterial;
            _gpuRenderer.Initialize(_gpuSurfaceNets, _gpuSurfaceNets.GetVolumeBounds(_volume.VoxelSize));

            Debug.Log($"[RoomScan] GPU Surface Nets initialized: voxels={_volume.VoxelCount}, " +
                      $"voxSize={_volume.VoxelSize}");
        }

        private void RunGPUExtraction()
        {
            _gpuExtractCount++;

            _gpuSurfaceNets.MinMeshWeight = _volume.MinMeshWeight;

            PlaneData[] planes = null;
            int numPlanes = 0;
            if (_planeDetector != null && _planeDetector.Planes.IsCreated && _planeDetector.PlaneCount > 0)
            {
                numPlanes = _planeDetector.PlaneCount;
                planes = new PlaneData[numPlanes];
                for (int i = 0; i < numPlanes; i++)
                    planes[i] = _planeDetector.Planes[i];
            }

            _gpuSurfaceNets.Extract(
                _volume.Volume, _volume.ColorVolume,
                _volume.VoxelSize, planes, numPlanes);

            if (_gpuExtractCount <= 3 || _gpuExtractCount % 50 == 0)
                Debug.Log($"[RoomScan] GPU extraction #{_gpuExtractCount}");
        }

        private int _updateCallCount;
        private int _totalEnqueued;

        /// <summary>
        /// Called by RoomScanner after each integration pass to enqueue dirty chunks.
        /// </summary>
        public void UpdateDirtyChunks()
        {
            if (useGPUSurfaceNets && _gpuSurfaceNets != null)
            {
                RunGPUExtraction();
                return;
            }
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
            int frozenSkipped = 0;

            for (int x = chunkMin.x; x <= chunkMax.x; x++)
            for (int y = chunkMin.y; y <= chunkMax.y; y++)
            for (int z = chunkMin.z; z <= chunkMax.z; z++)
            {
                tested++;
                int3 coord = new(x, y, z);
                if (!ChunkInFrustum(coord)) { frustumFailed++; continue; }
                if (_enqueuedCoords.Contains(coord)) { alreadyQueued++; continue; }

                float3 chunkCenter = ((float3)coord + 0.5f) * chunkWorldSize;
                float distToChunk = math.distance((float3)eyePos, chunkCenter);

                if (_chunks.TryGetValue(coord, out MeshChunkData existing))
                {
                    existing.MinObserveDistance = math.min(existing.MinObserveDistance, distToChunk);
                    if (distToChunk <= freezeDistanceThreshold)
                        existing.CloseObservations++;

                    if (existing.Frozen) { frozenSkipped++; continue; }
                }

                if (existing == null)
                {
                    existing = CreateChunk(coord);
                    existing.MinObserveDistance = distToChunk;
                    existing.CloseObservations = distToChunk <= freezeDistanceThreshold ? 1 : 0;
                    _chunks[coord] = existing;
                }

                existing.Dirty = true;
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
                          $"frozen={frozenSkipped}, enqueued={enqueued}, totalEnqueued={_totalEnqueued}, " +
                          $"chunks={_chunks.Count}, frozenTotal={_frozenChunkCount}");
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
                Mesher = new SurfaceNetsMesher
                {
                    MinMeshWeight = _volume.MinMeshWeight,
                    TsdfSmoothIterations = tsdfSmoothIterations,
                    TsdfSmoothSigma = tsdfSmoothSigma,
                    MeshSmoothIterations = meshSmoothIterations,
                    MeshSmoothLambda = meshSmoothLambda,
                    MeshSmoothBeta = meshSmoothBeta,
                    PlaneSnapThreshold = planeSnapThreshold,
                    TemporalAlphaMax = temporalAlphaMax,
                    TemporalAlphaMin = temporalAlphaMin,
                    TemporalDecayRate = temporalDecayRate,
                    ConvergenceThreshold = convergenceThreshold,
                    TemporalDeadzone = temporalDeadzone
                }
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

                if (!chunk.WeightData.IsCreated || chunk.WeightData.Length < totalSize)
                {
                    if (chunk.WeightData.IsCreated) chunk.WeightData.Dispose();
                    chunk.WeightData = new NativeArray<sbyte>(totalSize, Allocator.Persistent);
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
                        TsdfDest = chunk.VolumeData,
                        WeightDest = chunk.WeightData,
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

                NativeArray<PlaneData> planes = default;
                int numPlanes = 0;
                if (_planeDetector != null)
                {
                    planes = _planeDetector.Planes;
                    numPlanes = _planeDetector.PlaneCount;
                }

                bool populated = await chunk.Mesher.CreateMesh(
                    chunk.VolumeData, chunk.WeightData,
                    hasColorData ? chunk.ColorData : default,
                    size, _volume.VoxelSize, chunk.Mesh,
                    planes, numPlanes, ctkn);
                chunk.IsPopulated = populated;
                if (populated) _meshSuccesses++;

                int newVerts = chunk.Mesh.vertexCount;
                if (populated && chunk.PreviousVertexCount > 0)
                {
                    float delta = Mathf.Abs(newVerts - chunk.PreviousVertexCount) / (float)Mathf.Max(chunk.PreviousVertexCount, 1);
                    bool closeEnough = chunk.CloseObservations >= minCloseObservations;
                    chunk.StableCount = (delta < 0.05f && closeEnough) ? chunk.StableCount + 1 : 0;
                    if (chunk.StableCount >= stableCyclesRequired && !chunk.Frozen)
                    {
                        chunk.Frozen = true;
                        _frozenChunkCount++;
                        Debug.Log($"[RoomScan] Chunk {chunk.Coord} frozen: verts={newVerts}, " +
                                  $"minDist={chunk.MinObserveDistance:F2}, closeObs={chunk.CloseObservations}");
                    }
                }
                chunk.PreviousVertexCount = newVerts;

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
            _frozenChunkCount = 0;
            while (_meshQueue.TryDequeue(out _)) { }

            if (useGPUSurfaceNets && _gpuSurfaceNets != null)
            {
                _gpuSurfaceNets.Dispose();
                _gpuSurfaceNets = null;
                if (_gpuRenderer != null) Destroy(_gpuRenderer);
                InitGPUSurfaceNets();
            }
            else if (enabled)
            {
                StartWorkers();
            }
        }

        public void UnfreezeAll()
        {
            foreach (var chunk in _chunks.Values)
            {
                chunk.Frozen = false;
                chunk.StableCount = 0;
                chunk.PreviousVertexCount = 0;
                chunk.MinObserveDistance = float.MaxValue;
                chunk.CloseObservations = 0;
            }
            _frozenChunkCount = 0;
        }

        public void RemeshAll()
        {
            UnfreezeAll();
            foreach (var kvp in _chunks)
            {
                int3 coord = kvp.Key;
                if (_enqueuedCoords.Contains(coord)) continue;
                kvp.Value.Dirty = true;
                _meshQueue.Enqueue(coord);
                _enqueuedCoords.Add(coord);
                _mesherSemaphore.Release();
            }
            Debug.Log($"[RoomScan] RemeshAll: enqueued {_chunks.Count} chunks");
        }

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
            public NativeArray<sbyte> TsdfDest;
            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<sbyte> WeightDest;

            public void Execute(int i)
            {
                TsdfDest[DestOffset + i] = Source[i * 2];
                WeightDest[DestOffset + i] = Source[i * 2 + 1];
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
        public bool Frozen;
        public int StableCount;
        public int PreviousVertexCount;
        public float MinObserveDistance = float.MaxValue;
        public int CloseObservations;
        public NativeArray<sbyte> VolumeData;
        public NativeArray<sbyte> WeightData;
        public NativeArray<Color32> ColorData;
        public SurfaceNetsMesher Mesher;

        public void Dispose()
        {
            if (VolumeData.IsCreated) VolumeData.Dispose();
            if (WeightData.IsCreated) WeightData.Dispose();
            if (ColorData.IsCreated) ColorData.Dispose();
            Mesher?.Dispose();
            if (Mesh) UnityEngine.Object.Destroy(Mesh);
        }
    }
}
