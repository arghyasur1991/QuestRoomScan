using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    public enum SectorState
    {
        Empty,
        NoGeometry,
        MeshOnly,
        Training,
        SplatReady
    }

    public struct SectorInfo
    {
        public int Id;
        public int3 GridPos;
        public Bounds WorldAABB;
        public SectorState State;
        public int TrainingIteration;
        public float LossValue;
        public int GaussianCount;
        public float Priority;
    }

    /// <summary>
    /// Manages the 4x4x4 spatial sector grid. Schedules sectors for training
    /// based on gaze direction, proximity, keyframe coverage, and fairness.
    /// </summary>
    public class SectorScheduler
    {
        public const int GridDim = 4;
        public const int TotalSectors = GridDim * GridDim * GridDim;

        readonly SectorInfo[] _sectors = new SectorInfo[TotalSectors];
        readonly GSplatBuffers[] _sectorBuffers = new GSplatBuffers[TotalSectors];
        readonly int _maxGaussiansPerSector;

        float3 _volumeMin;
        float3 _sectorSize;

        int _currentSectorId = -1;
        int _targetItersPerSector;

        public SectorInfo[] Sectors => _sectors;
        public int CurrentSectorId => _currentSectorId;
        public GSplatBuffers CurrentBuffers => _currentSectorId >= 0 ? _sectorBuffers[_currentSectorId] : null;
        public ulong SplatMask { get; private set; }

        public SectorScheduler(int3 voxelCount, float voxelSize,
                               int maxGaussiansPerSector = 10000,
                               int targetItersPerSector = 300)
        {
            _maxGaussiansPerSector = maxGaussiansPerSector;
            _targetItersPerSector = targetItersPerSector;

            float3 volumeExtent = (float3)voxelCount * voxelSize;
            _volumeMin = -volumeExtent * 0.5f;
            _sectorSize = volumeExtent / GridDim;

            for (int i = 0; i < TotalSectors; i++)
            {
                int3 gp = new(i % GridDim, (i / GridDim) % GridDim, i / (GridDim * GridDim));
                float3 aabbMin = _volumeMin + (float3)gp * _sectorSize;
                float3 aabbMax = aabbMin + _sectorSize;
                float3 center = (aabbMin + aabbMax) * 0.5f;

                _sectors[i] = new SectorInfo
                {
                    Id = i,
                    GridPos = gp,
                    WorldAABB = new Bounds((Vector3)center, (Vector3)_sectorSize),
                    State = SectorState.Empty,
                    TrainingIteration = 0,
                    LossValue = float.MaxValue,
                    GaussianCount = 0,
                    Priority = 0
                };
            }
        }

        /// <summary>
        /// Mark a sector as having mesh data (called after mesh extraction).
        /// Skips sectors marked NoGeometry (seeding found zero vertices in their AABB).
        /// </summary>
        public void MarkMeshReady(int sectorId)
        {
            if (_sectors[sectorId].State == SectorState.Empty)
                _sectors[sectorId].State = SectorState.MeshOnly;
        }

        /// <summary>
        /// Mark a sector as permanently empty (seeding found no vertices in its AABB).
        /// MarkMeshReady will not resurrect sectors in this state.
        /// </summary>
        public void MarkNoGeometry(int sectorId)
        {
            _sectors[sectorId].State = SectorState.NoGeometry;
        }

        /// <summary>
        /// Returns the world-space AABB for a given sector.
        /// </summary>
        public Bounds GetSectorBounds(int sectorId) => _sectors[sectorId].WorldAABB;

        /// <summary>
        /// Compute sector ID from world position. Returns -1 if outside volume.
        /// </summary>
        public int WorldToSectorId(float3 worldPos)
        {
            float3 local = worldPos - _volumeMin;
            int3 gp = (int3)math.floor(local / _sectorSize);
            if (math.any(gp < 0) || math.any(gp >= GridDim)) return -1;
            return gp.x + gp.y * GridDim + gp.z * GridDim * GridDim;
        }

        /// <summary>
        /// Pick highest-priority sector matching the given state filter.
        /// </summary>
        int PickByScoredPriority(Vector3 headPos, Vector3 gazeDir,
                                  Plane[] frustumPlanes, SectorState requiredState)
        {
            float bestPriority = float.MinValue;
            int bestId = -1;

            for (int i = 0; i < TotalSectors; i++)
            {
                ref var s = ref _sectors[i];
                if (s.State != requiredState) continue;

                float priority = 0;

                if (frustumPlanes != null && GeometryUtility.TestPlanesAABB(frustumPlanes, s.WorldAABB))
                    priority += 10f;

                float dist = Vector3.Distance(headPos, s.WorldAABB.center);
                priority += 5f / (1f + dist);

                Vector3 toSector = (s.WorldAABB.center - headPos).normalized;
                float gazeDot = Vector3.Dot(gazeDir, toSector);
                priority += 3f * Mathf.Max(0, gazeDot);

                float trainFrac = (float)s.TrainingIteration / _targetItersPerSector;
                priority += 2f * (1f - Mathf.Clamp01(trainFrac));

                s.Priority = priority;
                if (priority > bestPriority)
                {
                    bestPriority = priority;
                    bestId = i;
                }
            }
            return bestId;
        }

        /// <summary>
        /// Pick the best MeshOnly sector for seeding. Returns -1 if none.
        /// </summary>
        public int PickNextMeshOnlySector(Vector3 headPos, Vector3 gazeDir, Plane[] frustumPlanes)
        {
            return PickByScoredPriority(headPos, gazeDir, frustumPlanes, SectorState.MeshOnly);
        }

        /// <summary>
        /// Pick the best Training sector for continued training. Returns -1 if none.
        /// </summary>
        public int PickNextTrainingSector(Vector3 headPos, Vector3 gazeDir, Plane[] frustumPlanes)
        {
            int id = PickByScoredPriority(headPos, gazeDir, frustumPlanes, SectorState.Training);
            _currentSectorId = id;
            return id;
        }

        /// <summary>
        /// Pick best sector from either MeshOnly or Training states (legacy).
        /// </summary>
        public int PickNextSector(Vector3 headPos, Vector3 gazeDir,
                                  Plane[] frustumPlanes, float timeSinceStart)
        {
            int id = PickNextTrainingSector(headPos, gazeDir, frustumPlanes);
            if (id < 0)
                id = PickNextMeshOnlySector(headPos, gazeDir, frustumPlanes);
            _currentSectorId = id;
            return id;
        }

        /// <summary>
        /// Get or allocate GPU buffers for a sector.
        /// </summary>
        public GSplatBuffers GetOrCreateBuffers(int sectorId)
        {
            if (_sectorBuffers[sectorId] == null)
                _sectorBuffers[sectorId] = new GSplatBuffers(_maxGaussiansPerSector, 2);
            return _sectorBuffers[sectorId];
        }

        /// <summary>
        /// Advance a sector's training state.
        /// </summary>
        public void AdvanceTraining(int sectorId, int iters, float loss, int gaussianCount)
        {
            ref var s = ref _sectors[sectorId];
            s.State = SectorState.Training;
            s.TrainingIteration += iters;
            s.LossValue = loss;
            s.GaussianCount = gaussianCount;

            if (s.TrainingIteration >= _targetItersPerSector)
            {
                s.State = SectorState.SplatReady;
                UpdateSplatMask();
            }
        }

        void UpdateSplatMask()
        {
            ulong mask = 0;
            for (int i = 0; i < TotalSectors; i++)
            {
                if (_sectors[i].State == SectorState.SplatReady)
                    mask |= 1UL << i;
            }
            SplatMask = mask;
        }

        /// <summary>
        /// Get all sectors in SplatReady state with their buffers.
        /// </summary>
        public void GetSplatReadySectors(List<(int id, GSplatBuffers buffers)> result)
        {
            result.Clear();
            for (int i = 0; i < TotalSectors; i++)
            {
                if (_sectors[i].State == SectorState.SplatReady && _sectorBuffers[i] != null)
                    result.Add((i, _sectorBuffers[i]));
            }
        }

        public void Dispose()
        {
            for (int i = 0; i < TotalSectors; i++)
            {
                _sectorBuffers[i]?.Dispose();
                _sectorBuffers[i] = null;
            }
        }
    }
}
