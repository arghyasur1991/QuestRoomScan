using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Mathematics.Geometry;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public struct PlaneData
    {
        public float3 Normal;
        public float Distance;
        public float Confidence;
    }

    public class SurfaceNetsMesher : IDisposable
    {
        private NativeList<Vertex> _verts;
        private NativeList<int3> _vertCoords;
        private NativeArray<int> _coordVertMap;
        private NativeList<uint> _tris;
        private NativeReference<MinMaxAABB> _boundsRef;

        private NativeArray<sbyte> _smoothA;
        private NativeArray<sbyte> _smoothB;

        private NativeArray<float3> _posA;
        private NativeArray<float3> _posB;
        private NativeArray<float3> _origPositions;
        private NativeArray<float3> _normals;

        private NativeArray<float4> _prevState;

        private bool _busy;
        public bool IsBusy => _busy;

        public float MinMeshWeight { get; set; } = 0.15f;

        public int TsdfSmoothIterations { get; set; } = 0;
        public float TsdfSmoothSigma { get; set; } = 0.3f;

        public int MeshSmoothIterations { get; set; } = 1;
        public float MeshSmoothLambda { get; set; } = 0.33f;
        public float MeshSmoothBeta { get; set; } = 0.5f;

        public float PlaneSnapThreshold { get; set; } = 0.03f;

        public float TemporalAlphaMax { get; set; } = 0.85f;
        public float TemporalAlphaMin { get; set; } = 0.1f;
        public float TemporalDecayRate { get; set; } = 0.15f;
        public float ConvergenceThreshold { get; set; } = 0.005f;
        public float TemporalDeadzone { get; set; } = 0.001f;

        private const int InvalidVert = -1;
        private const float SbyteMax = sbyte.MaxValue;

        private static readonly byte[] CrnrOffsIdxA =
        {
            0, 1, 2, 3,
            4, 5, 6, 7,
            0, 1, 2, 3
        };

        private static readonly byte[] CrnrOffsIdxB =
        {
            1, 2, 3, 0,
            5, 6, 7, 4,
            4, 5, 6, 7
        };

        private static readonly int3[] CornerOffs =
        {
            new(0, 0, 0), new(1, 0, 0), new(1, 0, 1), new(0, 0, 1),
            new(0, 1, 0), new(1, 1, 0), new(1, 1, 1), new(0, 1, 1)
        };

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct Vertex
        {
            public float3 pos;
            public float3 norm;
            public Color32 color;

            public static readonly VertexAttributeDescriptor[] Layout =
            {
                new(VertexAttribute.Position),
                new(VertexAttribute.Normal),
                new(VertexAttribute.Color, VertexAttributeFormat.UNorm8, 4)
            };
        }

        #region Phase 1: TSDF Bilateral Smoothing

        [BurstCompile]
        private struct BilateralSmoothTsdfJob : IJobFor
        {
            [ReadOnly] public NativeArray<sbyte> Source;
            [ReadOnly] public NativeArray<sbyte> Weights;
            [WriteOnly] public NativeArray<sbyte> Dest;
            public int3 VoxCount;
            public float SigmaRange;
            public float MinWeightSbyte;

            public void Execute(int i)
            {
                sbyte rawVal = Source[i];
                if (rawVal == sbyte.MinValue || Weights[i] < MinWeightSbyte)
                {
                    Dest[i] = rawVal;
                    return;
                }

                int3 coord = new int3(
                    i % VoxCount.x,
                    i / VoxCount.x % VoxCount.y,
                    i / (VoxCount.x * VoxCount.y));

                if (coord.x < 1 || coord.y < 1 || coord.z < 1 ||
                    coord.x >= VoxCount.x - 1 || coord.y >= VoxCount.y - 1 ||
                    coord.z >= VoxCount.z - 1)
                {
                    Dest[i] = rawVal;
                    return;
                }

                float centerVal = rawVal / SbyteMax;
                float invSigSq2 = 1f / math.max(2f * SigmaRange * SigmaRange, 0.001f);
                float sumTsdf = 0f;
                float sumW = 0f;
                int sliceXY = VoxCount.x * VoxCount.y;

                for (int dz = -1; dz <= 1; dz++)
                {
                    int nz = i + dz * sliceXY;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int nzy = nz + dy * VoxCount.x;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int ni = nzy + dx;
                            sbyte nRaw = Source[ni];
                            if (nRaw == sbyte.MinValue) continue;
                            float nWeight = Weights[ni];
                            if (nWeight < MinWeightSbyte) continue;

                            float nVal = nRaw / SbyteMax;
                            float nConf = nWeight / SbyteMax;

                            int manhattan = math.abs(dx) + math.abs(dy) + math.abs(dz);
                            float spatialW = manhattan <= 1 ? 1f : (manhattan == 2 ? 0.7f : 0.5f);

                            float diff = nVal - centerVal;
                            float rangeW = math.exp(-diff * diff * invSigSq2);

                            float w = spatialW * rangeW * nConf;
                            sumTsdf += nVal * w;
                            sumW += w;
                        }
                    }
                }

                if (sumW > 0.001f)
                {
                    float smoothed = sumTsdf / sumW;
                    int rounded = (int)math.round(smoothed * SbyteMax);
                    Dest[i] = (sbyte)math.clamp(rounded, -127, 127);
                }
                else
                {
                    Dest[i] = rawVal;
                }
            }
        }

        #endregion

        #region Phase 2: Normal-Aware Vertex Smoothing

        [BurstCompile]
        private struct SmoothVerticesJob : IJobFor
        {
            [ReadOnly] public NativeArray<float3> InputPositions;
            [ReadOnly] public NativeArray<float3> Normals;
            [ReadOnly] public NativeArray<float3> OriginalPositions;
            [WriteOnly] public NativeArray<float3> OutputPositions;
            [ReadOnly] public NativeArray<int> CoordVertMap;
            [ReadOnly] public NativeList<int3> VertCoords;
            public int3 VoxCount;
            public float Lambda;
            public float Beta;

            public void Execute(int i)
            {
                float3 pos = InputPositions[i];
                float3 norm = Normals[i];
                int3 coord = VertCoords[i];

                float3 laplacian = float3.zero;
                float totalWeight = 0f;

                for (int axis = 0; axis < 3; axis++)
                {
                    for (int dir = -1; dir <= 1; dir += 2)
                    {
                        int3 nc = coord;
                        nc[axis] += dir;

                        if (nc[axis] < 0 || nc[axis] >= VoxCount[axis]) continue;

                        int flatIdx = nc.x + nc.y * VoxCount.x + nc.z * VoxCount.x * VoxCount.y;
                        if (flatIdx < 0 || flatIdx >= CoordVertMap.Length) continue;

                        int neighborVert = CoordVertMap[flatIdx];
                        if (neighborVert < 0 || neighborVert >= InputPositions.Length) continue;

                        float normalDot = math.dot(norm, Normals[neighborVert]);
                        float nw = math.max(0f, normalDot);

                        laplacian += InputPositions[neighborVert] * nw;
                        totalWeight += nw;
                    }
                }

                if (totalWeight < 0.001f)
                {
                    OutputPositions[i] = pos;
                    return;
                }

                laplacian /= totalWeight;

                float3 q = math.lerp(pos, laplacian, Lambda);
                float3 original = OriginalPositions[i];
                OutputPositions[i] = q - Beta * (q - original);
            }
        }

        #endregion

        #region Phase 3: Plane Snap

        [BurstCompile]
        private struct PlaneSnapJob : IJobFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<float3> Positions;
            [ReadOnly] public NativeArray<float3> Normals;
            [ReadOnly] public NativeArray<PlaneData> Planes;
            public int NumPlanes;
            public float SnapThreshold;

            public void Execute(int i)
            {
                float3 pos = Positions[i];
                float3 norm = Normals[i];

                float bestDist = SnapThreshold;
                int bestPlane = -1;

                for (int p = 0; p < NumPlanes; p++)
                {
                    float normalDot = math.abs(math.dot(norm, Planes[p].Normal));
                    if (normalDot < 0.9f) continue;

                    float dist = math.abs(math.dot(pos, Planes[p].Normal) - Planes[p].Distance);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestPlane = p;
                    }
                }

                if (bestPlane >= 0)
                {
                    PlaneData plane = Planes[bestPlane];
                    float signedDist = math.dot(pos, plane.Normal) - plane.Distance;
                    float strength = math.saturate(plane.Confidence);
                    Positions[i] = pos - signedDist * plane.Normal * strength;
                }
            }
        }

        #endregion

        #region Phase 4: Adaptive Temporal Blend

        [BurstCompile]
        private struct AdaptiveTemporalBlendJob : IJobFor
        {
            [NativeDisableParallelForRestriction]
            public NativeList<Vertex> Verts;
            [ReadOnly] public NativeList<int3> VertCoords;
            [NativeDisableParallelForRestriction]
            public NativeArray<float4> PrevState;
            public int3 VoxCount;
            public float AlphaMax;
            public float AlphaMin;
            public float DecayRate;
            public float ConvergeThresh;
            public float Deadzone;

            public void Execute(int i)
            {
                var v = Verts[i];
                int key = VertCoords[i].x + VertCoords[i].y * VoxCount.x
                        + VertCoords[i].z * VoxCount.x * VoxCount.y;

                float4 prev = PrevState[key];
                float prevAge = prev.w;

                if (prevAge < 0f)
                {
                    PrevState[key] = new float4(v.pos, 0f);
                    return;
                }

                float3 prevPos = prev.xyz;
                float dist = math.distance(v.pos, prevPos);

                if (dist < Deadzone)
                {
                    v.pos = prevPos;
                    Verts[i] = v;
                    PrevState[key] = new float4(prevPos, prevAge + 1f);
                    return;
                }

                float age = dist > ConvergeThresh ? 0f : prevAge + 1f;
                float alpha = AlphaMin + (AlphaMax - AlphaMin) * math.exp(-age * DecayRate);
                v.pos = math.lerp(prevPos, v.pos, alpha);
                Verts[i] = v;
                PrevState[key] = new float4(v.pos, age);
            }
        }

        #endregion

        #region Existing: VertexJob

        [BurstCompile]
        private struct VertexJob : IJob
        {
            [ReadOnly] public NativeArray<sbyte> Volume;
            [ReadOnly] public NativeArray<sbyte> Weights;
            [ReadOnly] public NativeArray<Color32> Colors;
            [ReadOnly] public int3 VoxCount;
            [ReadOnly] public float VoxSize;
            [ReadOnly] public float MinWeight;

            [WriteOnly] public NativeArray<int> CoordVertMap;
            public NativeList<Vertex> Verts;
            public NativeList<int3> VertCoords;
            public NativeReference<MinMaxAABB> BoundsRef;

            public void Execute()
            {
                int total = VoxCount.x * VoxCount.y * VoxCount.z;
                MinMaxAABB bounds = new();

                for (int i = 0; i < total; i++)
                {
                    int3 coord = IndexToCoord(i);

                    if (coord.x >= VoxCount.x - 1 ||
                        coord.y >= VoxCount.y - 1 ||
                        coord.z >= VoxCount.z - 1)
                    {
                        CoordVertMap[i] = InvalidVert;
                        continue;
                    }

                    float3 posCoord = default;
                    float3 dir = default;
                    byte numCrossings = 0;
                    byte numBadCrossings = 0;

                    float weightThreshSbyte = MinWeight * SbyteMax;

                    for (int e = 0; e < 12; e++)
                    {
                        int3 coordA = coord + CornerOffs[CrnrOffsIdxA[e]];
                        int3 coordB = coord + CornerOffs[CrnrOffsIdxB[e]];

                        int idxA = CoordToIndex(coordA);
                        int idxB = CoordToIndex(coordB);
                        sbyte rawA = Volume[idxA];
                        sbyte rawB = Volume[idxB];
                        bool emptyA = rawA == sbyte.MinValue;
                        bool emptyB = rawB == sbyte.MinValue;

                        if (Weights.IsCreated && Weights.Length > 0)
                        {
                            if (!emptyA && Weights[idxA] < weightThreshSbyte) emptyA = true;
                            if (!emptyB && Weights[idxB] < weightThreshSbyte) emptyB = true;
                        }

                        float valA = emptyA ? 0f : rawA / SbyteMax;
                        float valB = emptyB ? 0f : rawB / SbyteMax;

                        float change = valA - valB;
                        dir += new float3(coordA - coordB) * change;

                        bool crosses = valA < 0 != valB < 0;
                        if (crosses)
                        {
                            if (emptyA || emptyB)
                                numBadCrossings++;

                            float t = valA / change;
                            float3 crossingCoord = coordA + t * new float3(coordB - coordA);
                            posCoord += crossingCoord;
                            numCrossings++;
                        }
                    }

                    if (numCrossings < 3 || numCrossings == numBadCrossings)
                    {
                        CoordVertMap[i] = InvalidVert;
                        continue;
                    }

                    posCoord /= numCrossings;
                    float3 pos = CoordToPos(posCoord);
                    float3 norm = math.normalize(dir);

                    Color32 vertColor = new(20, 20, 25, 255);
                    if (Colors.IsCreated && Colors.Length > 0)
                    {
                        int cIdx = CoordToIndex(coord);
                        if (cIdx >= 0 && cIdx < Colors.Length)
                        {
                            Color32 c = Colors[cIdx];
                            if (c.a > 10)
                                vertColor = new Color32(c.r, c.g, c.b, 255);
                        }
                    }

                    Vertex vert = new()
                    {
                        pos = pos,
                        norm = norm,
                        color = vertColor
                    };

                    bounds.Encapsulate(pos);
                    CoordVertMap[i] = Verts.Length;
                    Verts.Add(vert);
                    VertCoords.Add(coord);
                }

                BoundsRef.Value = bounds;
            }

            private int3 IndexToCoord(int i)
            {
                return new int3(
                    i % VoxCount.x,
                    i / VoxCount.x % VoxCount.y,
                    i / (VoxCount.x * VoxCount.y)
                );
            }

            private int CoordToIndex(int3 c)
            {
                return c.x + c.y * VoxCount.x + c.z * VoxCount.x * VoxCount.y;
            }

            private float3 CoordToPos(float3 c)
            {
                return c * VoxSize + VoxSize * 0.5f;
            }

            private float ValueAt(int3 c)
            {
                return Volume[CoordToIndex(c)] / SbyteMax;
            }
        }

        #endregion

        #region Existing: IndexJob

        [BurstCompile]
        private struct IndexJob : IJob
        {
            [ReadOnly] public NativeArray<sbyte> Volume;
            [ReadOnly] public NativeArray<sbyte> Weights;
            [ReadOnly] public int3 VoxCount;
            [ReadOnly] public float MinWeight;
            [ReadOnly] public NativeList<int3> VertCoords;
            [ReadOnly] public NativeArray<int> CoordVertMap;
            public NativeList<uint> Tris;

            public void Execute()
            {
                int3 X = new(1, 0, 0);
                int3 Y = new(0, 1, 0);
                int3 Z = new(0, 0, 1);

                foreach (int3 coord in VertCoords)
                {
                    TrisForAxis(coord, X, Z, Y);
                    TrisForAxis(coord, Y, X, Z);
                    TrisForAxis(coord, Z, Y, X);
                }
            }

            private void TrisForAxis(int3 coord, int3 axis, int3 d1, int3 d2)
            {
                if (math.any(coord - d1 < int3.zero) || math.any(coord - d2 < int3.zero))
                    return;
                if (math.any(coord + axis >= VoxCount))
                    return;

                float va = ValueAt(coord);
                float vb = ValueAt(coord + axis);
                if (va < 0 == vb < 0) return;

                int a = CoordVertMap[Flatten(coord)];
                int b = CoordVertMap[Flatten(coord - d1)];
                int c = CoordVertMap[Flatten(coord - (d1 + d2))];
                int d = CoordVertMap[Flatten(coord - d2)];

                if (a == InvalidVert || b == InvalidVert || c == InvalidVert || d == InvalidVert)
                    return;

                if (va < 0)
                {
                    AddTri(c, b, a);
                    AddTri(d, c, a);
                }
                else
                {
                    AddTri(a, c, d);
                    AddTri(a, b, c);
                }
            }

            private void AddTri(int a, int b, int c)
            {
                Tris.Add((uint)a);
                Tris.Add((uint)b);
                Tris.Add((uint)c);
            }

            private int Flatten(int3 c)
            {
                return c.x + c.y * VoxCount.x + c.z * VoxCount.x * VoxCount.y;
            }

            private float ValueAt(int3 c)
            {
                int idx = Flatten(c);
                sbyte raw = Volume[idx];
                if (raw == sbyte.MinValue) return 0f;
                if (Weights.IsCreated && Weights.Length > 0 && Weights[idx] < MinWeight * SbyteMax)
                    return 0f;
                return raw / SbyteMax;
            }
        }

        #endregion

        public async Task<bool> CreateMesh(
            NativeArray<sbyte> volume, NativeArray<sbyte> weights, NativeArray<Color32> colors,
            int3 voxCount, float voxSize, Mesh mesh,
            NativeArray<PlaneData> detectedPlanes = default, int numPlanes = 0,
            CancellationToken ctkn = default)
        {
            if (_busy) throw new InvalidOperationException("Mesher is busy");
            _busy = true;

            if (!_verts.IsCreated)
            {
                int vertEstimate = volume.Length / 3;
                int triEstimate = vertEstimate * 6;

                _verts = new NativeList<Vertex>(vertEstimate, Allocator.Persistent);
                _vertCoords = new NativeList<int3>(vertEstimate, Allocator.Persistent);
                _tris = new NativeList<uint>(triEstimate, Allocator.Persistent);
                _boundsRef = new NativeReference<MinMaxAABB>(Allocator.Persistent);
                _coordVertMap = new NativeArray<int>(volume.Length, Allocator.Persistent);
            }

            if (_coordVertMap.IsCreated && _coordVertMap.Length < volume.Length)
            {
                _coordVertMap.Dispose();
                _coordVertMap = new NativeArray<int>(volume.Length, Allocator.Persistent);
            }

            _verts.Clear();
            _vertCoords.Clear();
            _tris.Clear();
            _boundsRef.Value = new MinMaxAABB();

            bool hasTriangles = false;

            try
            {
                // ── Phase 1: TSDF bilateral smoothing ──
                NativeArray<sbyte> meshVolume = volume;

                if (TsdfSmoothIterations > 0)
                {
                    int len = volume.Length;
                    EnsureNativeArray(ref _smoothA, len);
                    if (TsdfSmoothIterations > 1) EnsureNativeArray(ref _smoothB, len);

                    for (int iter = 0; iter < TsdfSmoothIterations; iter++)
                    {
                        NativeArray<sbyte> src = iter == 0 ? volume :
                            (iter % 2 == 1 ? _smoothA : _smoothB);
                        NativeArray<sbyte> dst = iter % 2 == 0 ? _smoothA : _smoothB;

                        var smoothJob = new BilateralSmoothTsdfJob
                        {
                            Source = src,
                            Weights = weights,
                            Dest = dst,
                            VoxCount = voxCount,
                            SigmaRange = TsdfSmoothSigma,
                            MinWeightSbyte = MinMeshWeight * SbyteMax
                        };
                        JobHandle handle = smoothJob.ScheduleParallelByRef(len, 256, default);
                        while (!handle.IsCompleted)
                            await Awaitable.NextFrameAsync(ctkn);
                        handle.Complete();
                        ctkn.ThrowIfCancellationRequested();
                    }

                    meshVolume = TsdfSmoothIterations % 2 == 1 ? _smoothA : _smoothB;
                }

                // ── Surface Nets: VertexJob ──
                var vertJob = new VertexJob
                {
                    Volume = meshVolume,
                    Weights = weights,
                    Colors = colors,
                    VoxCount = voxCount,
                    VoxSize = voxSize,
                    MinWeight = MinMeshWeight,
                    CoordVertMap = _coordVertMap,
                    Verts = _verts,
                    VertCoords = _vertCoords,
                    BoundsRef = _boundsRef
                };

                JobHandle vertHandle = vertJob.Schedule();
                while (!vertHandle.IsCompleted)
                    await Awaitable.NextFrameAsync(ctkn);
                vertHandle.Complete();
                ctkn.ThrowIfCancellationRequested();

                if (_verts.Length < 3)
                    return false;

                int vertCount = _verts.Length;

                // ── Phase 2: Normal-aware vertex smoothing ──
                if (MeshSmoothIterations > 0)
                {
                    EnsureNativeArray(ref _posA, vertCount);
                    EnsureNativeArray(ref _posB, vertCount);
                    EnsureNativeArray(ref _origPositions, vertCount);
                    EnsureNativeArray(ref _normals, vertCount);

                    NativeArray<Vertex> vertArray = _verts.AsArray();
                    for (int i = 0; i < vertCount; i++)
                    {
                        _posA[i] = vertArray[i].pos;
                        _origPositions[i] = vertArray[i].pos;
                        _normals[i] = vertArray[i].norm;
                    }

                    for (int iter = 0; iter < MeshSmoothIterations; iter++)
                    {
                        NativeArray<float3> src = iter % 2 == 0 ? _posA : _posB;
                        NativeArray<float3> dst = iter % 2 == 0 ? _posB : _posA;

                        var smoothVert = new SmoothVerticesJob
                        {
                            InputPositions = src,
                            Normals = _normals,
                            OriginalPositions = _origPositions,
                            OutputPositions = dst,
                            CoordVertMap = _coordVertMap,
                            VertCoords = _vertCoords,
                            VoxCount = voxCount,
                            Lambda = MeshSmoothLambda,
                            Beta = MeshSmoothBeta
                        };
                        JobHandle sh = smoothVert.ScheduleParallelByRef(vertCount, 64, default);
                        while (!sh.IsCompleted)
                            await Awaitable.NextFrameAsync(ctkn);
                        sh.Complete();
                        ctkn.ThrowIfCancellationRequested();
                    }

                    NativeArray<float3> finalPos = MeshSmoothIterations % 2 == 1 ? _posB : _posA;

                    // ── Phase 3: Plane snapping ──
                    if (detectedPlanes.IsCreated && numPlanes > 0 && PlaneSnapThreshold > 0f)
                    {
                        var snapJob = new PlaneSnapJob
                        {
                            Positions = finalPos,
                            Normals = _normals,
                            Planes = detectedPlanes,
                            NumPlanes = numPlanes,
                            SnapThreshold = PlaneSnapThreshold
                        };
                        JobHandle snapH = snapJob.ScheduleParallelByRef(vertCount, 64, default);
                        while (!snapH.IsCompleted)
                            await Awaitable.NextFrameAsync(ctkn);
                        snapH.Complete();
                        ctkn.ThrowIfCancellationRequested();
                    }

                    for (int i = 0; i < vertCount; i++)
                    {
                        var v = vertArray[i];
                        v.pos = finalPos[i];
                        vertArray[i] = v;
                    }
                }
                else if (detectedPlanes.IsCreated && numPlanes > 0 && PlaneSnapThreshold > 0f)
                {
                    EnsureNativeArray(ref _posA, vertCount);
                    EnsureNativeArray(ref _normals, vertCount);
                    NativeArray<Vertex> vertArray = _verts.AsArray();
                    for (int i = 0; i < vertCount; i++)
                    {
                        _posA[i] = vertArray[i].pos;
                        _normals[i] = vertArray[i].norm;
                    }

                    var snapJob = new PlaneSnapJob
                    {
                        Positions = _posA,
                        Normals = _normals,
                        Planes = detectedPlanes,
                        NumPlanes = numPlanes,
                        SnapThreshold = PlaneSnapThreshold
                    };
                    JobHandle snapH = snapJob.ScheduleParallelByRef(vertCount, 64, default);
                    while (!snapH.IsCompleted)
                        await Awaitable.NextFrameAsync(ctkn);
                    snapH.Complete();
                    ctkn.ThrowIfCancellationRequested();

                    for (int i = 0; i < vertCount; i++)
                    {
                        var v = vertArray[i];
                        v.pos = _posA[i];
                        vertArray[i] = v;
                    }
                }

                // ── Phase 4: Adaptive temporal vertex damping ──
                if (TemporalAlphaMax < 1f)
                {
                    int totalVoxels = voxCount.x * voxCount.y * voxCount.z;
                    if (!_prevState.IsCreated || _prevState.Length < totalVoxels)
                    {
                        if (_prevState.IsCreated) _prevState.Dispose();
                        _prevState = new NativeArray<float4>(totalVoxels, Allocator.Persistent);
                        for (int i = 0; i < totalVoxels; i++)
                            _prevState[i] = new float4(0, 0, 0, -1f);
                    }

                    var temporalJob = new AdaptiveTemporalBlendJob
                    {
                        Verts = _verts,
                        VertCoords = _vertCoords,
                        PrevState = _prevState,
                        VoxCount = voxCount,
                        AlphaMax = TemporalAlphaMax,
                        AlphaMin = TemporalAlphaMin,
                        DecayRate = TemporalDecayRate,
                        ConvergeThresh = ConvergenceThreshold,
                        Deadzone = TemporalDeadzone
                    };
                    JobHandle th = temporalJob.ScheduleParallelByRef(vertCount, 64, default);
                    while (!th.IsCompleted)
                        await Awaitable.NextFrameAsync(ctkn);
                    th.Complete();
                    ctkn.ThrowIfCancellationRequested();
                }

                // ── Recompute bounds after position modifications ──
                MinMaxAABB finalBounds = new();
                {
                    NativeArray<Vertex> va = _verts.AsArray();
                    for (int i = 0; i < va.Length; i++)
                        finalBounds.Encapsulate(va[i].pos);
                }

                // ── IndexJob ──
                int maxIndices = _verts.Length * 18;
                if (_tris.Capacity < maxIndices)
                    _tris.Capacity = maxIndices;

                var triJob = new IndexJob
                {
                    Volume = meshVolume,
                    Weights = weights,
                    VoxCount = voxCount,
                    MinWeight = MinMeshWeight,
                    VertCoords = _vertCoords,
                    CoordVertMap = _coordVertMap,
                    Tris = _tris
                };

                JobHandle triHandle = triJob.Schedule();
                while (!triHandle.IsCompleted)
                    await Awaitable.NextFrameAsync(ctkn);
                triHandle.Complete();
                ctkn.ThrowIfCancellationRequested();

                float3 size = finalBounds.Max - finalBounds.Min;
                Bounds bounds = new(finalBounds.Center, size);

                ApplyToMesh(_verts.AsArray(), _tris.AsArray(), bounds, mesh);
                hasTriangles = _tris.Length > 0;
            }
            finally
            {
                _busy = false;
            }

            return hasTriangles;
        }

        private static void EnsureNativeArray<T>(ref NativeArray<T> arr, int minLen)
            where T : unmanaged
        {
            if (arr.IsCreated && arr.Length >= minLen) return;
            if (arr.IsCreated) arr.Dispose();
            arr = new NativeArray<T>(minLen, Allocator.Persistent);
        }

        private static void ApplyToMesh(NativeArray<Vertex> verts, NativeArray<uint> tris,
            Bounds bounds, Mesh mesh)
        {
            Mesh.MeshDataArray meshDataArray = Mesh.AllocateWritableMeshData(1);
            Mesh.MeshData meshData = meshDataArray[0];

            meshData.SetVertexBufferParams(verts.Length, Vertex.Layout);
            meshData.SetIndexBufferParams(tris.Length, IndexFormat.UInt32);

            NativeArray<Vertex> vb = meshData.GetVertexData<Vertex>();
            vb.CopyFrom(verts);

            NativeArray<uint> ib = meshData.GetIndexData<uint>();
            ib.CopyFrom(tris);

            const MeshUpdateFlags flags = MeshUpdateFlags.DontNotifyMeshUsers |
                                          MeshUpdateFlags.DontRecalculateBounds |
                                          MeshUpdateFlags.DontValidateIndices;
            meshData.subMeshCount = 1;
            meshData.SetSubMesh(0, new SubMeshDescriptor(0, tris.Length), flags);

            Mesh.ApplyAndDisposeWritableMeshData(meshDataArray, mesh);
            mesh.bounds = bounds;
            mesh.MarkModified();
        }

        public void Dispose()
        {
            if (_verts.IsCreated) _verts.Dispose();
            if (_vertCoords.IsCreated) _vertCoords.Dispose();
            if (_coordVertMap.IsCreated) _coordVertMap.Dispose();
            if (_tris.IsCreated) _tris.Dispose();
            if (_boundsRef.IsCreated) _boundsRef.Dispose();
            if (_smoothA.IsCreated) _smoothA.Dispose();
            if (_smoothB.IsCreated) _smoothB.Dispose();
            if (_posA.IsCreated) _posA.Dispose();
            if (_posB.IsCreated) _posB.Dispose();
            if (_origPositions.IsCreated) _origPositions.Dispose();
            if (_normals.IsCreated) _normals.Dispose();
            if (_prevState.IsCreated) _prevState.Dispose();
        }
    }
}
