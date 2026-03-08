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
    public class SurfaceNetsMesher : IDisposable
    {
        private NativeList<Vertex> _verts;
        private NativeList<int3> _vertCoords;
        private NativeArray<int> _coordVertMap;
        private NativeList<uint> _tris;
        private NativeReference<MinMaxAABB> _boundsRef;

        private bool _busy;
        public bool IsBusy => _busy;

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

        [BurstCompile]
        private struct VertexJob : IJob
        {
            [ReadOnly] public NativeArray<sbyte> Volume;
            [ReadOnly] public int3 VoxCount;
            [ReadOnly] public float VoxSize;

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

                    for (int e = 0; e < 12; e++)
                    {
                        int3 coordA = coord + CornerOffs[CrnrOffsIdxA[e]];
                        int3 coordB = coord + CornerOffs[CrnrOffsIdxB[e]];

                        sbyte rawA = Volume[CoordToIndex(coordA)];
                        sbyte rawB = Volume[CoordToIndex(coordB)];
                        bool emptyA = rawA == sbyte.MinValue;
                        bool emptyB = rawB == sbyte.MinValue;
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

                    Vertex vert = new()
                    {
                        pos = pos,
                        norm = norm,
                        color = new Color32(128, 128, 128, 255)
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

        [BurstCompile]
        private struct IndexJob : IJob
        {
            [ReadOnly] public NativeArray<sbyte> Volume;
            [ReadOnly] public int3 VoxCount;
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
                return Volume[Flatten(c)] / SbyteMax;
            }
        }

        public async Task<bool> CreateMesh(
            NativeArray<sbyte> volume, int3 voxCount, float voxSize,
            Mesh mesh, CancellationToken ctkn = default)
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
                var vertJob = new VertexJob
                {
                    Volume = volume,
                    VoxCount = voxCount,
                    VoxSize = voxSize,
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

                int maxIndices = _verts.Length * 18;
                if (_tris.Capacity < maxIndices)
                    _tris.Capacity = maxIndices;

                var triJob = new IndexJob
                {
                    Volume = volume,
                    VoxCount = voxCount,
                    VertCoords = _vertCoords,
                    CoordVertMap = _coordVertMap,
                    Tris = _tris
                };

                JobHandle triHandle = triJob.Schedule();
                while (!triHandle.IsCompleted)
                    await Awaitable.NextFrameAsync(ctkn);
                triHandle.Complete();
                ctkn.ThrowIfCancellationRequested();

                MinMaxAABB b = _boundsRef.Value;
                float3 size = b.Max - b.Min;
                Bounds bounds = new(b.Center, size);
                ApplyToMesh(_verts.AsArray(), _tris.AsArray(), bounds, mesh);
                hasTriangles = _tris.Length > 0;
            }
            finally
            {
                _busy = false;
            }

            return hasTriangles;
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
        }
    }
}
