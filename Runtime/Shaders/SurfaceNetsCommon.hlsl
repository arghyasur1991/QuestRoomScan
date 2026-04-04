// Shared Surface Nets utilities used by both SurfaceNetsExtract.compute (TSDF)
// and DensitySurfaceNets.compute (object reconstruction).
//
// Contains: GPUVertex struct, corner/edge topology tables, coordinate helpers,
// common buffer declarations, and shared kernel implementations
// (ClearCounters, BuildVertexDispatchArgs, BuildIndirectArgs).

#ifndef SURFACE_NETS_COMMON_HLSL
#define SURFACE_NETS_COMMON_HLSL

// ──────────────────────────────────────────────────────────────────
//  Common parameters (both pipelines need these)
// ──────────────────────────────────────────────────────────────────
uint3  _VoxCount;
float  _VoxSize;
uint   _MaxVertices;

// ──────────────────────────────────────────────────────────────────
//  GPU Vertex (32 bytes, matches StructuredBuffer in render shader)
// ──────────────────────────────────────────────────────────────────
struct GPUVertex
{
    float3 pos;
    float3 norm;
    uint   packedColor;
    uint   voxelFlatIdx;
};

// ──────────────────────────────────────────────────────────────────
//  Common buffers
// ──────────────────────────────────────────────────────────────────
RWStructuredBuffer<int>       _CoordVertMap;
RWStructuredBuffer<GPUVertex> _Vertices;
RWStructuredBuffer<uint>      _Indices;
RWStructuredBuffer<uint>      _Counters;         // [0] = vertex count, [1] = index count
RWStructuredBuffer<uint>      _DispatchArgs;     // [0,1,2] = indirect dispatch group counts
RWStructuredBuffer<uint>      _DrawIndirectArgs; // [0..4]  = DrawProceduralIndirect args

// ──────────────────────────────────────────────────────────────────
//  Corner / edge tables  (Surface Nets cell topology)
// ──────────────────────────────────────────────────────────────────
static const int3 kCornerOffs[8] =
{
    int3(0,0,0), int3(1,0,0), int3(1,0,1), int3(0,0,1),
    int3(0,1,0), int3(1,1,0), int3(1,1,1), int3(0,1,1)
};
static const uint kEdgeA[12] = { 0,1,2,3, 4,5,6,7, 0,1,2,3 };
static const uint kEdgeB[12] = { 1,2,3,0, 5,6,7,4, 4,5,6,7 };

// ──────────────────────────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────────────────────────
int Flatten(int3 c)
{
    return c.x + c.y * (int)_VoxCount.x + c.z * (int)_VoxCount.x * (int)_VoxCount.y;
}

int3 Unflatten(int idx)
{
    uint uidx = (uint)idx;
    uint sliceXY = _VoxCount.x * _VoxCount.y;
    uint z = uidx / sliceXY;
    uint rem = uidx - z * sliceXY;
    uint y = rem / _VoxCount.x;
    uint x = rem - y * _VoxCount.x;
    return int3(x, y, z);
}

float3 VoxelCoordToWorld(float3 c)
{
    return (c + 0.5 - (float3)_VoxCount / 2.0) * _VoxSize;
}

uint PackColor(float4 c)
{
    return (uint)(saturate(c.r) * 255.0)
         | ((uint)(saturate(c.g) * 255.0) << 8)
         | ((uint)(saturate(c.b) * 255.0) << 16)
         | (255u << 24);
}

// ──────────────────────────────────────────────────────────────────
//  Shared kernel implementations
// ──────────────────────────────────────────────────────────────────

void ClearCountersImpl()
{
    _Counters[0] = 0;
    _Counters[1] = 0;
}

void BuildVertexDispatchArgsImpl()
{
    uint vertCount = min(_Counters[0], _MaxVertices);
    _DispatchArgs[0] = (vertCount + 63) / 64;
    _DispatchArgs[1] = 1;
    _DispatchArgs[2] = 1;
}

void BuildIndirectArgsImpl()
{
    _DrawIndirectArgs[0] = _Counters[1]; // indexCount
    _DrawIndirectArgs[1] = 1;            // instanceCount
    _DrawIndirectArgs[2] = 0;            // startIndex
    _DrawIndirectArgs[3] = 0;            // baseVertex
    _DrawIndirectArgs[4] = 0;            // startInstance
}

#endif // SURFACE_NETS_COMMON_HLSL
