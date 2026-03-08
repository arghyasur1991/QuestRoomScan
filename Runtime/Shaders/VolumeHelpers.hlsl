// Genesis RoomScan - Volume/voxel utility functions

SamplerState gsVolLinearClampSampler;
SamplerState gsVolPointClampSampler;

Texture3D<float> gsVolume;
uint3 gsVoxCount;
float gsVoxSize;
float gsVoxDist;
float gsVoxMin;
StructuredBuffer<float3> gsFrustumVolume;

Texture2D<float4> gsDilatedDepth;

int gsNumExclusions;
float3 gsExclusionHeads[64];

#define GS_EMPTY_VOXEL -1.0

float3 gsVoxelToWorld(uint3 indices)
{
    float3 pos = (float3)indices;
    pos += 0.5;
    pos -= (float3)gsVoxCount / 2.0;
    pos *= gsVoxSize;
    return pos;
}

float3 gsWorldToVoxelFloat(float3 pos)
{
    pos /= gsVoxSize;
    pos += (float3)gsVoxCount / 2.0;
    return pos;
}

uint3 gsWorldToVoxel(float3 pos)
{
    pos = gsWorldToVoxelFloat(pos);
    uint3 id = (uint3)floor(pos);
    id = clamp(id, uint3(0, 0, 0), gsVoxCount);
    return id;
}

float3 gsWorldToVoxelUVW(float3 pos)
{
    pos = gsWorldToVoxelFloat(pos);
    pos /= (float3)gsVoxCount;
    return saturate(pos);
}

float gsSampleDilatedDepth(float2 uv)
{
    return gsDilatedDepth.SampleLevel(gsVolPointClampSampler, uv, 0).z;
}
