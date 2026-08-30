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

int gsConfineToRoom;
int gsNumRoomClipPlanes;
float4 gsRoomClipPlanes[32];

int gsNumScreenStamps;
float4 gsScreenCenter[4];
float4 gsScreenInward[4];
float4 gsScreenAxis[4];
float4 gsScreenBitangent[4];

#define GS_EMPTY_VOXEL -1.0

float3 gsVoxelToWorld(uint3 indices)
{
    return ((float3)indices + 0.5 - (float3)gsVoxCount / 2.0) * gsVoxSize;
}

float3 gsWorldToVoxelFloat(float3 worldPos)
{
    return worldPos / gsVoxSize + (float3)gsVoxCount / 2.0;
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

bool gsInsideRoom(float3 worldPos)
{
    if (gsConfineToRoom == 0 || gsNumRoomClipPlanes <= 0) return true;
    for (int i = 0; i < gsNumRoomClipPlanes; i++)
    {
        float4 pl = gsRoomClipPlanes[i];
        if (dot(worldPos, pl.xyz) < pl.w)
            return false;
    }
    return true;
}

bool gsTryScreenStamp(float3 worldPos, out float sDistNorm)
{
    sDistNorm = 0;
    for (int s = 0; s < gsNumScreenStamps; s++)
    {
        float3 d = worldPos - gsScreenCenter[s].xyz;
        float3 n = gsScreenInward[s].xyz;
        float sd = dot(d, n);
        if (abs(sd) > gsScreenCenter[s].w) continue;
        if (abs(dot(d, gsScreenAxis[s].xyz)) > gsScreenInward[s].w) continue;
        if (abs(dot(d, gsScreenBitangent[s].xyz)) > gsScreenAxis[s].w) continue;
        sDistNorm = clamp(sd / gsVoxDist, -1.0, 1.0);
        return true;
    }
    return false;
}
