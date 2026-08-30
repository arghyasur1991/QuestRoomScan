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
int gsUseRoomAabb;
float3 gsRoomAabbMin;
float3 gsRoomAabbMax;
float gsRoomClampMax;

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
    if (gsUseRoomAabb != 0)
    {
        if (any(worldPos < gsRoomAabbMin) || any(worldPos > gsRoomAabbMax))
            return false;
    }
    for (int i = 0; i < gsNumRoomClipPlanes; i++)
    {
        float4 pl = gsRoomClipPlanes[i];
        if (dot(worldPos, pl.xyz) < pl.w)
            return false;
    }
    return true;
}

// Project a near-miss point onto the clip hull. Far AABB rejects stay cheap
// (hallway). A second plane pass runs only when the first hit a plane.
bool gsClampToRoom(inout float3 worldPos)
{
    if (gsConfineToRoom == 0 || gsNumRoomClipPlanes <= 0) return true;
    if (gsUseRoomAabb != 0)
    {
        if (any(worldPos < gsRoomAabbMin) || any(worldPos > gsRoomAabbMax))
            return false;
    }
    float3 orig = worldPos;
    int hit = 0;
    for (int i = 0; i < gsNumRoomClipPlanes; i++)
    {
        float4 pl = gsRoomClipPlanes[i];
        float d = pl.w - dot(worldPos, pl.xyz);
        if (d > 0)
        {
            worldPos += pl.xyz * d;
            hit = 1;
        }
    }
    if (hit == 0) return true;
    for (int j = 0; j < gsNumRoomClipPlanes; j++)
    {
        float4 pl = gsRoomClipPlanes[j];
        float d = (pl.w + 1e-3) - dot(worldPos, pl.xyz);
        if (d > 0)
            worldPos += pl.xyz * d;
    }
    float3 delta = worldPos - orig;
    float maxD = gsRoomClampMax;
    return dot(delta, delta) <= maxD * maxD;
}

// Depth through a closed wall overshoots along the view ray. Snap it onto
// the first exit plane. Overshoot past gsRoomClampMax is a real outside
// hit (open door / hallway) — skip so we do not carve or fake-fill.
bool gsClampDepthRayToRoom(float3 eyePos, inout float3 depthPos)
{
    if (gsConfineToRoom == 0 || gsNumRoomClipPlanes <= 0) return true;
    float3 delta = depthPos - eyePos;
    float distSq = dot(delta, delta);
    if (distSq < 1e-8) return true;
    float dist = sqrt(distSq);
    float3 dir = delta / dist;
    float t = dist;
    for (int i = 0; i < gsNumRoomClipPlanes; i++)
    {
        float4 pl = gsRoomClipPlanes[i];
        float nd = dot(dir, pl.xyz);
        if (nd >= -1e-5) continue;
        float tHit = (pl.w - dot(eyePos, pl.xyz)) / nd;
        if (tHit >= 0)
            t = min(t, tHit);
    }
    float over = dist - t;
    if (over > gsRoomClampMax) return false;
    if (over > 0)
        depthPos = eyePos + dir * t;
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
