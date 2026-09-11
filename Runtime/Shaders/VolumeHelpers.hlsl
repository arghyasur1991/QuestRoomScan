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
float4 gsExclusionP0[64];
float4 gsExclusionP1[64];
int gsEraseBody;
float gsEraseMaxWeight;

int gsConfineToRoom;
int gsNumRoomClipPlanes;
float4 gsRoomClipPlanes[32];
int gsUseRoomAabb;
float3 gsRoomAabbMin;
float3 gsRoomAabbMax;

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

// Single exit on purpose: DXC's Vulkan path flattens early returns into a
// function-named temp and then warns it is "potentially uninitialized".
bool gsInsideRoom(float3 worldPos)
{
    bool inside = true;
    if (gsConfineToRoom != 0 && gsNumRoomClipPlanes > 0)
    {
        if (gsUseRoomAabb != 0
            && (any(worldPos < gsRoomAabbMin) || any(worldPos > gsRoomAabbMax)))
            inside = false;

        for (int i = 0; inside && i < gsNumRoomClipPlanes; i++)
        {
            float4 pl = gsRoomClipPlanes[i];
            if (dot(worldPos, pl.xyz) < pl.w)
                inside = false;
        }
    }
    return inside;
}

// True when worldPos lies within tol of something that cuts the surface by
// construction rather than by a hole: a room clip plane, the room AABB, or
// the edge of the voxel volume. Boundary edges here are cuts, not holes.
bool gsNearSurfaceCut(float3 worldPos, float tol)
{
    bool near = false;
    float3 half = (float3)gsVoxCount * 0.5 * gsVoxSize;
    if (any(abs(worldPos) > half - tol))
        near = true;
    if (!near && gsConfineToRoom != 0 && gsNumRoomClipPlanes > 0)
    {
        if (gsUseRoomAabb != 0
            && (any(worldPos < gsRoomAabbMin + tol) || any(worldPos > gsRoomAabbMax - tol)))
            near = true;
        for (int i = 0; !near && i < gsNumRoomClipPlanes; i++)
        {
            float4 pl = gsRoomClipPlanes[i];
            if (dot(worldPos, pl.xyz) - pl.w < tol)
                near = true;
        }
    }
    return near;
}

bool gsTryScreenStamp(float3 worldPos, out float sDistNorm)
{
    sDistNorm = 0;
    bool hit = false;
    for (int s = 0; !hit && s < gsNumScreenStamps; s++)
    {
        float3 d = worldPos - gsScreenCenter[s].xyz;
        float3 n = gsScreenInward[s].xyz;
        float sd = dot(d, n);
        if (abs(sd) <= gsScreenCenter[s].w
            && abs(dot(d, gsScreenAxis[s].xyz)) <= gsScreenInward[s].w
            && abs(dot(d, gsScreenBitangent[s].xyz)) <= gsScreenAxis[s].w)
        {
            sDistNorm = clamp(sd / gsVoxDist, -1.0, 1.0);
            hit = true;
        }
    }
    return hit;
}

// Capsule: segment p0.xyz–p1.xyz, radius p0.w. p1.w >= 0.5 marks
// hand/forearm capsules the optional eraser may clear. Single exit
// for the Vulkan DXC path (no early return).
bool gsInsideExclusionCapsule(float3 p, float4 a, float4 b)
{
    bool inside = false;
    float3 ab = b.xyz - a.xyz;
    float abLen2 = max(dot(ab, ab), 1e-8);
    float t = saturate(dot(p - a.xyz, ab) / abLen2);
    float3 c = a.xyz + t * ab;
    float3 d = p - c;
    if (dot(d, d) < a.w * a.w)
        inside = true;
    return inside;
}

bool gsInsideAnyExclusion(float3 p)
{
    bool inside = false;
    for (int i = 0; i < gsNumExclusions; i++)
    {
        if (gsInsideExclusionCapsule(p, gsExclusionP0[i], gsExclusionP1[i]))
            inside = true;
    }
    return inside;
}

bool gsInsideErasableExclusion(float3 p)
{
    bool inside = false;
    for (int i = 0; i < gsNumExclusions; i++)
    {
        if (gsExclusionP1[i].w >= 0.5
            && gsInsideExclusionCapsule(p, gsExclusionP0[i], gsExclusionP1[i]))
            inside = true;
    }
    return inside;
}
