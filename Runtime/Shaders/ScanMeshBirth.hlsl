// Per-frame live-mesh birth: extract only stamps an age tick; the
// forward pass fills the time between dumps with a smoothstep fade
// and a short grow along the vertex normal.
#ifndef SCAN_MESH_BIRTH_HLSL
#define SCAN_MESH_BIRTH_HLSL

float _RSBirthFadeSec;
float _RSBirthGrow;
float _RSExtractTime;
float _RSExtractInterval;

float ScanMeshBirthFade(half packedAlpha, uint voxelFlatIdx)
{
    if (_RSBirthFadeSec < 0.001)
        return 1.0;

    float extracts = (float)packedAlpha * 255.0;
    float interval = max(_RSExtractInterval, 0.02);
    float age = extracts * interval + (_Time.y - _RSExtractTime);
    float h = frac(sin((float)voxelFlatIdx * 12.9898 + 78.233) * 43758.5453);
    age -= h * 0.1;
    return saturate(smoothstep(0.0, 1.0, age / _RSBirthFadeSec));
}

float3 ScanMeshBirthDisplace(float3 pos, float3 norm, float fade)
{
    float grow = (1.0 - fade) * _RSBirthGrow;
    return pos - normalize(norm + 1e-5) * grow;
}

#endif
