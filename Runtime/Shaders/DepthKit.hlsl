// Genesis RoomScan - Depth utility functions
// Adapted from Anaglyph/lasertag DepthKit (MIT)

Texture2DArray<float> gsDepthTex;
Texture2DArray<float4> gsDepthNormalTex;
uniform uint2 gsDepthTexSize;

SamplerState gsBilinearClampSampler;
SamplerState gsPointClampSampler;

uniform float4x4 gsDepthProj[2];
uniform float4x4 gsDepthProjInv[2];
uniform float4x4 gsDepthView[2];
uniform float4x4 gsDepthViewInv[2];

uniform float4 gsDepthZParams; // (near, far, 0, 0)

float3 gsDepthEyePos(int eye = 0)
{
    return gsDepthViewInv[eye]._m03_m13_m23;
}

float gsDepthSample(float2 uv, int eye = 0)
{
    return gsDepthTex.SampleLevel(gsPointClampSampler, float3(uv, eye), 0);
}

float gsDepthNDCToLinear(float depthNDC, int eye = 0)
{
    float z = depthNDC * 2.0 - 1.0;
    float A = gsDepthProj[eye][2][2];
    float B = gsDepthProj[eye][2][3];
    return abs(B / (z + A));
}

float4 gsDepthNormalSample(float2 uv, int eye = 0)
{
    return gsDepthNormalTex.SampleLevel(gsPointClampSampler, float3(uv, eye), 0);
}

float4 gsDepthWorldToHCS(float3 worldPos, int eye = 0)
{
    return mul(gsDepthProj[eye], mul(gsDepthView[eye], float4(worldPos, 1)));
}

float4 gsDepthHCStoWorldH(float4 hcs, int eye = 0)
{
    return mul(gsDepthViewInv[eye], mul(gsDepthProjInv[eye], hcs));
}

float3 gsDepthHCStoNDC(float4 hcs)
{
    return (hcs.xyz / hcs.w) * 0.5 + 0.5;
}

float4 gsDepthNDCtoHCS(float3 ndc)
{
    return float4(ndc * 2.0 - 1.0, 1);
}

float3 gsDepthWorldToNDC(float3 worldPos, int eye = 0)
{
    float4 hcs = gsDepthWorldToHCS(worldPos, eye);
    return gsDepthHCStoNDC(hcs);
}

float3 gsDepthNDCtoWorld(float3 ndc, int eye = 0)
{
    float4 hcs = gsDepthNDCtoHCS(ndc);
    float4 worldH = gsDepthHCStoWorldH(hcs, eye);
    return worldH.xyz / worldH.w;
}
