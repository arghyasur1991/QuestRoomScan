Shader "Genesis/SplatComposite"
{
    SubShader
    {
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc
#pragma require 2darray

#pragma multi_compile_local _ UNITY_SINGLE_PASS_STEREO STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

#include "UnityCG.cginc"

struct v2f
{
    float4 vertex : SV_POSITION;
};

v2f vert (uint vtxID : SV_VertexID)
{
    v2f o;
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
    o.vertex = float4(quadPos, 1, 1);
    return o;
}

#if defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON) || defined(STEREO_MULTIVIEW_ON)
UNITY_DECLARE_TEX2DARRAY(_GaussianSplatRT);
#else
Texture2D _GaussianSplatRT;
#endif

int _CustomStereoEyeIndex;

half4 frag (v2f i) : SV_Target
{
    half4 col;
    #if defined(UNITY_SINGLE_PASS_STEREO) || defined(STEREO_INSTANCING_ON) || defined(STEREO_MULTIVIEW_ON)
        float2 normalizedUV = float2(i.vertex.x / _ScreenParams.x, i.vertex.y / _ScreenParams.y);
        col = UNITY_SAMPLE_TEX2DARRAY(_GaussianSplatRT, float3(normalizedUV, _CustomStereoEyeIndex));
    #else
        col = _GaussianSplatRT.Load(int3(i.vertex.xy, 0));
    #endif

    col.rgb = GammaToLinearSpace(col.rgb);
    col.a = saturate(col.a * 1.5);
    return col;
}
ENDCG
        }
    }
}
