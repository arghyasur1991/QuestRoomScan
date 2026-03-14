Shader "Genesis/SplatRender"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" "Queue"="Transparent" }

        Pass
        {
            Name "SplatQuad"
            Tags { "LightMode"="SRPDefaultUnlit" }
            ZWrite Off
            ZTest LEqual
            // Back-to-front premultiplied alpha
            Blend One OneMinusSrcAlpha
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _SORT_RADIX

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct SplatViewData
            {
                float4 pos;      // clip-space center (per-eye)
                float2 axis1;    // screen-space 2D covariance axis
                float2 axis2;    // screen-space 2D covariance axis
                uint2 color;     // packed fp16 RGBA
            };

            StructuredBuffer<SplatViewData> _SplatViewData;
            uint _SplatCount;
            uint _EyeIndex;
            uint _IsStereo;

#ifdef _SORT_RADIX
            StructuredBuffer<uint> _OrderBuffer;
#else
            StructuredBuffer<uint2> _SortBuffer;  // .y = original index
#endif

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 quadUV : TEXCOORD0;
                nointerpolation half4 col : COLOR0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID, uint instID : SV_InstanceID)
            {
                Varyings o = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                uint splatIdx = instID;
                if (splatIdx >= _SplatCount)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

#ifdef _SORT_RADIX
                uint origIdx = _OrderBuffer[splatIdx];
#else
                uint origIdx = _SortBuffer[splatIdx].y;
#endif
                uint viewIdx = _IsStereo ? origIdx * 2 + _EyeIndex : origIdx;
                SplatViewData view = _SplatViewData[viewIdx];

                float4 centerClip = view.pos;
                if (centerClip.w <= 0)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

                o.col.r = f16tof32(view.color.x >> 16);
                o.col.g = f16tof32(view.color.x);
                o.col.b = f16tof32(view.color.y >> 16);
                o.col.a = f16tof32(view.color.y);

                float2 quadPos = float2(vertID & 1, (vertID >> 1) & 1) * 2.0 - 1.0;
                quadPos *= 2.0;
                o.quadUV = quadPos;

                float2 deltaScreen = (quadPos.x * view.axis1 + quadPos.y * view.axis2)
                                     * 2.0 / _ScreenParams.xy;
                o.positionHCS = centerClip;
                o.positionHCS.xy += deltaScreen * centerClip.w;

                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                float power = -dot(i.quadUV, i.quadUV);
                half alpha = (half)exp(power);
                alpha = saturate(alpha * i.col.a);
                if (alpha < 1.0h / 255.0h) discard;

                return half4(i.col.rgb * alpha, alpha);
            }
            ENDHLSL
        }
    }
}
