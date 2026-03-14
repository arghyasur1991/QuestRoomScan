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
            Blend SrcAlpha OneMinusSrcAlpha
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct SplatViewData
            {
                float3 worldPos;
                float viewDepth;
                float2 axis1;
                float2 axis2;
                uint2 color;
            };

            StructuredBuffer<SplatViewData> _SplatViewData;
            uint _SplatCount;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 quadUV : TEXCOORD0;
                half4 col : COLOR0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings o = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                uint quadVert = vertID % 4;
                uint splatIdx = vertID / 4;

                if (splatIdx >= _SplatCount)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

                SplatViewData view = _SplatViewData[splatIdx];

                if (view.viewDepth <= 0)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

                // Unpack fp16 color
                o.col.r = f16tof32(view.color.x >> 16);
                o.col.g = f16tof32(view.color.x);
                o.col.b = f16tof32(view.color.y >> 16);
                o.col.a = f16tof32(view.color.y);

                // Quad [-2, 2] mapped to quad corners
                float2 quadPos = float2(quadVert & 1, (quadVert >> 1) & 1) * 2.0 - 1.0;
                quadPos *= 2.0;
                o.quadUV = quadPos;

                // Clip-space center from world position (Unity handles all platform conventions)
                float4 centerClip = TransformWorldToHClip(view.worldPos);
                if (centerClip.w <= 0)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

                // Offset quad corners using precomputed screen-space axes
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

                return half4(i.col.rgb, alpha);
            }
            ENDHLSL
        }
    }
}
