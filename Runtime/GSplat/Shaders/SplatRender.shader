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

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GaussianData
            {
                float3 pos;
                float3 conic;   // inverse 2D covariance
                float3 color;   // RGB (SH evaluated, clamped)
                float  opacity; // sigmoid
                float  radius;  // screen-space radius
            };

            StructuredBuffer<GaussianData> _SplatData;
            StructuredBuffer<uint>         _SplatIndices; // sorted by depth
            uint _SplatCount;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 quadUV : TEXCOORD0;
                float3 color : TEXCOORD1;
                float3 conic : TEXCOORD2;
                float  opacity : TEXCOORD3;
                float2 center : TEXCOORD4;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint quadVert = vertID % 4;
                uint splatIdx = vertID / 4;
                if (splatIdx >= _SplatCount)
                {
                    OUT.positionHCS = float4(0, 0, -1, 0);
                    return OUT;
                }

                uint idx = _SplatIndices[splatIdx];
                GaussianData g = _SplatData[idx];

                // Quad corners in clip space around the projected center
                float4 clipPos = TransformWorldToHClip(g.pos);
                float2 screenCenter = clipPos.xy / clipPos.w;

                float r = g.radius * 2.0 / _ScreenParams.xy; // normalized screen radius

                float2 offsets[4] = {
                    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
                };
                float2 off = offsets[quadVert] * r;
                OUT.positionHCS = float4((screenCenter + off) * clipPos.w, clipPos.z, clipPos.w);
                OUT.quadUV = offsets[quadVert] * g.radius;
                OUT.center = float2(0, 0);
                OUT.color = g.color;
                OUT.conic = g.conic;
                OUT.opacity = g.opacity;

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 delta = IN.quadUV - IN.center;
                float sigma = 0.5 * (IN.conic.x * delta.x * delta.x +
                                      IN.conic.z * delta.y * delta.y) +
                              IN.conic.y * delta.x * delta.y;

                if (sigma < 0 || sigma >= 5.55) discard;

                float alpha = min(0.999, IN.opacity * exp(-sigma));
                if (alpha < 1.0 / 255.0) discard;

                return half4(IN.color, alpha);
            }
            ENDHLSL
        }
    }
}
