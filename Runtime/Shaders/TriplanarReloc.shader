Shader "Hidden/Genesis/TriplanarReloc"
{
    Properties
    {
        _MainTex ("", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" }

        // ── Pass 0: Reserved (unused, kept for index stability) ──
        Pass
        {
            ZTest Always ZWrite Off Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            float4 vert(float4 pos : POSITION) : SV_POSITION { return float4(-2,-2,0,1); }
            half4 frag() : SV_Target { return 0; }
            ENDHLSL
        }

        // ── Pass 1: Dilation ──
        // Fill empty texels by averaging non-empty 3x3 neighbors.
        Pass
        {
            ZTest Always ZWrite Off Cull Off

            HLSLPROGRAM
            #pragma vertex vertDilate
            #pragma fragment fragDilate

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_point_clamp);
            float4 _MainTex_TexelSize;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv          : TEXCOORD0;
            };

            Varyings vertDilate(float4 positionOS : POSITION, float2 uv : TEXCOORD0)
            {
                Varyings OUT;
                OUT.positionHCS = TransformObjectToHClip(positionOS.xyz);
                OUT.uv = uv;
                return OUT;
            }

            half4 fragDilate(Varyings IN) : SV_Target
            {
                float2 uv = IN.uv;
                half4 center = SAMPLE_TEXTURE2D(_MainTex, sampler_point_clamp, uv);

                if (center.a > 0.01)
                    return center;

                float2 ts = _MainTex_TexelSize.xy;
                half3 sum = half3(0, 0, 0);
                half count = 0;

                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    half4 s = SAMPLE_TEXTURE2D(_MainTex, sampler_point_clamp,
                                               uv + float2(dx, dy) * ts);
                    if (s.a > 0.01)
                    {
                        sum += s.rgb;
                        count += 1.0;
                    }
                }

                if (count < 0.5)
                    return half4(0, 0, 0, 0);

                return half4(sum / count, 0.5);
            }
            ENDHLSL
        }
    }
}
