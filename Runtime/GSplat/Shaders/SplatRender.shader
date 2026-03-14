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

            StructuredBuffer<float> _Means;       // [N*3] world xyz
            StructuredBuffer<float> _FeaturesDC;  // [N*3] SH DC coefficients
            StructuredBuffer<float> _Opacities;   // [N]   raw opacity (pre-sigmoid)
            StructuredBuffer<float> _Scales;      // [N*3] log-space scales
            uint _SplatCount;
            float _SplatSize;

            static const float SH_C0 = 0.28209479177387814;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 quadUV : TEXCOORD0;
                float3 color : TEXCOORD1;
                float  opacity : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float Sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

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

                float3 pos = float3(
                    _Means[splatIdx * 3],
                    _Means[splatIdx * 3 + 1],
                    _Means[splatIdx * 3 + 2]);

                float3 dc = float3(
                    _FeaturesDC[splatIdx * 3],
                    _FeaturesDC[splatIdx * 3 + 1],
                    _FeaturesDC[splatIdx * 3 + 2]);
                float3 col = saturate(SH_C0 * dc + 0.5);

                float rawOpacity = _Opacities[splatIdx];
                float alpha = Sigmoid(rawOpacity);

                float3 logScale = float3(
                    _Scales[splatIdx * 3],
                    _Scales[splatIdx * 3 + 1],
                    _Scales[splatIdx * 3 + 2]);
                float avgScale = (exp(logScale.x) + exp(logScale.y) + exp(logScale.z)) / 3.0;
                float splatWorld = max(avgScale * _SplatSize, 0.005);

                float4 clipPos = TransformWorldToHClip(pos);

                float2 offsets[4] = {
                    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
                };

                float3 camRight = UNITY_MATRIX_V[0].xyz;
                float3 camUp = UNITY_MATRIX_V[1].xyz;
                float3 worldOffset = (offsets[quadVert].x * camRight + offsets[quadVert].y * camUp) * splatWorld;
                float4 cornerClip = TransformWorldToHClip(pos + worldOffset);

                OUT.positionHCS = cornerClip;
                OUT.quadUV = offsets[quadVert];
                OUT.color = col;
                OUT.opacity = alpha;

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float d2 = dot(IN.quadUV, IN.quadUV);
                if (d2 > 1.0) discard;

                float falloff = exp(-4.0 * d2);
                float alpha = IN.opacity * falloff;
                if (alpha < 1.0 / 255.0) discard;

                return half4(IN.color, alpha);
            }
            ENDHLSL
        }
    }
}
