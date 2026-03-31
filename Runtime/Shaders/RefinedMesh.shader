Shader "Genesis/RefinedMesh"
{
    Properties
    {
        _MainTex ("Atlas", 2D) = "white" {}
        _BumpMap ("Normal Map", 2D) = "bump" {}
        _NormalStrength ("Normal Strength", Float) = 0.0
        _LightDir ("Light Direction", Vector) = (0.3, 1.0, 0.2, 0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        Pass
        {
            Name "RefinedUnlit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 posWS  : POSITION;
                float3 normal : NORMAL;
                float4 tangent: TANGENT;
                float2 uv     : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 posCS    : SV_POSITION;
                float2 uv       : TEXCOORD0;
                float3 worldNml : TEXCOORD1;
                float3 worldTan : TEXCOORD2;
                float3 worldBit : TEXCOORD3;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);
            TEXTURE2D(_BumpMap);
            SAMPLER(sampler_BumpMap);

            float _NormalStrength;
            float4 _LightDir;

            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.posCS = TransformWorldToHClip(v.posWS.xyz);
                o.uv = v.uv;
                o.worldNml = v.normal;
                o.worldTan = v.tangent.xyz;
                o.worldBit = cross(v.normal, v.tangent.xyz) * v.tangent.w;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                half4 col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);

                half4 nSample = SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, i.uv);
                half3 tn;
                tn.xy = (nSample.rg * 2.0 - 1.0) * _NormalStrength;
                tn.z = sqrt(saturate(1.0 - dot(tn.xy, tn.xy)));

                float3 N = normalize(i.worldNml);
                float3 T = normalize(i.worldTan);
                float3 B = normalize(i.worldBit);
                float3 worldNormal = normalize(T * tn.x + B * tn.y + N * tn.z);

                float3 lightDir = normalize(_LightDir.xyz);
                half NdotL = abs(dot(worldNormal, lightDir));
                half lighting = NdotL * 0.4 + 0.6;

                col.rgb *= lighting;
                return col;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            ZWrite On
            ColorMask 0
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 posWS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 posCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.posCS = TransformWorldToHClip(v.posWS.xyz);
                return o;
            }

            half4 frag(Varyings i) : SV_Target { return 0; }
            ENDHLSL
        }
    }
    FallBack "Universal Render Pipeline/Unlit"
}
