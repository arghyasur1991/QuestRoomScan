Shader "Genesis/ScanMeshVertexColor"
{
    Properties
    {
        _Smoothness ("Smoothness", Range(0,1)) = 0.3
        [Toggle(_DEBUG_SOLID)] _DebugSolid ("Debug Solid Color", Float) = 0
        [Toggle(_SHOW_NORMALS)] _ShowNormals ("Show Normals", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        Pass
        {
            Name "VertexColorUnlit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ DOTS_INSTANCING_ON
            #pragma shader_feature_local _DEBUG_SOLID
            #pragma shader_feature_local _SHOW_NORMALS

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Smoothness;
                float _DebugSolid;
                float _ShowNormals;
            CBUFFER_END

            TEXTURE2D(_RSCamTex);
            SAMPLER(sampler_RSCamTex);
            float3 _RSCamPos;
            float4x4 _RSCamInvRot;
            float2 _RSCamFocalLen;
            float2 _RSCamPrincipalPt;
            float2 _RSCamSensorRes;
            float2 _RSCamCurrentRes;
            float _RSCamExposure;
            float _RSCamAvailable;

            float2 ProjectToCameraUV(float3 worldPos)
            {
                float3 localPos = mul((float3x3)_RSCamInvRot, worldPos - _RSCamPos);
                if (localPos.z <= 0.001) return float2(-1, -1);

                float2 sensorPt = float2(
                    (localPos.x / localPos.z) * _RSCamFocalLen.x + _RSCamPrincipalPt.x,
                    (localPos.y / localPos.z) * _RSCamFocalLen.y + _RSCamPrincipalPt.y);

                float2 scaleFactor = _RSCamCurrentRes / _RSCamSensorRes;
                scaleFactor /= max(scaleFactor.x, scaleFactor.y);
                float2 cropMin = _RSCamSensorRes * (1.0 - scaleFactor) * 0.5;
                float2 cropSize = _RSCamSensorRes * scaleFactor;
                return (sensorPt - cropMin) / cropSize;
            }

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float4 color : COLOR;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_TRANSFER_INSTANCE_ID(IN, OUT);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                float3 ws = TransformObjectToWorld(IN.positionOS.xyz);
                OUT.positionHCS = TransformWorldToHClip(ws);
                OUT.positionWS = ws;
                OUT.normalWS = TransformObjectToWorldNormal(IN.normalOS);
                OUT.color = IN.color;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                #ifdef _DEBUG_SOLID
                return half4(1, 0.2, 0.1, 1);
                #endif

                #ifdef _SHOW_NORMALS
                float3 n = normalize(IN.normalWS);
                return half4(n * 0.5 + 0.5, 1);
                #endif

                if (_RSCamAvailable > 0.5)
                {
                    float2 uv = ProjectToCameraUV(IN.positionWS);
                    if (uv.x > 0.01 && uv.x < 0.99 && uv.y > 0.01 && uv.y < 0.99)
                    {
                        half3 camColor = SAMPLE_TEXTURE2D(_RSCamTex, sampler_RSCamTex, uv).rgb;
                        camColor = saturate(camColor * _RSCamExposure);
                        return half4(camColor, 1);
                    }
                }

                return half4(IN.color.rgb, 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            ZWrite On
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_TRANSFER_INSTANCE_ID(IN, OUT);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode"="DepthNormals" }
            ZWrite On

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_TRANSFER_INSTANCE_ID(IN, OUT);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.normalWS = TransformObjectToWorldNormal(IN.normalOS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                float3 n = normalize(IN.normalWS);
                return half4(n * 0.5 + 0.5, 1);
            }
            ENDHLSL
        }
    }
}
