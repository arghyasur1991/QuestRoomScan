Shader "Genesis/ScanMeshVertexColor"
{
    Properties
    {
        _Smoothness ("Smoothness", Range(0,1)) = 0.3
        [Toggle(_DEBUG_SOLID)] _DebugSolid ("Debug Solid Color", Float) = 0
        [Toggle(_SHOW_NORMALS)] _ShowNormals ("Show Normals", Float) = 0
        [Toggle(_TRIPLANAR_ONLY)] _TriplanarOnly ("Triplanar Only (persistence eval)", Float) = 0
        [Toggle(_VERTEX_ONLY)] _VertexOnly ("Vertex Colors Only", Float) = 0
        [Toggle(_LIT)] _Lit ("Lit (use Unity lighting)", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        Pass
        {
            Name "VertexColorForward"
            Tags { "LightMode"="UniversalForwardOnly" }
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma shader_feature_local _DEBUG_SOLID
            #pragma shader_feature_local _SHOW_NORMALS
            #pragma shader_feature_local _TRIPLANAR_ONLY
            #pragma shader_feature_local _VERTEX_ONLY
            #pragma multi_compile_local _ _LIT
            #pragma multi_compile _ _ADDITIONAL_LIGHTS

            #ifdef _LIT
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #else
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #endif

            CBUFFER_START(UnityPerMaterial)
                float _Smoothness;
                float _DebugSolid;
                float _ShowNormals;
                float _TriplanarOnly;
                float _VertexOnly;
                float _Lit;
            CBUFFER_END

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;

            half4 UnpackColor(uint packed)
            {
                return half4(
                    (packed        & 0xFF) / 255.0h,
                    ((packed >> 8) & 0xFF) / 255.0h,
                    ((packed >> 16)& 0xFF) / 255.0h,
                    ((packed >> 24)& 0xFF) / 255.0h);
            }

            TEXTURE2D_ARRAY(_RSKeyframeTex);
            SAMPLER(sampler_RSKeyframeTex);
            float4 _RSKeyframeData[112];
            int _RSKeyframeCount;
            float _RSCamExposure;

            TEXTURE2D(_RSTriXZ);  SAMPLER(sampler_RSTriXZ);
            TEXTURE2D(_RSTriXY);  SAMPLER(sampler_RSTriXY);
            TEXTURE2D(_RSTriYZ);  SAMPLER(sampler_RSTriYZ);
            float _RSTriAvailable;

            float4 gsVoxCount;
            float gsVoxSize;

            float3 WorldToVoxelUVW(float3 pos)
            {
                pos /= gsVoxSize;
                pos += gsVoxCount.xyz / 2.0;
                pos /= gsVoxCount.xyz;
                return saturate(pos);
            }

            float2 ProjectToKeyframeUV(float3 worldPos, int idx)
            {
                int o = idx * 7;
                float3 camPos = _RSKeyframeData[o + 0].xyz;

                float4x4 invRot;
                invRot[0] = _RSKeyframeData[o + 1];
                invRot[1] = _RSKeyframeData[o + 2];
                invRot[2] = _RSKeyframeData[o + 3];
                invRot[3] = _RSKeyframeData[o + 4];

                float4 intrA = _RSKeyframeData[o + 5];
                float2 focalLen = intrA.xy;
                float2 principalPt = intrA.zw;

                float4 intrB = _RSKeyframeData[o + 6];
                float2 sensorRes = intrB.xy;
                float2 currentRes = intrB.zw;

                float3 localPos = mul((float3x3)invRot, worldPos - camPos);
                if (localPos.z <= 0.001) return float2(-1, -1);

                float2 sensorPt = float2(
                    (localPos.x / localPos.z) * focalLen.x + principalPt.x,
                    (localPos.y / localPos.z) * focalLen.y + principalPt.y);

                float2 scaleFactor = currentRes / sensorRes;
                scaleFactor /= max(scaleFactor.x, scaleFactor.y);
                float2 cropMin = sensorRes * (1.0 - scaleFactor) * 0.5;
                float2 cropSize = sensorRes * scaleFactor;
                return (sensorPt - cropMin) / cropSize;
            }

            float2 SignedTriUV(float2 baseUV, float normalComponent)
            {
                return float2(baseUV.x, normalComponent > 0 ? baseUV.y * 0.5 + 0.5 : baseUV.y * 0.5);
            }

            half3 SampleTriplanar(float3 worldPos, float3 normal)
            {
                float3 absN = abs(normal);
                float3 blend = absN / (absN.x + absN.y + absN.z + 0.001);
                float3 uvw = WorldToVoxelUVW(worldPos);

                float2 uvXZ = SignedTriUV(uvw.xz, normal.y);
                float2 uvXY = SignedTriUV(uvw.xy, normal.z);
                float2 uvYZ = SignedTriUV(uvw.yz, normal.x);

                half4 colXZ = SAMPLE_TEXTURE2D(_RSTriXZ, sampler_RSTriXZ, uvXZ);
                half4 colXY = SAMPLE_TEXTURE2D(_RSTriXY, sampler_RSTriXY, uvXY);
                half4 colYZ = SAMPLE_TEXTURE2D(_RSTriYZ, sampler_RSTriYZ, uvYZ);

                half3 rgb = colXZ.rgb * blend.y + colXY.rgb * blend.z + colYZ.rgb * blend.x;
                half totalAlpha = colXZ.a * blend.y + colXY.a * blend.z + colYZ.a * blend.x;

                return totalAlpha > 0.01 ? rgb : half3(-1, -1, -1);
            }

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float4 color : COLOR;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint idx = _SurfaceIndices[vertID];
                GPUVertex gv = _SurfaceVerts[idx];

                OUT.positionWS  = gv.pos;
                OUT.positionHCS = TransformWorldToHClip(gv.pos);
                OUT.normalWS    = gv.norm;
                OUT.color       = UnpackColor(gv.packedColor);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                #ifdef _DEBUG_SOLID
                return half4(1, 0.2, 0.1, 1);
                #endif

                float3 normal = normalize(IN.normalWS);

                #ifdef _SHOW_NORMALS
                return half4(normal * 0.5 + 0.5, 1);
                #endif

                half3 baseColor = IN.color.rgb;

                #ifndef _VERTEX_ONLY
                {
                    bool resolved = false;

                    #ifndef _TRIPLANAR_ONLY
                    if (_RSKeyframeCount > 0)
                    {
                        float bestScore = -1;
                        float2 bestUV = float2(0, 0);
                        int bestIdx = -1;

                        [loop] for (int i = 0; i < _RSKeyframeCount; i++)
                        {
                            float2 uv = ProjectToKeyframeUV(IN.positionWS, i);
                            if (uv.x < 0.02 || uv.x > 0.98 || uv.y < 0.02 || uv.y > 0.98)
                                continue;

                            float3 camPos = _RSKeyframeData[i * 7 + 0].xyz;
                            float3 viewDir = normalize(IN.positionWS - camPos);
                            float score = -dot(viewDir, normal);
                            if (score > bestScore)
                            {
                                bestScore = score;
                                bestUV = uv;
                                bestIdx = i;
                            }
                        }

                        if (bestIdx >= 0)
                        {
                            half3 c = SAMPLE_TEXTURE2D_ARRAY(
                                _RSKeyframeTex, sampler_RSKeyframeTex, bestUV, bestIdx).rgb;
                            baseColor = saturate(c * _RSCamExposure);
                            resolved = true;
                        }
                    }
                    #endif

                    if (!resolved && _RSTriAvailable > 0.5)
                    {
                        half3 tri = SampleTriplanar(IN.positionWS, normal);
                        if (tri.r >= 0)
                            baseColor = tri;
                    }
                }
                #endif

                #ifdef _LIT
                {
                    half3 lighting = half3(0, 0, 0);

                    Light mainLight = GetMainLight();
                    lighting += mainLight.color * saturate(dot(normal, mainLight.direction));

                    uint addCount = GetAdditionalLightsCount();
                    for (uint li = 0u; li < addCount; li++)
                    {
                        Light addLight = GetAdditionalLight(li, IN.positionWS);
                        lighting += addLight.color * addLight.distanceAttenuation
                                  * saturate(dot(normal, addLight.direction));
                    }

                    half3 ambient = SampleSH(normal);
                    baseColor *= (lighting + ambient);
                }
                #endif

                return half4(baseColor, 1);
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

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                uint idx = _SurfaceIndices[vertID];
                OUT.positionHCS = TransformWorldToHClip(_SurfaceVerts[idx].pos);
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

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                uint idx = _SurfaceIndices[vertID];
                GPUVertex gv = _SurfaceVerts[idx];
                OUT.positionHCS = TransformWorldToHClip(gv.pos);
                OUT.normalWS    = gv.norm;
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
