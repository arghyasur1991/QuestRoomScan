Shader "Genesis/ScanMeshVertexColor"
{
    Properties
    {
        _Smoothness ("Smoothness", Range(0,1)) = 0.3
        [Toggle(_DEBUG_SOLID)] _DebugSolid ("Debug Solid Color", Float) = 0
        [Toggle(_SHOW_NORMALS)] _ShowNormals ("Show Normals", Float) = 0
        [Toggle(_VERTEX_ONLY)] _VertexOnly ("Vertex Colors Only", Float) = 0
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
            #pragma shader_feature_local _DEBUG_SOLID
            #pragma shader_feature_local _SHOW_NORMALS
            #pragma shader_feature_local _VERTEX_ONLY

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _Smoothness;
                float _DebugSolid;
                float _ShowNormals;
                float _VertexOnly;
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

            // Triplanar persistent textures
            TEXTURE2D(_RSTriXZ);  SAMPLER(sampler_RSTriXZ);
            TEXTURE2D(_RSTriXY);  SAMPLER(sampler_RSTriXY);
            TEXTURE2D(_RSTriYZ);  SAMPLER(sampler_RSTriYZ);
            float _RSTriAvailable;

            // TSDF volume (for freeze tint feedback)
            TEXTURE3D(gsVolume);
            SAMPLER(sampler_gsVolume);

            // Volume params (set by VolumeIntegrator as globals)
            float4 gsVoxCount;
            float gsVoxSize;

            float3 WorldToVoxelUVW(float3 pos)
            {
                pos /= gsVoxSize;
                pos += gsVoxCount.xyz / 2.0;
                pos /= gsVoxCount.xyz;
                return saturate(pos);
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

            bool IsVoxelFrozen(float3 worldPos)
            {
                float3 uvw = WorldToVoxelUVW(worldPos);
                float2 tsdf = SAMPLE_TEXTURE3D_LOD(gsVolume, sampler_gsVolume, uvw, 0).rg;
                return tsdf.g < 0;
            }

            half3 ApplyFreezeTint(half3 color, float3 worldPos)
            {
                if (IsVoxelFrozen(worldPos))
                    color = lerp(color, half3(0.3, 0.5, 0.9), 0.25);
                return color;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                #ifdef _DEBUG_SOLID
                return half4(ApplyFreezeTint(half3(1, 0.2, 0.1), IN.positionWS), 1);
                #endif

                float3 normal = normalize(IN.normalWS);

                #ifdef _SHOW_NORMALS
                return half4(ApplyFreezeTint(half3(normal * 0.5 + 0.5), IN.positionWS), 1);
                #endif

                #ifdef _VERTEX_ONLY
                return half4(ApplyFreezeTint(IN.color.rgb, IN.positionWS), 1);
                #endif

                // Priority 1: Triplanar persistent texture
                if (_RSTriAvailable > 0.5)
                {
                    half3 tri = SampleTriplanar(IN.positionWS, normal);
                    if (tri.r >= 0) return half4(ApplyFreezeTint(tri, IN.positionWS), 1);
                }

                // Priority 2: Vertex colors
                return half4(ApplyFreezeTint(IN.color.rgb, IN.positionWS), 1);
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
