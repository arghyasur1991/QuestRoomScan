Shader "Hidden/Genesis/TriplanarReloc"
{
    Properties
    {
        _MainTex ("", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" }

        // ── Pass 0: Forward-splat relocation ──
        // Each old texel has its exact 3D position from (UV + stored depth).
        // The vertex shader applies relocation R, computes the new triplanar UV,
        // and positions the point there. Rendered as a point cloud.
        Pass
        {
            ZTest Always ZWrite Off Cull Off

            HLSLPROGRAM
            #pragma vertex vertSplat
            #pragma fragment fragSplat

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_OldTri);
            TEXTURE2D(_OldDepth);
            SAMPLER(sampler_point_clamp);

            float4x4 _Reloc;
            int      _SrcFace;    // source face: 0=XZ, 1=XY, 2=YZ
            int      _DstFace;    // destination face being rendered to
            float3   _VoxCount;
            float    _VoxSize;
            float2   _TexSize;

            float2 SignedTriUV(float2 baseUV, float normalComponent)
            {
                return float2(baseUV.x, normalComponent > 0 ? baseUV.y * 0.5 + 0.5 : baseUV.y * 0.5);
            }

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                half4  color       : COLOR;
            };

            Varyings vertSplat(uint vertexID : SV_VertexID)
            {
                Varyings OUT;
                OUT.positionHCS = float4(-2, -2, 0, 1);
                OUT.color = half4(0, 0, 0, 0);

                uint w = (uint)_TexSize.x;
                uint h = (uint)_TexSize.y;
                uint px = vertexID % w;
                uint py = vertexID / w;
                if (py >= h) return OUT;

                float2 uv = ((float2)uint2(px, py) + 0.5) / _TexSize;

                half4 oldColor = SAMPLE_TEXTURE2D_LOD(_OldTri, sampler_point_clamp, uv, 0);
                if (oldColor.a < 0.01) return OUT;

                float depth = SAMPLE_TEXTURE2D_LOD(_OldDepth, sampler_point_clamp, uv, 0).r;

                float s     = uv.y >= 0.5 ? 1.0 : -1.0;
                float baseV = uv.y >= 0.5 ? (uv.y - 0.5) * 2.0 : uv.y * 2.0;

                float3 oldUVW;
                float3 faceN;
                if (_SrcFace == 0)      { oldUVW = float3(uv.x, depth, baseV); faceN = float3(0, s, 0); }
                else if (_SrcFace == 1) { oldUVW = float3(uv.x, baseV, depth); faceN = float3(0, 0, s); }
                else                    { oldUVW = float3(depth, uv.x, baseV); faceN = float3(s, 0, 0); }

                float3 vc = _VoxCount;
                float  vs = _VoxSize;
                float3 oldWorldPos = (oldUVW * vc - vc * 0.5) * vs;

                float3 newWorldPos = mul(_Reloc, float4(oldWorldPos, 1)).xyz;
                float3 newN = normalize(mul((float3x3)_Reloc, faceN));

                float3 newLocal = newWorldPos / vs + vc * 0.5;
                if (any(newLocal < 0.0) || any(newLocal > vc))
                    return OUT;
                float3 newUVW = saturate(newLocal / vc);

                float3 absN = abs(newN);
                float2 newTriUV;
                int newFace;
                if (absN.y >= absN.x && absN.y >= absN.z)
                {
                    newTriUV = SignedTriUV(newUVW.xz, newN.y);
                    newFace = 0;
                }
                else if (absN.z >= absN.x)
                {
                    newTriUV = SignedTriUV(newUVW.xy, newN.z);
                    newFace = 1;
                }
                else
                {
                    newTriUV = SignedTriUV(newUVW.yz, newN.x);
                    newFace = 2;
                }

                if (newFace != _DstFace)
                    return OUT;

                float2 clipXY = newTriUV * 2.0 - 1.0;
                #if UNITY_UV_STARTS_AT_TOP
                    clipXY.y = -clipXY.y;
                #endif
                OUT.positionHCS = float4(clipXY, 0.5, 1.0);
                OUT.color = half4(oldColor.rgb, oldColor.a);
                return OUT;
            }

            half4 fragSplat(Varyings IN) : SV_Target
            {
                return IN.color;
            }
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
