Shader "Hidden/Genesis/TriplanarReloc"
{
    Properties
    {
        _MainTex ("", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            ZTest Always ZWrite Off Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_OldTriXZ); TEXTURE2D(_OldTriXY); TEXTURE2D(_OldTriYZ);
            SAMPLER(sampler_linear_clamp);

            float4x4 _InvReloc;
            int      _Face;
            float3   _VoxCount;
            float    _VoxSize;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv          : TEXCOORD0;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = IN.uv;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 uv = IN.uv;

                float s     = uv.y >= 0.5 ? 1.0 : -1.0;
                float baseV = uv.y >= 0.5 ? (uv.y - 0.5) * 2.0 : uv.y * 2.0;

                float3 uvw, faceN;
                if (_Face == 0)      { uvw = float3(uv.x, 0.5, baseV); faceN = float3(0, s, 0); }
                else if (_Face == 1) { uvw = float3(uv.x, baseV, 0.5); faceN = float3(0, 0, s); }
                else                 { uvw = float3(0.5, uv.x, baseV); faceN = float3(s, 0, 0); }

                float3 vc = _VoxCount;
                float  vs = _VoxSize;
                float3 worldPos = (uvw * vc - vc * 0.5) * vs;

                float3 oldPos = mul(_InvReloc, float4(worldPos, 1)).xyz;
                float3 oldN   = normalize(mul((float3x3)_InvReloc, faceN));

                float3 oldLocal = oldPos / vs + vc * 0.5;
                if (any(oldLocal < 0.5) || any(oldLocal > vc - 0.5))
                    return half4(0, 0, 0, 0);

                float3 oldUVW  = saturate(oldLocal / vc);
                float3 absOldN = abs(oldN);
                float2 oldTriUV;
                half4  color;

                if (absOldN.y >= absOldN.x && absOldN.y >= absOldN.z)
                {
                    oldTriUV = float2(oldUVW.x, oldN.y > 0 ? oldUVW.z * 0.5 + 0.5 : oldUVW.z * 0.5);
                    color = SAMPLE_TEXTURE2D(_OldTriXZ, sampler_linear_clamp, oldTriUV);
                }
                else if (absOldN.z >= absOldN.x)
                {
                    oldTriUV = float2(oldUVW.x, oldN.z > 0 ? oldUVW.y * 0.5 + 0.5 : oldUVW.y * 0.5);
                    color = SAMPLE_TEXTURE2D(_OldTriXY, sampler_linear_clamp, oldTriUV);
                }
                else
                {
                    oldTriUV = float2(oldUVW.y, oldN.x > 0 ? oldUVW.z * 0.5 + 0.5 : oldUVW.z * 0.5);
                    color = SAMPLE_TEXTURE2D(_OldTriYZ, sampler_linear_clamp, oldTriUV);
                }

                return color;
            }
            ENDHLSL
        }
    }
}
