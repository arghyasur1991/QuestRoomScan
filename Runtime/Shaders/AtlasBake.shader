Shader "Hidden/Genesis/AtlasBake"
{
    Properties
    {
        _KeyframeTex ("Keyframe", 2D) = "black" {}
        _OcclusionTex ("Occlusion Linear Z", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }

        // ─────────────────────────────────────────────────────────
        //  Pass 0: Occlusion — render original mesh from keyframe
        //  camera, output linear camera-space Z to a color RT.
        //  This avoids depth buffer linearization complexity.
        // ─────────────────────────────────────────────────────────
        Pass
        {
            Name "OcclusionLinearZ"
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vertOcc
            #pragma fragment fragOcc

            float4x4 _OccViewProjMat;
            float4x4 _OccViewMat;

            struct OccAttr { float3 pos : POSITION; };
            struct OccVary
            {
                float4 posCS   : SV_POSITION;
                float  linearZ : TEXCOORD0;
            };

            OccVary vertOcc(OccAttr v)
            {
                OccVary o;
                o.posCS = mul(_OccViewProjMat, float4(v.pos, 1.0));
                o.linearZ = mul(_OccViewMat, float4(v.pos, 1.0)).z;
                return o;
            }

            half4 fragOcc(OccVary i) : SV_Target
            {
                return half4(i.linearZ, 0, 0, 1);
            }
            ENDHLSL
        }

        // ─────────────────────────────────────────────────────────
        //  Pass 1: UV-space bake — vertex positions are atlas UV
        //  coords mapped to [-1,1]. Fragment shader projects the
        //  interpolated world position into the keyframe image,
        //  checks occlusion, and writes color + score-as-depth.
        //  ZTest Greater ensures only the highest-scoring keyframe
        //  survives per texel (replaces CPU bestScore[] array).
        // ─────────────────────────────────────────────────────────
        Pass
        {
            Name "AtlasBake"
            ZWrite On
            ZTest Greater
            Cull Off

            HLSLPROGRAM
            #pragma vertex vertBake
            #pragma fragment fragBake

            sampler2D _KeyframeTex;
            sampler2D _OcclusionTex;

            float4x4 _ViewMat;
            float4   _CamPos;
            float    _Fx, _Fy, _Cx, _Cy;
            float    _ImgW, _ImgH;
            float    _CropX, _CropY;

            struct BakeAttr
            {
                float3 pos    : POSITION;
                float3 normal : NORMAL;
                float3 worldP : TEXCOORD1;
            };

            struct BakeVary
            {
                float4 posCS  : SV_POSITION;
                float3 worldP : TEXCOORD0;
                float3 normal : TEXCOORD1;
            };

            BakeVary vertBake(BakeAttr v)
            {
                BakeVary o;
                o.posCS = float4(v.pos.xy, 0.5, 1.0);
                o.worldP = v.worldP;
                o.normal = v.normal;
                return o;
            }

            void fragBake(BakeVary i,
                          out half4  outColor : SV_Target,
                          out float  outDepth : SV_Depth)
            {
                float3 worldP = i.worldP;
                float3 norm = normalize(i.normal);

                float3 camPt = mul(_ViewMat, float4(worldP, 1.0)).xyz;
                if (camPt.z <= 0.001)
                    discard;

                float invZ = 1.0 / camPt.z;
                float screenX = _Fx * camPt.x * invZ + _Cx - _CropX;
                float screenY = _Fy * camPt.y * invZ + _Cy - _CropY;

                if (screenX < 0 || screenX >= _ImgW || screenY < 0 || screenY >= _ImgH)
                    discard;

                // Occlusion: projection renders screenY=0 at RT top, but
                // tex2D v=0 samples RT bottom, so flip Y for correct lookup
                float2 occUV = float2(screenX / _ImgW, 1.0 - screenY / _ImgH);
                float closestZ = tex2D(_OcclusionTex, occUV).r;
                if (camPt.z > closestZ + 0.05)
                    discard;

                float3 viewDir = normalize(_CamPos.xyz - worldP);
                float dotNV = max(dot(norm, viewDir), 0.0);
                if (dotNV <= 0.05)
                    discard;

                float dist = length(_CamPos.xyz - worldP);
                float score = dotNV / max(dist, 0.1);
                outDepth = saturate(score * 0.1);

                // Keyframe UV: screenY and tex2D both use bottom-to-top, no flip
                float2 kfUV = float2(screenX / _ImgW, screenY / _ImgH);
                outColor = tex2D(_KeyframeTex, kfUV);
            }
            ENDHLSL
        }
    }
    FallBack Off
}
