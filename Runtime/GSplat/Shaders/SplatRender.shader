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
            Blend One OneMinusSrcAlpha
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _SORT_RADIX

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct SplatViewData
            {
                float3 worldPos;
                float depth;
                float3 cov3dA;  // upper triangle: (c00, c01, c02)
                float3 cov3dB;  // upper triangle: (c11, c12, c22)
                uint2 color;
            };

            StructuredBuffer<SplatViewData> _SplatViewData;
            float4x4 _SplatVP;
            float4x4 _SplatView;  // render-time view matrix (for EWA projection)
            float2 _SplatFocal;   // (fx, fy) focal lengths in pixels
            float2 _SplatScreen;  // (width, height) in pixels
            uint _SplatCount;

#ifdef _SORT_RADIX
            StructuredBuffer<uint> _OrderBuffer;
#else
            StructuredBuffer<uint2> _SortBuffer;  // .y = original index
#endif

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 quadUV : TEXCOORD0;
                nointerpolation half4 col : COLOR0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            void DecomposeCovariance(float3 cov2d, out float2 v1, out float2 v2)
            {
                float diag1 = cov2d.x, diag2 = cov2d.z, offDiag = cov2d.y;
                float mid = 0.5 * (diag1 + diag2);
                float radius = length(float2((diag1 - diag2) * 0.5, offDiag));
                float lambda1 = mid + radius;
                float lambda2 = max(mid - radius, 0.1);

                float2 rawVec = float2(offDiag, lambda1 - diag1);
                float len = length(rawVec);
                float2 diagVec = len > 1e-6 ? rawVec / len : float2(1, 0);
                diagVec.y = -diagVec.y;

                float maxSize = 4096.0;
                v1 = min(sqrt(2.0 * lambda1), maxSize) * diagVec;
                v2 = min(sqrt(2.0 * lambda2), maxSize) * float2(diagVec.y, -diagVec.x);
            }

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings o = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                uint quadVert = vertID & 3;
                uint splatIdx = vertID >> 2;

                if (splatIdx >= _SplatCount)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

#ifdef _SORT_RADIX
                uint origIdx = _OrderBuffer[splatIdx];
#else
                uint origIdx = _SortBuffer[splatIdx].y;
#endif
                SplatViewData sv = _SplatViewData[origIdx];

                float4 centerClip = mul(_SplatVP, float4(sv.worldPos, 1));
                if (centerClip.w <= 0)
                {
                    o.positionHCS = asfloat(0x7fc00000);
                    return o;
                }

                // EWA projection: 3D cov → 2D cov using the render-time view matrix (+Z fwd)
                float3 pView = float3(
                    _SplatView[0][0]*sv.worldPos.x + _SplatView[0][1]*sv.worldPos.y + _SplatView[0][2]*sv.worldPos.z + _SplatView[0][3],
                    _SplatView[1][0]*sv.worldPos.x + _SplatView[1][1]*sv.worldPos.y + _SplatView[1][2]*sv.worldPos.z + _SplatView[1][3],
                    _SplatView[2][0]*sv.worldPos.x + _SplatView[2][1]*sv.worldPos.y + _SplatView[2][2]*sv.worldPos.z + _SplatView[2][3]
                );

                float fx = _SplatFocal.x, fy = _SplatFocal.y;

                // FOV limiter: clamp projected position to avoid numerical issues
                // at extreme view angles (matches ProjectCov3DEWA in GSplatHelpers.hlsl)
                float tanFovX = 0.5 * _SplatScreen.x / fx;
                float tanFovY = 0.5 * _SplatScreen.y / fy;
                float limX = 1.3 * tanFovX;
                float limY = 1.3 * tanFovY;
                pView.x = pView.z * clamp(pView.x / pView.z, -limX, limX);
                pView.y = pView.z * clamp(pView.y / pView.z, -limY, limY);

                float rz = 1.0 / pView.z;
                float rz2 = rz * rz;

                float j00 = fx * rz;
                float j11 = fy * rz;
                float j20 = -fx * pView.x * rz2;
                float j21 = -fy * pView.y * rz2;

                // T = J * W (W = top-left 3x3 of view matrix)
                float3 mr0 = float3(_SplatView[0][0], _SplatView[0][1], _SplatView[0][2]);
                float3 mr1 = float3(_SplatView[1][0], _SplatView[1][1], _SplatView[1][2]);
                float3 mr2 = float3(_SplatView[2][0], _SplatView[2][1], _SplatView[2][2]);
                float3 t0 = j00 * mr0 + j20 * mr2;
                float3 t1 = j11 * mr1 + j21 * mr2;

                // cov2D = T * Σ3D * T^T (symmetric, store upper triangle)
                float v00 = sv.cov3dA.x, v01 = sv.cov3dA.y, v02 = sv.cov3dA.z;
                float v11 = sv.cov3dB.x, v12 = sv.cov3dB.y, v22 = sv.cov3dB.z;
                float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                                    t0.x*v01 + t0.y*v11 + t0.z*v12,
                                    t0.x*v02 + t0.y*v12 + t0.z*v22);
                float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                                    t1.x*v01 + t1.y*v11 + t1.z*v12,
                                    t1.x*v02 + t1.y*v12 + t1.z*v22);
                float3 cov2d = float3(dot(tv0, t0) + 0.3, dot(tv0, t1), dot(tv1, t1) + 0.3);

                float2 axis1, axis2;
                DecomposeCovariance(cov2d, axis1, axis2);

                o.col.r = f16tof32(sv.color.x >> 16);
                o.col.g = f16tof32(sv.color.x);
                o.col.b = f16tof32(sv.color.y >> 16);
                o.col.a = f16tof32(sv.color.y);

                float2 quadPos = float2(quadVert & 1, (quadVert >> 1) & 1) * 2.0 - 1.0;
                quadPos *= 2.0;
                o.quadUV = quadPos;

                float2 deltaScreen = (quadPos.x * axis1 + quadPos.y * axis2)
                                     * 2.0 / _SplatScreen;
                o.positionHCS = centerClip;
                o.positionHCS.xy += deltaScreen * centerClip.w;

                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                float power = -dot(i.quadUV, i.quadUV);
                half alpha = (half)exp(power);
                alpha = saturate(alpha * i.col.a);
                if (alpha < 1.0h / 255.0h) discard;

                return half4(i.col.rgb * alpha, alpha);
            }
            ENDHLSL
        }
    }
}
