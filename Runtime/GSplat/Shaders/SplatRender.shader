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
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float> _Means;       // [N*3] world xyz
            StructuredBuffer<float> _FeaturesDC;  // [N*3] SH DC coefficients
            StructuredBuffer<float> _Opacities;   // [N]   raw opacity (pre-sigmoid)
            StructuredBuffer<float> _Scales;      // [N*3] log-space scales
            StructuredBuffer<float> _Quats;       // [N*4] rotation quaternions (w,x,y,z)
            uint _SplatCount;
            float _SplatSize;

            static const float SH_C0 = 0.28209479177387814;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 quadUV : TEXCOORD0;
                nointerpolation half3 color : TEXCOORD1;
                nointerpolation half opacity : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float Sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

            float3x3 QuatToRotmat(float4 q)
            {
                float s = rsqrt(dot(q, q));
                float w = q.x * s, x = q.y * s, y = q.z * s, z = q.w * s;
                return float3x3(
                    1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y),
                    2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x),
                    2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)
                );
            }

            // Decompose 2D covariance into oriented axis vectors (matches UGS approach).
            // Returns two screen-space axis vectors encoding the elliptical Gaussian shape.
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
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint quadVert = vertID % 4;
                uint splatIdx = vertID / 4;
                if (splatIdx >= _SplatCount)
                {
                    OUT.positionHCS = asfloat(0x7fc00000);
                    return OUT;
                }

                float3 pos = float3(
                    _Means[splatIdx * 3],
                    _Means[splatIdx * 3 + 1],
                    _Means[splatIdx * 3 + 2]);

                float3 scale = exp(float3(
                    _Scales[splatIdx * 3],
                    _Scales[splatIdx * 3 + 1],
                    _Scales[splatIdx * 3 + 2]));

                float4 quat = float4(
                    _Quats[splatIdx * 4],
                    _Quats[splatIdx * 4 + 1],
                    _Quats[splatIdx * 4 + 2],
                    _Quats[splatIdx * 4 + 3]);

                float3 dc = float3(
                    _FeaturesDC[splatIdx * 3],
                    _FeaturesDC[splatIdx * 3 + 1],
                    _FeaturesDC[splatIdx * 3 + 2]);

                OUT.color = (half3)saturate(SH_C0 * dc + 0.5);
                OUT.opacity = (half)Sigmoid(_Opacities[splatIdx]);

                // --- View-space transform ---
                // Unity's UNITY_MATRIX_V negates Z (OpenGL convention: -Z forward).
                // Negate Z to match training convention (positive Z = forward/depth).
                float3 pView = mul(UNITY_MATRIX_V, float4(pos, 1)).xyz;
                pView.z = -pView.z;
                if (pView.z <= 0.01)
                {
                    OUT.positionHCS = asfloat(0x7fc00000);
                    return OUT;
                }

                // --- 3D covariance: R * S * S^T * R^T ---
                float3x3 R = QuatToRotmat(quat);
                float gs = _SplatSize;
                float3x3 Sc = float3x3(
                    gs * scale.x, 0, 0,
                    0, gs * scale.y, 0,
                    0, 0, gs * scale.z);
                float3x3 M = mul(R, Sc);
                float3x3 covWorld = mul(M, transpose(M));

                // --- EWA projection: 3D cov → 2D cov in pixel space ---
                float2 screenSize = _ScreenParams.xy;
                float fx = abs(UNITY_MATRIX_P[0][0]) * 0.5 * screenSize.x;
                float fy = abs(UNITY_MATRIX_P[1][1]) * 0.5 * screenSize.y;

                float rz = 1.0 / pView.z;
                float rz2 = rz * rz;
                float tanFovX = 0.5 * screenSize.x / fx;
                float tanFovY = 0.5 * screenSize.y / fy;
                float limX = 1.3 * tanFovX;
                float limY = 1.3 * tanFovY;
                pView.x = pView.z * clamp(pView.x * rz, -limX, limX);
                pView.y = pView.z * clamp(pView.y * rz, -limY, limY);

                // Negate mr2 to match the Z-flip above
                float3 mr0 = float3(UNITY_MATRIX_V[0][0], UNITY_MATRIX_V[0][1], UNITY_MATRIX_V[0][2]);
                float3 mr1 = float3(UNITY_MATRIX_V[1][0], UNITY_MATRIX_V[1][1], UNITY_MATRIX_V[1][2]);
                float3 mr2 = -float3(UNITY_MATRIX_V[2][0], UNITY_MATRIX_V[2][1], UNITY_MATRIX_V[2][2]);
                float3 t0 = fx * rz * mr0 + (-fx * pView.x * rz2) * mr2;
                float3 t1 = fy * rz * mr1 + (-fy * pView.y * rz2) * mr2;

                float3 cv0 = mul(covWorld, t0);
                float3 cv1 = mul(covWorld, t1);
                float3 cov2d = float3(
                    dot(t0, cv0) + 0.3,
                    dot(t0, cv1),
                    dot(t1, cv1) + 0.3);

                // --- Decompose 2D covariance into oriented axis vectors ---
                float2 axis1, axis2;
                DecomposeCovariance(cov2d, axis1, axis2);

                // --- Position quad using eigenvector axes ---
                float4 centerClip = TransformWorldToHClip(pos);
                if (centerClip.w <= 0)
                {
                    OUT.positionHCS = asfloat(0x7fc00000);
                    return OUT;
                }

                float2 quadPos = float2(quadVert & 1, (quadVert >> 1) & 1) * 2.0 - 1.0;
                quadPos *= 2.0;

                float2 deltaScreen = (quadPos.x * axis1 + quadPos.y * axis2) * 2.0 / screenSize;
                OUT.positionHCS = centerClip;
                OUT.positionHCS.xy += deltaScreen * centerClip.w;
                OUT.quadUV = quadPos;

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float power = -dot(IN.quadUV, IN.quadUV);
                half alpha = (half)exp(power);
                alpha = saturate(alpha * IN.opacity);
                if (alpha < 1.0h / 255.0h) discard;

                return half4(IN.color, alpha);
            }
            ENDHLSL
        }
    }
}
