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
            StructuredBuffer<float> _Quats;       // [N*4] rotation quaternions (w,x,y,z)
            uint _SplatCount;
            float _SplatSize;

            static const float SH_C0 = 0.28209479177387814;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                nointerpolation float2 centerPixel : TEXCOORD0;
                nointerpolation float3 conic : TEXCOORD1;
                nointerpolation float3 color : TEXCOORD2;
                nointerpolation float opacity : TEXCOORD3;
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

                OUT.color = saturate(SH_C0 * dc + 0.5);
                OUT.opacity = Sigmoid(_Opacities[splatIdx]);

                // --- View-space transform ---
                float3 pView = mul(UNITY_MATRIX_V, float4(pos, 1)).xyz;
                if (pView.z <= 0.01)
                {
                    OUT.positionHCS = float4(0, 0, -1, 0);
                    return OUT;
                }

                // --- 3D covariance: R * S * S^T * R^T ---
                float3x3 R = QuatToRotmat(quat);
                float gs = _SplatSize;
                float3x3 S = float3x3(
                    gs * scale.x, 0, 0,
                    0, gs * scale.y, 0,
                    0, 0, gs * scale.z);
                float3x3 M = mul(R, S);
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

                // T = J * W (2x3 = Jacobian of projection * view rotation)
                float3 mr0 = float3(UNITY_MATRIX_V[0][0], UNITY_MATRIX_V[0][1], UNITY_MATRIX_V[0][2]);
                float3 mr1 = float3(UNITY_MATRIX_V[1][0], UNITY_MATRIX_V[1][1], UNITY_MATRIX_V[1][2]);
                float3 mr2 = float3(UNITY_MATRIX_V[2][0], UNITY_MATRIX_V[2][1], UNITY_MATRIX_V[2][2]);
                float3 t0 = fx * rz * mr0 + (-fx * pView.x * rz2) * mr2;
                float3 t1 = fy * rz * mr1 + (-fy * pView.y * rz2) * mr2;

                // cov2d = T * covWorld * T^T  (+0.3 regularization on diagonal)
                float3 cv0 = mul(covWorld, t0);
                float3 cv1 = mul(covWorld, t1);
                float3 cov2d = float3(
                    dot(t0, cv0) + 0.3,
                    dot(t0, cv1),
                    dot(t1, cv1) + 0.3);

                // --- Conic (inverse 2D covariance) and pixel radius ---
                float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
                if (det <= 0)
                {
                    OUT.positionHCS = float4(0, 0, -1, 0);
                    return OUT;
                }
                float invDet = 1.0 / det;
                OUT.conic = float3(cov2d.z * invDet, -cov2d.y * invDet, cov2d.x * invDet);

                float b = 0.5 * (cov2d.x + cov2d.z);
                float disc = sqrt(max(0.1, b * b - det));
                float radius = ceil(3.0 * sqrt(b + disc));

                // --- Quad positioning in clip space ---
                float4 centerClip = TransformWorldToHClip(pos);
                OUT.centerPixel = (centerClip.xy / centerClip.w + 1.0) * 0.5 * screenSize;

                float2 offsets[4] = {
                    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
                };
                float2 ndcOffset = offsets[quadVert] * radius * 2.0 / screenSize;
                OUT.positionHCS = float4(
                    centerClip.xy + ndcOffset * centerClip.w,
                    centerClip.z, centerClip.w);

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 delta = IN.positionHCS.xy - IN.centerPixel;
                float power = -0.5 * (IN.conic.x * delta.x * delta.x
                                     + 2.0 * IN.conic.y * delta.x * delta.y
                                     + IN.conic.z * delta.y * delta.y);
                if (power > 0.0) discard;

                float alpha = min(0.99, IN.opacity * exp(power));
                if (alpha < 1.0 / 255.0) discard;

                return half4(IN.color, alpha);
            }
            ENDHLSL
        }
    }
}
