Shader "Genesis/DepthVisualize"
{
    Properties
    {
        _MainTex ("Unused (RawImage compat)", 2D) = "black" {}
        _NearColor ("Near Color", Color) = (1, 0, 0, 1)
        _FarColor ("Far Color", Color) = (0, 0, 1, 1)
        _NearDist ("Near Distance", Float) = 0.3
        _FarDist ("Far Distance", Float) = 5.0
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Overlay" }
        Cull Off
        ZWrite Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // Depth is a Texture2DArray (stereo); read from the global
            // set by DepthCapture.SetGlobalShaderProperties().
            Texture2DArray<float> gsDepthTex;
            SamplerState gsPointClampSampler;
            uniform float4x4 gsDepthProj[2];

            float4 _NearColor;
            float4 _FarColor;
            float _NearDist;
            float _FarDist;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            half4 frag(v2f i) : SV_Target
            {
                float depthNDC = gsDepthTex.SampleLevel(gsPointClampSampler, float3(i.uv, 0), 0);
                if (depthNDC <= 0.001) return half4(0, 0, 0, 0.8);

                float z = depthNDC * 2.0 - 1.0;
                float A = gsDepthProj[0][2][2];
                float B = gsDepthProj[0][2][3];
                float linearDepth = abs(B / (z + A));

                float t = saturate((linearDepth - _NearDist) / (_FarDist - _NearDist));
                return half4(lerp(_NearColor.rgb, _FarColor.rgb, t), 1);
            }
            ENDHLSL
        }
    }
}
