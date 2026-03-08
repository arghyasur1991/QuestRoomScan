Shader "Genesis/DepthVisualize"
{
    Properties
    {
        _MainTex ("Depth Texture", 2D) = "black" {}
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

            sampler2D _MainTex;
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
                float depth = tex2D(_MainTex, i.uv).r;
                if (depth <= 0.001) return half4(0, 0, 0, 0.8);
                float t = saturate((depth - _NearDist) / (_FarDist - _NearDist));
                return half4(lerp(_NearColor.rgb, _FarColor.rgb, t), 1);
            }
            ENDHLSL
        }
    }
}
