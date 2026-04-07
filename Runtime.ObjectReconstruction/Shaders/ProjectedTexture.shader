Shader "Hidden/ObjectReconstruction/ProjectedTexture"
{
    Properties
    {
        _MainTex ("Projected Texture", 2D) = "gray" {}
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float4 color      : COLOR;
                float3 normalOS   : NORMAL;
                float2 uv0        : TEXCOORD0; // projected UV
                float2 uv1        : TEXCOORD1; // x = blend factor (1=texture, 0=vertex color)
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 vertColor  : TEXCOORD0;
                float3 normalWS   : TEXCOORD1;
                float3 normalOS   : TEXCOORD2;
                float2 projUV     : TEXCOORD3;
                float  blend      : TEXCOORD4;
            };

            Varyings vert(Attributes i)
            {
                Varyings o;
                o.positionCS = TransformObjectToHClip(i.positionOS.xyz);
                o.vertColor = i.color.rgb;
                o.normalWS = TransformObjectToWorldNormal(i.normalOS);
                o.normalOS = i.normalOS;
                o.projUV = i.uv0;
                o.blend = i.uv1.x;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float3 nWS = normalize(i.normalWS);
                float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
                float ndl = saturate(dot(nWS, lightDir));
                float lighting = lerp(0.3, 1.0, ndl);

                // Canonical camera is at +X looking along -X.
                // MC inverted normals: camera-facing surface has normalOS.x < 0.
                float canonicalFacing = saturate(-normalize(i.normalOS).x * 3.0);
                float effectiveBlend = i.blend * canonicalFacing;

                float3 texColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.projUV).rgb;
                float3 baseColor = lerp(i.vertColor, texColor, effectiveBlend);

                return half4(baseColor * lighting, 1.0);
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            Cull Off
            ZWrite On
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex vertDepth
            #pragma fragment fragDepth
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            float4 vertDepth(float4 positionOS : POSITION) : SV_POSITION
            {
                return TransformObjectToHClip(positionOS.xyz);
            }

            half4 fragDepth() : SV_Target { return 0; }
            ENDHLSL
        }
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Cull Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
                float3 normal : NORMAL;
                float2 uv0 : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 vertColor : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float3 normalOS : TEXCOORD2;
                float2 projUV : TEXCOORD3;
                float blend : TEXCOORD4;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.vertColor = v.color.rgb;
                o.normalWS = UnityObjectToWorldNormal(v.normal);
                o.normalOS = v.normal;
                o.projUV = v.uv0;
                o.blend = v.uv1.x;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float ndl = saturate(dot(normalize(i.normalWS), normalize(float3(0.5, 1, 0.3))));
                float lighting = lerp(0.3, 1.0, ndl);

                float canonicalFacing = saturate(-normalize(i.normalOS).x * 3.0);
                float effectiveBlend = i.blend * canonicalFacing;

                float3 texColor = tex2D(_MainTex, i.projUV).rgb;
                float3 baseColor = lerp(i.vertColor, texColor, effectiveBlend);
                return fixed4(baseColor * lighting, 1);
            }
            ENDCG
        }
    }
}
