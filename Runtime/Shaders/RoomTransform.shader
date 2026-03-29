// Blends the real baked atlas with themed triplanar textures driven by a
// per-vertex progress value.  At progress=0 you see the real room; at
// progress=1 the themed version.  Triplanar projection means themed textures
// tile naturally across any geometry without UV dependency.
Shader "Genesis/RoomTransform"
{
    Properties
    {
        _MainTex        ("Atlas",            2D) = "white" {}
        _ThemeTexTop    ("Theme Top",        2D) = "gray"  {}
        _ThemeTexSide   ("Theme Side",       2D) = "gray"  {}
        _ThemeEmissive  ("Theme Emissive",   2D) = "black" {}
        _ThemeColor     ("Theme Color",  Color)  = (1,1,1,1)
        _EmissiveColor  ("Emissive Color", Color) = (0,0,0,1)
        _TransformGlobal("Transform Progress", Range(0,1)) = 0
        _TriplanarScale ("Triplanar Scale",  Float) = 1.0
        _TriplanarSharpness ("Triplanar Sharpness", Float) = 4.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        Pass
        {
            Name "TransformUnlit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 posWS  : POSITION;
                float3 normal : NORMAL;
                float2 uv     : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 posCS    : SV_POSITION;
                float2 uv       : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float3 worldNml : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            TEXTURE2D(_MainTex);        SAMPLER(sampler_MainTex);
            TEXTURE2D(_ThemeTexTop);     SAMPLER(sampler_ThemeTexTop);
            TEXTURE2D(_ThemeTexSide);    SAMPLER(sampler_ThemeTexSide);
            TEXTURE2D(_ThemeEmissive);   SAMPLER(sampler_ThemeEmissive);

            CBUFFER_START(UnityPerMaterial)
                half4  _ThemeColor;
                half4  _EmissiveColor;
                half   _TransformGlobal;
                float  _TriplanarScale;
                float  _TriplanarSharpness;
            CBUFFER_END

            // Triplanar blend weights from world normal
            half3 TriplanarWeights(half3 n, float sharpness)
            {
                half3 w = abs(n);
                w = pow(w, sharpness);
                return w / (w.x + w.y + w.z + 1e-5);
            }

            half3 TriplanarSample(TEXTURE2D_PARAM(texTop, sampTop),
                                  TEXTURE2D_PARAM(texSide, sampSide),
                                  float3 wp, half3 weights, float scale)
            {
                float2 uvXZ = wp.xz * scale;
                float2 uvXY = wp.xy * scale;
                float2 uvYZ = wp.yz * scale;

                half3 colTop  = SAMPLE_TEXTURE2D(texTop,  sampTop,  uvXZ).rgb;
                half3 colSideX = SAMPLE_TEXTURE2D(texSide, sampSide, uvYZ).rgb;
                half3 colSideZ = SAMPLE_TEXTURE2D(texSide, sampSide, uvXY).rgb;

                return colTop * weights.y + colSideX * weights.x + colSideZ * weights.z;
            }

            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.posCS    = TransformWorldToHClip(v.posWS.xyz);
                o.uv       = v.uv;
                o.worldPos = v.posWS.xyz;
                o.worldNml = v.normal;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                half progress = _TransformGlobal;

                half3 realColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).rgb;

                half3 nml = normalize(i.worldNml);
                half3 weights = TriplanarWeights(nml, _TriplanarSharpness);
                half3 themeColor = TriplanarSample(
                    TEXTURE2D_ARGS(_ThemeTexTop, sampler_ThemeTexTop),
                    TEXTURE2D_ARGS(_ThemeTexSide, sampler_ThemeTexSide),
                    i.worldPos, weights, _TriplanarScale);
                themeColor *= _ThemeColor.rgb;

                half3 emissive = TriplanarSample(
                    TEXTURE2D_ARGS(_ThemeEmissive, sampler_ThemeEmissive),
                    TEXTURE2D_ARGS(_ThemeEmissive, sampler_ThemeEmissive),
                    i.worldPos, weights, _TriplanarScale);
                emissive *= _EmissiveColor.rgb * step(0.05, progress);

                half3 final = lerp(realColor, themeColor, progress) + emissive * progress;
                return half4(final, 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            ZWrite On
            ColorMask 0
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 posWS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 posCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.posCS = TransformWorldToHClip(v.posWS.xyz);
                return o;
            }

            half4 frag(Varyings i) : SV_Target { return 0; }
            ENDHLSL
        }
    }
    FallBack "Universal Render Pipeline/Unlit"
}
