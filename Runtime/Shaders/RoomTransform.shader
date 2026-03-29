// Room transformation shader — blends the real baked atlas into a themed
// version using perceptual blending that respects the room's own structure.
//
// Key techniques:
//   - World-space value noise drives an organic transition boundary
//   - Overlay blend preserves room texture detail instead of flat replacement
//   - Luminance-aware darkening dims bright areas (horror: lights going out)
//   - Screen-space edge detection (ddx/ddy) highlights structural edges
//   - Boundary glow at the moving frontier of transformation
Shader "Genesis/RoomTransform"
{
    Properties
    {
        _MainTex            ("Atlas",                2D)         = "white" {}
        _ThemeTexTop        ("Theme Top",            2D)         = "gray"  {}
        _ThemeTexSide       ("Theme Side",           2D)         = "gray"  {}
        _ThemeEmissive      ("Theme Emissive",       2D)         = "black" {}
        _ThemeColor         ("Theme Color",      Color)          = (1,1,1,1)
        _EmissiveColor      ("Emissive Color",   Color)          = (0,0,0,1)
        _TransformGlobal    ("Transform Progress",   Range(0,1)) = 0
        _TriplanarScale     ("Triplanar Scale",      Float)      = 1.0
        _TriplanarSharpness ("Triplanar Sharpness",  Float)      = 4.0

        [Header(Blending)]
        _DarkenBrights      ("Darken Brights",       Range(0,2)) = 0.8
        _EdgeGlow           ("Edge Glow Strength",   Range(0,50)) = 15.0
        _BoundaryGlow       ("Boundary Glow",        Range(0,3)) = 1.5
        _NoiseScale         ("Transition Noise Scale", Float)    = 2.0
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
                half   _DarkenBrights;
                half   _EdgeGlow;
                half   _BoundaryGlow;
                float  _NoiseScale;
            CBUFFER_END

            // ── Noise (ALU only, no texture fetch) ──────────────────────

            float Hash3to1(float3 p)
            {
                p = frac(p * float3(443.897, 441.423, 437.195));
                p += dot(p, p.yzx + 19.19);
                return frac((p.x + p.y) * p.z);
            }

            float ValueNoise(float3 p)
            {
                float3 i = floor(p);
                float3 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);

                float n000 = Hash3to1(i);
                float n100 = Hash3to1(i + float3(1,0,0));
                float n010 = Hash3to1(i + float3(0,1,0));
                float n110 = Hash3to1(i + float3(1,1,0));
                float n001 = Hash3to1(i + float3(0,0,1));
                float n101 = Hash3to1(i + float3(1,0,1));
                float n011 = Hash3to1(i + float3(0,1,1));
                float n111 = Hash3to1(i + float3(1,1,1));

                return lerp(
                    lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
                    lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y),
                    f.z);
            }

            // ── Triplanar helpers ───────────────────────────────────────

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
                half3 colTop   = SAMPLE_TEXTURE2D(texTop,  sampTop,  wp.xz * scale).rgb;
                half3 colSideX = SAMPLE_TEXTURE2D(texSide, sampSide, wp.yz * scale).rgb;
                half3 colSideZ = SAMPLE_TEXTURE2D(texSide, sampSide, wp.xy * scale).rgb;
                return colTop * weights.y + colSideX * weights.x + colSideZ * weights.z;
            }

            // ── Overlay blend (Photoshop-style) ─────────────────────────

            half3 OverlayBlend(half3 base, half3 blend)
            {
                half3 lo = 2.0 * base * blend;
                half3 hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
                return half3(
                    base.r < 0.5 ? lo.r : hi.r,
                    base.g < 0.5 ? lo.g : hi.g,
                    base.b < 0.5 ? lo.b : hi.b);
            }

            // ─────────────────────────────────────────────────────────────

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

                // ── Real room ───────────────────────────────────────────
                half3 realColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).rgb;
                float realLum = dot(realColor, half3(0.2126, 0.7152, 0.0722));

                // ── Noise-driven transition mask ────────────────────────
                // Remap progress [0,1] into a threshold that sweeps through
                // the noise field.  Smoothstep gives a soft boundary.
                float noise = ValueNoise(i.worldPos * _NoiseScale);
                // Expand progress range so 0 = nothing, 1 = everything covered
                float threshold = progress * 1.4 - 0.2;
                float mask = smoothstep(threshold - 0.12, threshold + 0.08, noise);
                // mask: 0 = still real, 1 = fully transformed
                mask = saturate(mask * step(0.001, progress));

                // ── Luminance darkening ─────────────────────────────────
                // Bright areas in the real room dim as transformation grows.
                // Horror themes use high _DarkenBrights to kill the lights.
                float darken = lerp(1.0, saturate(1.0 - realLum * _DarkenBrights), mask);
                half3 adjustedReal = realColor * darken;

                // ── Theme triplanar ─────────────────────────────────────
                half3 nml = normalize(i.worldNml);
                half3 weights = TriplanarWeights(nml, _TriplanarSharpness);
                half3 themeColor = TriplanarSample(
                    TEXTURE2D_ARGS(_ThemeTexTop, sampler_ThemeTexTop),
                    TEXTURE2D_ARGS(_ThemeTexSide, sampler_ThemeTexSide),
                    i.worldPos, weights, _TriplanarScale) * _ThemeColor.rgb;

                half3 emissiveTex = TriplanarSample(
                    TEXTURE2D_ARGS(_ThemeEmissive, sampler_ThemeEmissive),
                    TEXTURE2D_ARGS(_ThemeEmissive, sampler_ThemeEmissive),
                    i.worldPos, weights, _TriplanarScale);

                // ── Overlay blend ───────────────────────────────────────
                // Preserves room structure: shadows stay shadows, highlights
                // get tinted rather than replaced.
                half3 blended = OverlayBlend(adjustedReal, themeColor);

                // Mix: untransformed pixels keep adjusted real, transformed
                // pixels get the overlay blend.
                half3 color = lerp(adjustedReal, blended, mask);

                // ── Edge glow ───────────────────────────────────────────
                // Screen-space derivatives of real room luminance detect
                // structural edges (corners, moldings, furniture outlines).
                float lumDx = ddx(realLum);
                float lumDy = ddy(realLum);
                float edge = saturate(sqrt(lumDx * lumDx + lumDy * lumDy) * _EdgeGlow);
                color += _EmissiveColor.rgb * edge * mask;

                // ── Emissive map (circuits / veins) ─────────────────────
                color += emissiveTex * _EmissiveColor.rgb * mask;

                // ── Boundary glow ───────────────────────────────────────
                // Hot emissive line at the advancing frontier of the
                // transformation. Falls off exponentially from the edge.
                float distToBoundary = abs(noise - threshold);
                float boundary = exp(-distToBoundary * 25.0)
                               * _BoundaryGlow
                               * step(0.01, progress) * step(progress, 0.99);
                color += _EmissiveColor.rgb * boundary;

                return half4(color, 1);
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
