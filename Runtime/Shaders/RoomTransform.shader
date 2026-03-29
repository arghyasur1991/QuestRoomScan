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
        _BumpMap            ("Normal Map",           2D)         = "bump"  {}
        _ThemeTexTop        ("Theme Top",            2D)         = "gray"  {}
        _ThemeTexSide       ("Theme Side",           2D)         = "gray"  {}
        _ThemeEmissive      ("Theme Emissive",       2D)         = "black" {}
        _ThemeColor         ("Theme Color",      Color)          = (1,1,1,1)
        _EmissiveColor      ("Emissive Color",   Color)          = (0,0,0,1)
        _TransformGlobal    ("Transform Progress",   Range(0,1)) = 0
        _TriplanarScale     ("Triplanar Scale",      Float)      = 1.0
        _TriplanarSharpness ("Triplanar Sharpness",  Float)      = 4.0
        _NormalStrength     ("Normal Strength",      Float)      = 1.0
        _LightDir           ("Light Direction",  Vector)         = (0.3, 1.0, 0.2, 0)

        [Header(Blending)]
        _DarkenBrights      ("Darken Brights",       Range(0,2)) = 0.8
        _EdgeGlow           ("Edge Glow Strength",   Range(0,50)) = 15.0
        _BoundaryGlow       ("Boundary Glow",        Range(0,3)) = 1.5
        _NoiseScale         ("Transition Noise Scale", Float)    = 2.0

        [Header(Effects)]
        _PulseFreq          ("Pulse Frequency",      Range(0,8)) = 0
        _PulseAmp           ("Pulse Amplitude",      Range(0,1)) = 0.3
        _Desaturation       ("Desaturation",         Range(0,1)) = 0
        _HueShift           ("Hue Shift Tint",   Color)          = (1,1,1,1)
        _ScanlineIntensity  ("Scanline Intensity",   Range(0,1)) = 0
        _ChromaticAberration("Chromatic Aberration", Range(0,0.01)) = 0
        _FresnelIntensity   ("Fresnel Intensity",    Range(0,3)) = 0
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
                float4 posWS    : POSITION;
                float3 normal   : NORMAL;
                float4 tangent  : TANGENT;
                float2 uv       : TEXCOORD0;
                float4 color    : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 posCS    : SV_POSITION;
                float2 uv       : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float3 worldNml : TEXCOORD2;
                float3 worldTan : TEXCOORD3;
                float3 worldBit : TEXCOORD4;
                float  surfType : TEXCOORD5;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            TEXTURE2D(_MainTex);        SAMPLER(sampler_MainTex);
            TEXTURE2D(_BumpMap);        SAMPLER(sampler_BumpMap);
            TEXTURE2D(_ThemeTexTop);     SAMPLER(sampler_ThemeTexTop);
            TEXTURE2D(_ThemeTexSide);    SAMPLER(sampler_ThemeTexSide);
            TEXTURE2D(_ThemeEmissive);   SAMPLER(sampler_ThemeEmissive);

            CBUFFER_START(UnityPerMaterial)
                half4  _ThemeColor;
                half4  _EmissiveColor;
                half   _TransformGlobal;
                float  _TriplanarScale;
                float  _TriplanarSharpness;
                float  _NormalStrength;
                float4 _LightDir;
                half   _DarkenBrights;
                half   _EdgeGlow;
                half   _BoundaryGlow;
                float  _NoiseScale;
                half   _PulseFreq;
                half   _PulseAmp;
                half   _Desaturation;
                half4  _HueShift;
                half   _ScanlineIntensity;
                half   _ChromaticAberration;
                half   _FresnelIntensity;
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
                o.worldTan = v.tangent.xyz;
                o.worldBit = cross(v.normal, v.tangent.xyz) * v.tangent.w;
                o.surfType = v.color.r * 255.0;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                half progress = _TransformGlobal;

                // ── Real room ───────────────────────────────────────────
                half3 realColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).rgb;
                float realLum = dot(realColor, half3(0.2126, 0.7152, 0.0722));

                // ── Atlas normal map (TBN) ──────────────────────────────
                half4 nSample = SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, i.uv);
                half3 tn;
                tn.xy = (nSample.rg * 2.0 - 1.0) * _NormalStrength;
                tn.z = sqrt(saturate(1.0 - dot(tn.xy, tn.xy)));

                float3 N = normalize(i.worldNml);
                float3 T = normalize(i.worldTan);
                float3 B = normalize(i.worldBit);
                float3 realNormal = normalize(T * tn.x + B * tn.y + N * tn.z);

                // ── Noise-driven transition mask ────────────────────────
                float noise = ValueNoise(i.worldPos * _NoiseScale);
                float threshold = progress * 1.4 - 0.2;
                float mask = 1.0 - smoothstep(threshold - 0.12, threshold + 0.08, noise);
                mask = saturate(mask * step(0.001, progress));

                // ── Luminance darkening ─────────────────────────────────
                float darken = lerp(1.0, saturate(1.0 - realLum * _DarkenBrights), mask);
                half3 adjustedReal = realColor * darken;

                // ── Theme triplanar ─────────────────────────────────────
                half3 nml = N;
                half3 weights = TriplanarWeights(nml, _TriplanarSharpness);
                half3 themeColor = TriplanarSample(
                    TEXTURE2D_ARGS(_ThemeTexTop, sampler_ThemeTexTop),
                    TEXTURE2D_ARGS(_ThemeTexSide, sampler_ThemeTexSide),
                    i.worldPos, weights, _TriplanarScale) * _ThemeColor.rgb;

                half3 emissiveTex = TriplanarSample(
                    TEXTURE2D_ARGS(_ThemeEmissive, sampler_ThemeEmissive),
                    TEXTURE2D_ARGS(_ThemeEmissive, sampler_ThemeEmissive),
                    i.worldPos, weights, _TriplanarScale);

                // ── Theme height-derived normal perturbation ────────────
                float themeLum = dot(themeColor, half3(0.2126, 0.7152, 0.0722));
                float tdx = ddx(themeLum);
                float tdy = ddy(themeLum);
                float3 themeNormal = normalize(float3(-tdx * 4.0, -tdy * 4.0, 0.15));
                themeNormal = normalize(T * themeNormal.x + B * themeNormal.y + N * themeNormal.z);

                // Blend between real room normal and theme-derived normal
                float3 worldNormal = normalize(lerp(realNormal, themeNormal, mask));

                // ── Half-lambert directional light ──────────────────────
                float3 lightDir = normalize(_LightDir.xyz);
                half NdotL = dot(worldNormal, lightDir);
                half halfLambert = NdotL * 0.5 + 0.5;

                // ── Chromatic aberration at boundary ─────────────────────
                // Shift UV reads for R and B channels near the frontier
                half3 chromaReal = adjustedReal;
                if (_ChromaticAberration > 0.0001)
                {
                    float chromaMask = exp(-abs(noise - threshold) * 20.0) * mask;
                    float2 offset = float2(_ChromaticAberration, 0);
                    half rr = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv + offset).r;
                    half bb = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv - offset).b;
                    chromaReal = lerp(adjustedReal, half3(rr, adjustedReal.g, bb) * darken, chromaMask);
                }

                // ── Overlay blend ───────────────────────────────────────
                half3 blended = OverlayBlend(chromaReal, themeColor);

                half3 color = lerp(chromaReal, blended, mask);
                color *= halfLambert;

                // ── Desaturation + hue tint ─────────────────────────────
                if (_Desaturation > 0.001)
                {
                    float grey = dot(color, half3(0.2126, 0.7152, 0.0722));
                    half3 desatColor = lerp(color, grey * _HueShift.rgb, _Desaturation);
                    color = lerp(color, desatColor, mask);
                }

                // ── Edge glow ───────────────────────────────────────────
                float lumDx = ddx(realLum);
                float lumDy = ddy(realLum);
                float edge = saturate(sqrt(lumDx * lumDx + lumDy * lumDy) * _EdgeGlow);
                color += _EmissiveColor.rgb * edge * mask;

                // ── Emissive map (circuits / veins) ─────────────────────
                half3 emissiveContrib = emissiveTex * _EmissiveColor.rgb * mask;

                // ── Animated emissive pulse ──────────────────────────────
                if (_PulseFreq > 0.001)
                {
                    half pulse = sin(_Time.y * _PulseFreq * 6.2832) * _PulseAmp;
                    emissiveContrib *= (1.0 + pulse);
                }
                color += emissiveContrib;

                // ── Fresnel edge highlight ───────────────────────────────
                if (_FresnelIntensity > 0.001)
                {
                    float3 viewDir = normalize(_WorldSpaceCameraPos - i.worldPos);
                    half fresnel = 1.0 - saturate(dot(viewDir, worldNormal));
                    fresnel = pow(fresnel, 3.0);
                    color += _EmissiveColor.rgb * fresnel * _FresnelIntensity * mask;
                }

                // ── Scanlines / holographic ──────────────────────────────
                if (_ScanlineIntensity > 0.001)
                {
                    float scanline = frac(i.worldPos.y * 120.0 + _Time.y * 2.0);
                    scanline = step(0.5, scanline);
                    color = lerp(color, color * (1.0 - _ScanlineIntensity * 0.5), scanline * mask);
                }

                // ── Boundary glow ───────────────────────────────────────
                float distToBoundary = abs(noise - threshold);
                float boundary = exp(-distToBoundary * 25.0)
                               * _BoundaryGlow
                               * step(0.01, progress) * step(progress, 0.99);
                color += _EmissiveColor.rgb * boundary;

                // ── Precursor shadow (dark band ahead of frontier) ──────
                float aheadDist = noise - threshold;
                float precursor = smoothstep(0.0, 0.2, aheadDist)
                                * smoothstep(0.4, 0.2, aheadDist)
                                * step(0.01, progress) * step(progress, 0.99);
                color *= lerp(1.0, 0.7, precursor);

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
