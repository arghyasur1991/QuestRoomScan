using System;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>Per-surface override for transformation parameters.</summary>
    [Serializable]
    public struct SurfaceOverride
    {
        [Tooltip("Multiplier on transition noise scale for this surface type")]
        public float noiseScaleMultiplier;
        [Tooltip("Emissive multiplier for this surface type")]
        public float emissiveMultiplier;
        [Tooltip("Spread rate multiplier (higher = spreads faster on this surface)")]
        public float spreadRateMultiplier;
        [Tooltip("Triplanar texture scale override")]
        public float triplanarScaleMultiplier;

        public static SurfaceOverride Default => new SurfaceOverride
        {
            noiseScaleMultiplier = 1f,
            emissiveMultiplier = 1f,
            spreadRateMultiplier = 1f,
            triplanarScaleMultiplier = 1f,
        };
    }

    /// <summary>
    /// Pluggable theme definition for room transformation.  Swap the ThemePack
    /// and the entire room changes character.  Texture fields may be left null —
    /// <see cref="GetTriplanarTop"/>, <see cref="GetTriplanarSide"/>, and
    /// <see cref="GetEmissiveMap"/> will return procedural placeholders based on
    /// the configured colours.
    /// </summary>
    [CreateAssetMenu(fileName = "NewThemePack", menuName = "RoomScan/ThemePack")]
    public class ThemePack : ScriptableObject
    {
        [Header("Identity")]
        public string displayName;

        [Header("Triplanar Textures")]
        [Tooltip("Floor / ceiling / horizontal surface texture (triplanar Y axis)")]
        public Texture2D triplanarTop;
        [Tooltip("Wall / vertical surface texture (triplanar X and Z axes)")]
        public Texture2D triplanarSide;

        [Header("Emissive")]
        [Tooltip("Glowing overlay (circuits, veins, runes). Black = no glow.")]
        public Texture2D emissiveMap;

        [Header("Colors")]
        public Color themeColor = Color.white;
        public Color emissiveColor = Color.black;

        [Header("Blending")]
        [Tooltip("How much bright areas in the real room dim (0 = none, 1.5+ = horror darkness)")]
        [Range(0, 2)] public float darkenBrights = 0.8f;
        [Tooltip("Strength of emissive glow on structural edges detected from the room texture")]
        [Range(0, 50)] public float edgeGlow = 15f;
        [Tooltip("Intensity of the glowing frontier at the advancing edge of transformation")]
        [Range(0, 3)] public float boundaryGlow = 1.5f;
        [Tooltip("Scale of world-space noise driving the organic transition boundary (smaller = larger patches)")]
        public float transitionNoiseScale = 2f;

        [Header("Effects")]
        [Tooltip("Animated emissive pulse frequency (Hz). 0 = off.")]
        [Range(0, 8)] public float pulseFrequency = 0f;
        [Tooltip("Pulse amplitude — how strong the brightness oscillation is.")]
        [Range(0, 1)] public float pulseAmplitude = 0.3f;
        [Tooltip("Desaturation factor in transformed regions (0 = full color, 1 = greyscale)")]
        [Range(0, 1)] public float desaturation = 0f;
        [Tooltip("Hue tint applied to desaturated colour in transformed areas")]
        public Color hueShift = Color.white;
        [Tooltip("Horizontal scanline / holographic intensity (0 = off)")]
        [Range(0, 1)] public float scanlineIntensity = 0f;
        [Tooltip("Chromatic aberration at transformation boundary (0 = off)")]
        [Range(0, 0.01f)] public float chromaticAberration = 0f;
        [Tooltip("Fresnel edge highlight intensity (0 = off)")]
        [Range(0, 3)] public float fresnelIntensity = 0f;

        [Header("Surface Overrides (Floor, Ceiling, Wall, Furniture)")]
        [Tooltip("Per-surface parameter multipliers. Index 0=Floor, 1=Ceiling, 2=Wall, 3=Furniture")]
        public SurfaceOverride[] surfaceOverrides = new SurfaceOverride[]
        {
            SurfaceOverride.Default, // Floor
            SurfaceOverride.Default, // Ceiling
            SurfaceOverride.Default, // Wall
            SurfaceOverride.Default, // Furniture
        };

        /// <summary>Get the override for a given surface type (returns Default for Unknown).</summary>
        public SurfaceOverride GetSurfaceOverride(SurfaceType type)
        {
            int idx = type switch
            {
                SurfaceType.Floor     => 0,
                SurfaceType.Ceiling   => 1,
                SurfaceType.Wall      => 2,
                SurfaceType.Furniture => 3,
                _ => -1
            };
            if (idx >= 0 && idx < surfaceOverrides.Length) return surfaceOverrides[idx];
            return SurfaceOverride.Default;
        }

        [Header("VFX / Audio (optional)")]
        public GameObject particlePrefab;
        public AudioClip ambientAudio;
        public AudioClip transformSound;

        // Runtime-generated fallback textures (not serialized)
        [System.NonSerialized] private Texture2D _fallbackTop;
        [System.NonSerialized] private Texture2D _fallbackSide;
        [System.NonSerialized] private Texture2D _fallbackEmissive;

        public Texture2D GetTriplanarTop()
        {
            if (triplanarTop != null) return triplanarTop;
            _fallbackTop ??= ProceduralTextures.GenerateGrid(
                themeColor * 0.3f, themeColor, 32);
            return _fallbackTop;
        }

        public Texture2D GetTriplanarSide()
        {
            if (triplanarSide != null) return triplanarSide;
            _fallbackSide ??= ProceduralTextures.GenerateCircuitLines(
                themeColor * 0.2f, themeColor);
            return _fallbackSide;
        }

        public Texture2D GetEmissiveMap()
        {
            if (emissiveMap != null) return emissiveMap;
            _fallbackEmissive ??= ProceduralTextures.GenerateNoiseEmissive(
                Color.black, emissiveColor);
            return _fallbackEmissive;
        }

        /// <summary>
        /// Applies this theme's textures and colours to a material that uses
        /// the Genesis/RoomTransform shader.
        /// </summary>
        public void ApplyToMaterial(Material mat)
        {
            if (mat == null) return;
            mat.SetTexture("_ThemeTexTop",   GetTriplanarTop());
            mat.SetTexture("_ThemeTexSide",  GetTriplanarSide());
            mat.SetTexture("_ThemeEmissive", GetEmissiveMap());
            mat.SetColor("_ThemeColor",      themeColor);
            mat.SetColor("_EmissiveColor",   emissiveColor);
            mat.SetFloat("_DarkenBrights",   darkenBrights);
            mat.SetFloat("_EdgeGlow",        edgeGlow);
            mat.SetFloat("_BoundaryGlow",    boundaryGlow);
            mat.SetFloat("_NoiseScale",      transitionNoiseScale);
            mat.SetFloat("_PulseFreq",       pulseFrequency);
            mat.SetFloat("_PulseAmp",        pulseAmplitude);
            mat.SetFloat("_Desaturation",    desaturation);
            mat.SetColor("_HueShift",        hueShift);
            mat.SetFloat("_ScanlineIntensity", scanlineIntensity);
            mat.SetFloat("_ChromaticAberration", chromaticAberration);
            mat.SetFloat("_FresnelIntensity", fresnelIntensity);
        }

        private void OnDestroy()
        {
            DestroyFallback(ref _fallbackTop);
            DestroyFallback(ref _fallbackSide);
            DestroyFallback(ref _fallbackEmissive);
        }

        static void DestroyFallback(ref Texture2D tex)
        {
            if (tex != null)
            {
                if (Application.isPlaying) Destroy(tex);
                else DestroyImmediate(tex);
                tex = null;
            }
        }
    }
}
