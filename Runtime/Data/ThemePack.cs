using UnityEngine;

namespace Genesis.RoomScan
{
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
