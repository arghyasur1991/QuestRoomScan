using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public static class ThemePackEditor
    {
        private const string ThemePath = "Assets/Game/Themes";

        [MenuItem("RoomScan/Create Test ThemePacks")]
        static void CreateTestThemePacks()
        {
            if (!AssetDatabase.IsValidFolder(ThemePath))
            {
                EnsureFolder("Assets/Game");
                EnsureFolder(ThemePath);
            }

            CreateThemePack("Xenotech",
                themeColor: new Color(0.1f, 0.6f, 0.9f, 1f),
                emissiveColor: new Color(0.2f, 0.8f, 1.0f, 1f),
                darkenBrights: 0.5f, edgeGlow: 20f,
                boundaryGlow: 1.2f, noiseScale: 2.5f,
                pulseFreq: 1.5f, pulseAmp: 0.25f,
                desaturation: 0.15f, hueShift: new Color(0.7f, 0.85f, 1.0f, 1f),
                scanlineIntensity: 0.3f, chromaticAberration: 0.002f,
                fresnelIntensity: 0.8f);

            CreateThemePack("Corruption",
                themeColor: new Color(0.4f, 0.05f, 0.05f, 1f),
                emissiveColor: new Color(0.8f, 0.1f, 0.0f, 1f),
                darkenBrights: 1.6f, edgeGlow: 30f,
                boundaryGlow: 2.2f, noiseScale: 1.5f,
                pulseFreq: 0.7f, pulseAmp: 0.5f,
                desaturation: 0.6f, hueShift: new Color(0.6f, 0.3f, 0.3f, 1f),
                scanlineIntensity: 0f, chromaticAberration: 0.004f,
                fresnelIntensity: 1.2f);

            CreateThemePack("Haunted",
                themeColor: new Color(0.25f, 0.25f, 0.3f, 1f),
                emissiveColor: new Color(0.15f, 0.2f, 0.3f, 1f),
                darkenBrights: 0.9f, edgeGlow: 8f,
                boundaryGlow: 0f, noiseScale: 2f,
                pulseFreq: 0.3f, pulseAmp: 0.15f,
                desaturation: 0.8f, hueShift: new Color(0.6f, 0.65f, 0.8f, 1f),
                scanlineIntensity: 0f, chromaticAberration: 0.001f,
                fresnelIntensity: 0.3f,
                flickerIntensity: 0.15f, flickerSpeed: 3f,
                fogDensity: 0.3f, fogColor: new Color(0.02f, 0.02f, 0.05f, 1f),
                progressionMode: ProgressionMode.GlobalIntensity,
                colorTemperatureShift: 0.8f);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("[ThemePackEditor] Created Xenotech + Corruption + Haunted test ThemePacks");
        }

        static void CreateThemePack(string name, Color themeColor, Color emissiveColor,
            float darkenBrights = 0.8f, float edgeGlow = 15f,
            float boundaryGlow = 1.5f, float noiseScale = 2f,
            float pulseFreq = 0f, float pulseAmp = 0.3f,
            float desaturation = 0f, Color? hueShift = null,
            float scanlineIntensity = 0f, float chromaticAberration = 0f,
            float fresnelIntensity = 0f,
            float flickerIntensity = 0f, float flickerSpeed = 3f,
            float fogDensity = 0f, Color? fogColor = null,
            ProgressionMode progressionMode = ProgressionMode.SpatialSpread,
            float colorTemperatureShift = 0f)
        {
            string path = $"{ThemePath}/{name}.asset";
            var existing = AssetDatabase.LoadAssetAtPath<ThemePack>(path);
            if (existing != null)
            {
                Debug.Log($"[ThemePackEditor] {name} already exists at {path}, skipping");
                return;
            }

            var pack = ScriptableObject.CreateInstance<ThemePack>();
            pack.displayName = name;
            pack.themeColor = themeColor;
            pack.emissiveColor = emissiveColor;
            pack.darkenBrights = darkenBrights;
            pack.edgeGlow = edgeGlow;
            pack.boundaryGlow = boundaryGlow;
            pack.transitionNoiseScale = noiseScale;
            pack.pulseFrequency = pulseFreq;
            pack.pulseAmplitude = pulseAmp;
            pack.desaturation = desaturation;
            pack.hueShift = hueShift ?? Color.white;
            pack.scanlineIntensity = scanlineIntensity;
            pack.chromaticAberration = chromaticAberration;
            pack.fresnelIntensity = fresnelIntensity;
            pack.flickerIntensity = flickerIntensity;
            pack.flickerSpeed = flickerSpeed;
            pack.fogDensity = fogDensity;
            pack.fogColor = fogColor ?? new Color(0.02f, 0.02f, 0.05f, 1f);
            pack.progressionMode = progressionMode;
            pack.colorTemperatureShift = colorTemperatureShift;

            AssetDatabase.CreateAsset(pack, path);
            Debug.Log($"[ThemePackEditor] Created {name} at {path}");
        }

        /// <summary>
        /// Bakes the procedural fallback textures for a ThemePack to PNG files
        /// and assigns them back to the SO fields.
        /// </summary>
        [MenuItem("CONTEXT/ThemePack/Bake Placeholder PNGs")]
        static void BakePlaceholderPNGs(MenuCommand cmd)
        {
            var pack = cmd.context as ThemePack;
            if (pack == null) return;

            string assetPath = AssetDatabase.GetAssetPath(pack);
            string dir = System.IO.Path.GetDirectoryName(assetPath);
            string baseName = pack.displayName ?? pack.name;

            pack.triplanarTop = BakeTexture(pack.GetTriplanarTop(), dir, $"{baseName}_top");
            pack.triplanarSide = BakeTexture(pack.GetTriplanarSide(), dir, $"{baseName}_side");
            pack.emissiveMap = BakeTexture(pack.GetEmissiveMap(), dir, $"{baseName}_emissive");

            EditorUtility.SetDirty(pack);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[ThemePackEditor] Baked placeholder PNGs for {baseName}");
        }

        static Texture2D BakeTexture(Texture2D source, string dir, string name)
        {
            if (source == null) return null;
            byte[] png = source.EncodeToPNG();
            string path = $"{dir}/{name}.png";
            System.IO.File.WriteAllBytes(path, png);
            AssetDatabase.ImportAsset(path);

            var importer = AssetImporter.GetAtPath(path) as TextureImporter;
            if (importer != null)
            {
                importer.textureType = TextureImporterType.Default;
                importer.wrapMode = TextureWrapMode.Repeat;
                importer.filterMode = FilterMode.Bilinear;
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            string parent = System.IO.Path.GetDirectoryName(path)?.Replace('\\', '/');
            string folder = System.IO.Path.GetFileName(path);
            if (!string.IsNullOrEmpty(parent) && !AssetDatabase.IsValidFolder(parent))
                EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, folder);
        }
    }
}
