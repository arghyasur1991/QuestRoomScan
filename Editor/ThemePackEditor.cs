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
                boundaryGlow: 1.2f, noiseScale: 2.5f);

            CreateThemePack("Corruption",
                themeColor: new Color(0.4f, 0.05f, 0.05f, 1f),
                emissiveColor: new Color(0.8f, 0.1f, 0.0f, 1f),
                darkenBrights: 1.6f, edgeGlow: 30f,
                boundaryGlow: 2.2f, noiseScale: 1.5f);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("[ThemePackEditor] Created Xenotech + Corruption test ThemePacks");
        }

        static void CreateThemePack(string name, Color themeColor, Color emissiveColor,
            float darkenBrights = 0.8f, float edgeGlow = 15f,
            float boundaryGlow = 1.5f, float noiseScale = 2f)
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
