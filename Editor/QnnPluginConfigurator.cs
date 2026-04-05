#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// Auto-configures QNN .so plugin import settings for Android arm64.
    /// Runs once per editor session when QNN libs are present.
    /// </summary>
    [InitializeOnLoad]
    internal static class QnnPluginConfigurator
    {
        const string SESSION_KEY = "QnnPluginConfigurator_Checked";
        const string QNN_DIR = "Assets/Plugins/OnnxRuntime/Android/QNN";

        static QnnPluginConfigurator()
        {
            if (SessionState.GetBool(SESSION_KEY, false))
                return;
            SessionState.SetBool(SESSION_KEY, true);
            EditorApplication.delayCall += ConfigureQnnPlugins;
        }

        static void ConfigureQnnPlugins()
        {
            string fullPath = Path.GetFullPath(QNN_DIR);
            if (!Directory.Exists(fullPath))
                return;

            var soFiles = Directory.GetFiles(fullPath, "*.so");
            if (soFiles.Length == 0)
                return;

            int configured = 0;
            foreach (string soFile in soFiles)
            {
                string assetPath = QNN_DIR + "/" + Path.GetFileName(soFile);
                var importer = AssetImporter.GetAtPath(assetPath) as PluginImporter;
                if (importer == null)
                    continue;

                importer.SetCompatibleWithAnyPlatform(false);
                importer.SetCompatibleWithEditor(false);
                importer.SetCompatibleWithPlatform(BuildTarget.Android, true);
                importer.SetPlatformData(BuildTarget.Android, "CPU", "ARM64");
                importer.isPreloaded = false;
                importer.SaveAndReimport();
                configured++;
            }

            if (configured > 0)
                Debug.Log($"[QnnPluginConfigurator] Configured {configured} QNN .so plugins for Android ARM64");
        }
    }
}
#endif
