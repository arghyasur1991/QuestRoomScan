// URP setup helpers for the Room Scan setup wizard.
//
// A new Unity 6 project ships with GraphicsSettings + QualitySettings
// already wired to a UniversalRenderPipelineAsset GUID, but the .asset
// files themselves are not generated until you either install URP via
// Package Manager into a fresh project or copy them in manually. The
// result is a project that references a pipeline asset that doesn't
// exist on disk — every shader falls back to magenta and most of QRS's
// rendering pipeline silently does nothing.
//
// EnsureURPSetup:
//   1. Looks for a UniversalRenderPipelineAsset already wired into
//      GraphicsSettings.defaultRenderPipeline. If one exists and resolves
//      on disk, it's a no-op.
//   2. Otherwise creates Assets/Settings/URP-Pipeline.asset and
//      Assets/Settings/URP-Renderer.asset using URP's own internal asset
//      factory (so they get the same default-resource wiring as the
//      "Create > Rendering > URP Asset (with Universal Renderer)" menu).
//   3. Applies Quest-friendly defaults (4x MSAA, no HDR, single shadow
//      cascade, etc.) via SerializedObject so we don't depend on internal
//      setters that move between URP versions.
//   4. Assigns the new pipeline asset to GraphicsSettings.defaultRenderPipeline
//      and to every QualitySettings level so referenced shaders resolve.

using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        const string URP_DIR            = "Assets/Settings";
        const string URP_PIPELINE_PATH  = URP_DIR + "/URP-Pipeline.asset";
        const string URP_RENDERER_PATH  = URP_DIR + "/URP-Renderer.asset";

        bool _urpConfigured;
        UniversalRenderPipelineAsset _urpAssetCached;

        void RefreshURPState()
        {
            _urpAssetCached = GraphicsSettings.defaultRenderPipeline as UniversalRenderPipelineAsset;
            _urpConfigured  = _urpAssetCached != null
                              && AllQualityLevelsUseUrp(_urpAssetCached);
        }

        static bool AllQualityLevelsUseUrp(UniversalRenderPipelineAsset target)
        {
            int prev = QualitySettings.GetQualityLevel();
            try
            {
                int count = QualitySettings.names.Length;
                for (int i = 0; i < count; i++)
                {
                    QualitySettings.SetQualityLevel(i, applyExpensiveChanges: false);
                    if (QualitySettings.renderPipeline != target) return false;
                }
                return true;
            }
            finally
            {
                QualitySettings.SetQualityLevel(prev, applyExpensiveChanges: false);
            }
        }

        /// <summary>
        /// Idempotent: ensures URP-Pipeline.asset + URP-Renderer.asset exist
        /// at Assets/Settings/, and that GraphicsSettings + every quality
        /// level point to the pipeline asset. Safe to call repeatedly.
        /// </summary>
        static UniversalRenderPipelineAsset EnsureURPSetup()
        {
            try
            {
                if (!Directory.Exists(URP_DIR))
                    Directory.CreateDirectory(URP_DIR);

                var pipeline = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(URP_PIPELINE_PATH);

                if (pipeline == null)
                {
                    pipeline = CreateURPPipelineAsset();
                    if (pipeline == null) return null;
                    Debug.Log($"[RoomScan Setup] Created {URP_PIPELINE_PATH} (+ renderer).");
                }

                ApplyQuestFriendlyDefaults(pipeline);

                // Wire into GraphicsSettings + every quality level.
                if (GraphicsSettings.defaultRenderPipeline != pipeline)
                {
                    GraphicsSettings.defaultRenderPipeline = pipeline;
                    Debug.Log("[RoomScan Setup] Set GraphicsSettings.defaultRenderPipeline \u2192 " +
                              URP_PIPELINE_PATH);
                }

                AssignToAllQualityLevels(pipeline);

                AssetDatabase.SaveAssets();
                return pipeline;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[RoomScan Setup] EnsureURPSetup failed: {ex.Message}\n{ex.StackTrace}");
                return null;
            }
        }

        static UniversalRenderPipelineAsset CreateURPPipelineAsset()
        {
            // Mirror of the menu item Assets > Create > Rendering > URP Asset
            // (with Universal Renderer). The renderer asset is created first
            // (sibling of the pipeline asset, suffixed _Renderer by URP), then
            // passed into UniversalRenderPipelineAsset.Create which links it
            // as RendererDataList[0]. We use reflection because
            // CreateRendererAsset is internal.

            var t = typeof(UniversalRenderPipelineAsset);
            var createRendererAsset = t.GetMethod("CreateRendererAsset",
                BindingFlags.NonPublic | BindingFlags.Static);
            if (createRendererAsset == null)
            {
                Debug.LogError("[RoomScan Setup] URP package internal API moved: " +
                               "UniversalRenderPipelineAsset.CreateRendererAsset not found.");
                return null;
            }

            // Signature: (string path, RendererType type, bool relativePath, string suffix)
            // Pass relativePath=false + our explicit URP_RENDERER_PATH so the
            // file lands exactly where we want (otherwise URP appends
            // "_Renderer" to the pipeline filename).
            ScriptableRendererData rendererData;
            var paramInfos = createRendererAsset.GetParameters();
            if (paramInfos.Length == 4)
            {
                rendererData = (ScriptableRendererData)createRendererAsset.Invoke(null, new object[]
                {
                    URP_RENDERER_PATH, RendererType.UniversalRenderer, false, "Renderer",
                });
            }
            else if (paramInfos.Length == 2)
            {
                // Older signature without relativePath/suffix.
                rendererData = (ScriptableRendererData)createRendererAsset.Invoke(null, new object[]
                {
                    URP_PIPELINE_PATH, RendererType.UniversalRenderer,
                });
            }
            else
            {
                Debug.LogError($"[RoomScan Setup] Unexpected CreateRendererAsset signature ({paramInfos.Length} args).");
                return null;
            }

            if (rendererData == null)
            {
                Debug.LogError("[RoomScan Setup] URP renderer asset creation returned null.");
                return null;
            }

            var pipeline = UniversalRenderPipelineAsset.Create(rendererData);
            AssetDatabase.CreateAsset(pipeline, URP_PIPELINE_PATH);
            return pipeline;
        }

        /// <summary>
        /// Quest-friendly URP defaults: 4x MSAA (Quest GPU has dedicated
        /// MSAA hardware so it's nearly free), HDR off (saves ~30% bandwidth),
        /// shadow distance trimmed to 30m, single cascade, soft shadows off.
        /// SRP batcher stays on. Editing through SerializedObject so we
        /// don't touch internal setters.
        /// </summary>
        static void ApplyQuestFriendlyDefaults(UniversalRenderPipelineAsset asset)
        {
            var so = new SerializedObject(asset);
            bool changed = false;

            void SetInt(string prop, int value)
            {
                var p = so.FindProperty(prop);
                if (p != null && p.intValue != value) { p.intValue = value; changed = true; }
            }
            void SetBool(string prop, bool value)
            {
                var p = so.FindProperty(prop);
                if (p != null && p.boolValue != value) { p.boolValue = value; changed = true; }
            }
            void SetFloat(string prop, float value)
            {
                var p = so.FindProperty(prop);
                if (p != null && Mathf.Abs(p.floatValue - value) > 1e-4f)
                { p.floatValue = value; changed = true; }
            }

            SetInt   ("m_MSAA",                       4);
            SetBool  ("m_SupportsHDR",                false);
            SetInt   ("m_ShadowCascadeCount",         1);
            SetFloat ("m_ShadowDistance",             30f);
            SetBool  ("m_SoftShadowsSupported",       false);
            SetBool  ("m_UseSRPBatcher",              true);
            // Single Pass Instanced (multi-view) is set in PlayerSettings via
            // VRProjectBootstrap; URP picks it up automatically.

            if (changed)
            {
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(asset);
                Debug.Log("[RoomScan Setup] Applied Quest-friendly defaults to URP-Pipeline.asset " +
                          "(4x MSAA, no HDR, 1 shadow cascade, 30m shadow distance, no soft shadows).");
            }
        }

        static void AssignToAllQualityLevels(UniversalRenderPipelineAsset pipeline)
        {
            int prev = QualitySettings.GetQualityLevel();
            try
            {
                int count = QualitySettings.names.Length;
                int touched = 0;
                for (int i = 0; i < count; i++)
                {
                    QualitySettings.SetQualityLevel(i, applyExpensiveChanges: false);
                    if (QualitySettings.renderPipeline != pipeline)
                    {
                        QualitySettings.renderPipeline = pipeline;
                        touched++;
                    }
                }
                if (touched > 0)
                    Debug.Log($"[RoomScan Setup] Assigned URP pipeline to {touched} quality level(s).");
            }
            finally
            {
                QualitySettings.SetQualityLevel(prev, applyExpensiveChanges: false);
            }
        }
    }
}
