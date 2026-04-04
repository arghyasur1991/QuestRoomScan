#if HAS_AI_INFERENCE
using System.IO;
using Genesis.RoomScan.ObjectReconstruction;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        const string RECON_PKG_SHADERS = "Packages/com.genesis.roomscan/Runtime.ObjectReconstruction/Shaders/";
        const string RECON_ONNX_DIR = "Assets/Game/ObjectReconstruction/OnnxSource";
        const string RECON_TEST_IMAGES_DIR = "Assets/Game/ObjectReconstruction/TestImages";
        const string RECON_SENTIS_DIR = "ObjectReconstruction";

        static readonly string[] TestImageNames =
        {
            "backpack_raw",
            "chair",
            "clock_raw",
            "hamburger",
            "robot",
            "shoe_raw",
        };

        ObjectReconstructionModule _objectReconstruction;
        bool _reconTriplaneShaderAssigned;
        bool _reconSurfaceNetsShaderAssigned;
        bool _reconTestImagesAssigned;
        bool _reconSentisModelsExist;
        bool _reconOnnxModelsExist;

        static readonly string[] SentisModelNames =
        {
            "triposr_uint8.sentis",
            "nerf_decoder.sentis",
            "u2netp.sentis"
        };

        static readonly (string onnx, string sentis, bool quantize)[] ModelConversions =
        {
            ("triposr_fp32.onnx", "triposr_uint8.sentis", true),
            ("nerf_decoder.onnx", "nerf_decoder.sentis", false),
            ("u2netp.onnx", "u2netp.sentis", false),
        };

        partial void RefreshObjectReconstruction()
        {
            _objectReconstruction = FindAny<ObjectReconstructionModule>();
            if (_objectReconstruction != null)
            {
                _reconTriplaneShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "triplaneGridSampleShader");
                _reconSurfaceNetsShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "densitySurfaceNetsShader");
                var so = new SerializedObject(_objectReconstruction);
                var imgProp = so.FindProperty("testImages");
                _reconTestImagesAssigned = imgProp != null && imgProp.arraySize > 0;
            }
            else
            {
                _reconTriplaneShaderAssigned = false;
                _reconSurfaceNetsShaderAssigned = false;
                _reconTestImagesAssigned = false;
            }

            string sentisDir = Path.Combine(Application.streamingAssetsPath, RECON_SENTIS_DIR);
            _reconSentisModelsExist = AllSentisModelsExist(sentisDir);
            _reconOnnxModelsExist = AllOnnxModelsExist();
        }

        partial void DrawObjectReconstructionOptionalStatus()
        {
            StatusRowOptional("ObjectReconstructionModule", _objectReconstruction != null);
            if (_objectReconstruction == null) return;

            StatusRow("  Triplane grid sample shader", _reconTriplaneShaderAssigned);
            StatusRow("  Density surface nets shader", _reconSurfaceNetsShaderAssigned);
            StatusRow("  Test images", _reconTestImagesAssigned);

            if (_reconSentisModelsExist)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models (.sentis)", EditorStyles.label);
                GUILayout.Label("OK", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
            }
            else if (_reconOnnxModelsExist)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models", EditorStyles.label);
                GUILayout.Label("Needs Conversion", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Convert Models (Uint8)", GUILayout.Height(24)))
                    ConvertModels(QuantizationType.Uint8);
                if (GUILayout.Button("Convert Models (FP16)", GUILayout.Height(24)))
                    ConvertModels(QuantizationType.Float16);
                EditorGUILayout.EndHorizontal();
            }
            else
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models", EditorStyles.label);
                GUILayout.Label("Missing", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.HelpBox(
                    $"Place triposr_fp32.onnx, nerf_decoder.onnx, and u2netp.onnx in {RECON_ONNX_DIR}/",
                    MessageType.Warning);
            }
        }

        partial void CheckObjectReconstructionAnyMissing(ref bool anyMissing)
        {
            // Fully optional, don't flag as required
        }

        partial void DrawObjectReconstructionShaderStatus(ref bool needsFix)
        {
            if (_objectReconstruction == null) return;

            if (!_reconTriplaneShaderAssigned)
            {
                StatusRow("Reconstruction triplane grid sample shader", false);
                needsFix = true;
            }
            if (!_reconSurfaceNetsShaderAssigned)
            {
                StatusRow("Reconstruction density surface nets shader", false);
                needsFix = true;
            }
            if (!_reconTestImagesAssigned)
            {
                StatusRow("Reconstruction test images", false);
                needsFix = true;
            }
        }

        partial void WireObjectReconstructionComponents()
        {
            if (_objectReconstruction != null)
                WireComponent(_objectReconstruction);
        }

        partial void SetupObjectReconstructionIfAvailable(GameObject root)
        {
            SetupObjectReconstructionModule(root);
        }

        static void WireObjectReconstructionComponent(Component component)
        {
            if (component is ObjectReconstructionModule orm)
                WireObjectReconstructionFields(orm);
        }

        internal static void SetupObjectReconstructionModule(GameObject root)
        {
            if (root.GetComponent<ObjectReconstructionModule>() == null)
                Undo.AddComponent<ObjectReconstructionModule>(root);

            var module = root.GetComponent<ObjectReconstructionModule>();
            if (module != null)
                WireObjectReconstructionFields(module);
        }

        private static void WireObjectReconstructionFields(ObjectReconstructionModule orm)
        {
            var so = new SerializedObject(orm);

            AssignCompute(so, "triplaneGridSampleShader",
                RECON_PKG_SHADERS + "TriplaneGridSample.compute");
            AssignCompute(so, "densitySurfaceNetsShader",
                RECON_PKG_SHADERS + "DensitySurfaceNets.compute");

            var imagesProp = so.FindProperty("testImages");
            if (imagesProp != null && imagesProp.arraySize == 0)
            {
                var found = FindTestImages();
                if (found.Length > 0)
                {
                    imagesProp.arraySize = found.Length;
                    for (int i = 0; i < found.Length; i++)
                        imagesProp.GetArrayElementAtIndex(i).objectReferenceValue = found[i];
                }
            }

            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(orm);
        }

        private static Texture2D[] FindTestImages()
        {
            var result = new System.Collections.Generic.List<Texture2D>();
            foreach (var name in TestImageNames)
            {
                string[] guids = AssetDatabase.FindAssets($"{name} t:Texture2D",
                    new[] { RECON_TEST_IMAGES_DIR });
                if (guids.Length > 0)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                    var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
                    if (tex != null)
                        result.Add(tex);
                }
            }
            return result.ToArray();
        }

        private static bool AllSentisModelsExist(string dir)
        {
            foreach (var name in SentisModelNames)
                if (!File.Exists(Path.Combine(dir, name))) return false;
            return true;
        }

        private static bool AllOnnxModelsExist()
        {
            foreach (var (onnx, _, _) in ModelConversions)
            {
                string path = Path.Combine(RECON_ONNX_DIR, onnx);
                if (!File.Exists(path)) return false;
            }
            return true;
        }

        private static void ConvertModels(QuantizationType quantType)
        {
            string sentisDir = Path.Combine(Application.streamingAssetsPath, RECON_SENTIS_DIR);
            Directory.CreateDirectory(sentisDir);

            int total = ModelConversions.Length;
            for (int i = 0; i < total; i++)
            {
                var (onnxName, sentisName, shouldQuantize) = ModelConversions[i];

                // Override sentis name if FP16 is selected for quantizable models
                if (shouldQuantize && quantType == QuantizationType.Float16)
                    sentisName = sentisName.Replace("uint8", "fp16");

                string onnxPath = Path.Combine(RECON_ONNX_DIR, onnxName);
                string sentisPath = Path.Combine(sentisDir, sentisName);

                if (!File.Exists(onnxPath))
                {
                    Debug.LogWarning($"[ObjectReconstruction] ONNX not found: {onnxPath}");
                    continue;
                }

                EditorUtility.DisplayProgressBar("Converting Models",
                    $"Converting {onnxName}... ({i + 1}/{total})", (float)i / total);

                try
                {
                    var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(onnxPath);
                    if (modelAsset == null)
                    {
                        AssetDatabase.ImportAsset(onnxPath);
                        modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(onnxPath);
                    }

                    if (modelAsset == null)
                    {
                        Debug.LogError($"[ObjectReconstruction] Failed to import {onnxPath} as ModelAsset");
                        continue;
                    }

                    var model = ModelLoader.Load(modelAsset);

                    if (shouldQuantize)
                        ModelQuantizer.QuantizeWeights(quantType, ref model);

                    ModelWriter.Save(sentisPath, model);
                    Debug.Log($"[ObjectReconstruction] Saved {sentisPath} ({new FileInfo(sentisPath).Length / 1048576} MB)");
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[ObjectReconstruction] Failed to convert {onnxName}: {e.Message}");
                }
            }

            EditorUtility.ClearProgressBar();
            AssetDatabase.Refresh();
            Debug.Log("[ObjectReconstruction] Model conversion complete");
        }
    }
}
#endif
