#if HAS_ONNXRUNTIME
using System.IO;
using Genesis.RoomScan.ObjectReconstruction;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        const string RECON_PKG_SHADERS = "Packages/com.genesis.roomscan/Runtime.ObjectReconstruction/Shaders/";
        const string RECON_ONNX_DIR = "Assets/Game/ObjectReconstruction/OnnxSource";
        const string RECON_TEST_IMAGES_DIR = "Assets/Game/ObjectReconstruction/TestImages";
        const string RECON_STREAMING_DIR = "ObjectReconstruction";

        static readonly string[] TestImageNames =
        {
            "backpack_raw",
            "chair",
            "clock_raw",
            "hamburger",
            "robot",
            "shoe_raw",
        };

        static readonly string[] OnnxModelNames =
        {
            "triposr_part1.onnx",
            "triposr_part2.onnx",
            "nerf_decoder.onnx",
            "u2netp.onnx"
        };

        ObjectReconstructionModule _objectReconstruction;
        bool _reconTriplaneShaderAssigned;
        bool _reconSurfaceNetsShaderAssigned;
        bool _reconMarchingCubesShaderAssigned;
        bool _reconPostprocessShaderAssigned;
        bool _reconVertexColorShaderAssigned;
        bool _reconTestImagesAssigned;
        bool _reconOnnxModelsDeployed;
        bool _reconOnnxSourcesExist;

        partial void RefreshObjectReconstruction()
        {
            _objectReconstruction = FindAny<ObjectReconstructionModule>();
            if (_objectReconstruction != null)
            {
                _reconTriplaneShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "triplaneGridSampleShader");
                _reconSurfaceNetsShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "densitySurfaceNetsShader");
                _reconMarchingCubesShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "densityMarchingCubesShader");
                _reconPostprocessShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "decoderPostprocessShader");
                _reconVertexColorShaderAssigned = AreFieldsAssigned(_objectReconstruction,
                    "vertexColorShader");
                var so = new SerializedObject(_objectReconstruction);
                var imgProp = so.FindProperty("testImages");
                _reconTestImagesAssigned = imgProp != null && imgProp.arraySize > 0;
            }
            else
            {
                _reconTriplaneShaderAssigned = false;
                _reconSurfaceNetsShaderAssigned = false;
                _reconMarchingCubesShaderAssigned = false;
                _reconPostprocessShaderAssigned = false;
                _reconVertexColorShaderAssigned = false;
                _reconTestImagesAssigned = false;
            }

            string streamingDir = Path.Combine(Application.streamingAssetsPath, RECON_STREAMING_DIR);
            _reconOnnxModelsDeployed = AllOnnxModelsDeployed(streamingDir);
            _reconOnnxSourcesExist = AllOnnxSourcesExist();
        }

        partial void DrawObjectReconstructionOptionalStatus()
        {
            StatusRowOptional("ObjectReconstructionModule", _objectReconstruction != null);
            if (_objectReconstruction == null) return;

            StatusRow("  Triplane grid sample shader", _reconTriplaneShaderAssigned);
            StatusRow("  Density surface nets shader", _reconSurfaceNetsShaderAssigned);
            StatusRow("  Marching cubes shader", _reconMarchingCubesShaderAssigned);
            StatusRow("  Decoder postprocess shader", _reconPostprocessShaderAssigned);
            StatusRow("  Vertex color shader", _reconVertexColorShaderAssigned);
            StatusRow("  Test images", _reconTestImagesAssigned);

            if (_reconOnnxModelsDeployed)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models (.onnx)", EditorStyles.label);
                GUILayout.Label("OK", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
            }
            else if (_reconOnnxSourcesExist)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models", EditorStyles.label);
                GUILayout.Label("Needs Deploy", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();

                if (GUILayout.Button("Deploy Models to StreamingAssets", GUILayout.Height(24)))
                    DeployOnnxModels();
            }
            else
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models", EditorStyles.label);
                GUILayout.Label("Missing", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.HelpBox(
                    $"Place triposr_part1.onnx, triposr_part2.onnx, nerf_decoder.onnx, and u2netp.onnx in {RECON_ONNX_DIR}/\n" +
                    "Generate split ONNX via: python TripoSR/split_triposr.py",
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
            if (!_reconMarchingCubesShaderAssigned)
            {
                StatusRow("Reconstruction marching cubes shader", false);
                needsFix = true;
            }
            if (!_reconPostprocessShaderAssigned)
            {
                StatusRow("Reconstruction decoder postprocess shader", false);
                needsFix = true;
            }
            if (!_reconVertexColorShaderAssigned)
            {
                StatusRow("Reconstruction vertex color shader", false);
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
            AssignCompute(so, "densityMarchingCubesShader",
                RECON_PKG_SHADERS + "DensityMarchingCubes.compute");
            AssignCompute(so, "decoderPostprocessShader",
                RECON_PKG_SHADERS + "DecoderPostprocess.compute");
            AssignAsset<Shader>(so, "vertexColorShader",
                RECON_PKG_SHADERS + "VertexColor.shader");

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

        private static bool AllOnnxModelsDeployed(string dir)
        {
            foreach (var name in OnnxModelNames)
                if (!File.Exists(Path.Combine(dir, name))) return false;
            return true;
        }

        private static bool AllOnnxSourcesExist()
        {
            foreach (var name in OnnxModelNames)
            {
                string path = Path.Combine(RECON_ONNX_DIR, name);
                if (!File.Exists(path)) return false;
            }
            return true;
        }

        private static void DeployOnnxModels()
        {
            string streamingDir = Path.Combine(Application.streamingAssetsPath, RECON_STREAMING_DIR);
            Directory.CreateDirectory(streamingDir);

            int total = OnnxModelNames.Length;
            for (int i = 0; i < total; i++)
            {
                string onnxName = OnnxModelNames[i];
                string srcPath = Path.Combine(RECON_ONNX_DIR, onnxName);
                string dstPath = Path.Combine(streamingDir, onnxName);

                if (!File.Exists(srcPath))
                {
                    Debug.LogWarning($"[ObjectReconstruction] ONNX not found: {srcPath}");
                    continue;
                }

                EditorUtility.DisplayProgressBar("Deploying Models",
                    $"Copying {onnxName}... ({i + 1}/{total})", (float)i / total);

                File.Copy(srcPath, dstPath, overwrite: true);
                var fi = new FileInfo(dstPath);
                Debug.Log($"[ObjectReconstruction] Deployed {dstPath} ({fi.Length / 1048576} MB)");
            }

            EditorUtility.ClearProgressBar();
            AssetDatabase.Refresh();
            Debug.Log("[ObjectReconstruction] Model deployment complete");
        }
    }
}
#endif
