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

        static readonly string[] DeployedModelNames =
        {
            "triposr_part1.onnx",
            "triposr_part2.onnx",
            "nerf_decoder.onnx",
            "u2netp.onnx"
        };

        enum ModelPrecision { FP32, FP16, INT8, INT8_QDQ }

        static readonly string[] PrecisionSuffixes = { "fp32", "fp16", "int8", "int8_qdq" };
        static readonly string[] PrecisionLabels = { "FP32 (1.7GB)", "FP16 (840MB)", "INT8 (460MB)", "QDQ (415MB)" };

        ObjectReconstructionModule _objectReconstruction;
        bool _reconTriplaneShaderAssigned;
        bool _reconSurfaceNetsShaderAssigned;
        bool _reconMarchingCubesShaderAssigned;
        bool _reconPostprocessShaderAssigned;
        bool _reconVertexColorShaderAssigned;
        bool _reconTestImagesAssigned;
        bool _reconOnnxModelsDeployed;
        string _reconDeployedPrecision;
        bool[] _reconAvailablePrecisions = new bool[4];

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
            _reconOnnxModelsDeployed = AllDeployedModelsExist(streamingDir);
            _reconDeployedPrecision = DetectDeployedPrecision(streamingDir);

            for (int i = 0; i < 4; i++)
                _reconAvailablePrecisions[i] = IsPrecisionAvailable((ModelPrecision)i);
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
                string deployLabel = string.IsNullOrEmpty(_reconDeployedPrecision)
                    ? "OK"
                    : $"OK ({_reconDeployedPrecision.ToUpper()})";
                GUILayout.Label(deployLabel, EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
            }

            bool anyAvailable = _reconAvailablePrecisions[0]
                             || _reconAvailablePrecisions[1]
                             || _reconAvailablePrecisions[2]
                             || _reconAvailablePrecisions[3];

            if (anyAvailable)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField("  Deploy Models to StreamingAssets:",
                    EditorStyles.miniLabel);

                EditorGUILayout.BeginHorizontal();
                GUILayout.Space(20);
                for (int i = 0; i < 4; i++)
                {
                    using (new EditorGUI.DisabledScope(!_reconAvailablePrecisions[i]))
                    {
                        if (GUILayout.Button(PrecisionLabels[i], GUILayout.Height(22)))
                            DeployOnnxModels((ModelPrecision)i);
                    }
                }
                EditorGUILayout.EndHorizontal();
            }
            else if (!_reconOnnxModelsDeployed)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("  Models", EditorStyles.label);
                GUILayout.Label("Missing", EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.HelpBox(
                    $"Run: python TripoSR/export_onnx.py --deploy all\n" +
                    $"This generates and copies all model variants to {RECON_ONNX_DIR}/",
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

        private static bool AllDeployedModelsExist(string dir)
        {
            foreach (var name in DeployedModelNames)
                if (!File.Exists(Path.Combine(dir, name))) return false;
            return true;
        }

        /// <summary>
        /// Check if a complete set of models exists in OnnxSource for the given precision.
        /// u2netp uses FP32 for FP32/FP16/INT8 (FP16 broken, INT8 same size),
        /// but uses QDQ variant for INT8_QDQ (70% smaller, works on QNN HTP).
        /// </summary>
        private static bool IsPrecisionAvailable(ModelPrecision precision)
        {
            string suffix = PrecisionSuffixes[(int)precision];
            string u2netpName = precision == ModelPrecision.INT8_QDQ
                ? "u2netp_int8_qdq.onnx"
                : "u2netp.onnx";
            string[] required =
            {
                $"triposr_part1_{suffix}.onnx",
                $"triposr_part2_{suffix}.onnx",
                $"nerf_decoder_{suffix}.onnx",
                u2netpName,
            };

            foreach (var name in required)
                if (!File.Exists(Path.Combine(RECON_ONNX_DIR, name))) return false;
            return true;
        }

        /// <summary>
        /// Detect which precision is currently deployed via the .precision marker,
        /// falling back to file size comparison against OnnxSource variants.
        /// </summary>
        private static string DetectDeployedPrecision(string streamingDir)
        {
            string markerPath = Path.Combine(streamingDir, ".precision");
            if (File.Exists(markerPath))
            {
                string marker = File.ReadAllText(markerPath).Trim();
                if (!string.IsNullOrEmpty(marker)) return marker;
            }

            string deployedPart1 = Path.Combine(streamingDir, "triposr_part1.onnx");
            if (!File.Exists(deployedPart1)) return null;

            long deployedSize = new FileInfo(deployedPart1).Length;
            for (int i = PrecisionSuffixes.Length - 1; i >= 0; i--)
            {
                string suffix = PrecisionSuffixes[i];
                string srcPath = Path.Combine(RECON_ONNX_DIR, $"triposr_part1_{suffix}.onnx");
                if (File.Exists(srcPath) && new FileInfo(srcPath).Length == deployedSize)
                    return suffix;
            }

            return null;
        }

        private static void DeployOnnxModels(ModelPrecision precision)
        {
            string suffix = PrecisionSuffixes[(int)precision];
            string streamingDir = Path.Combine(Application.streamingAssetsPath, RECON_STREAMING_DIR);
            Directory.CreateDirectory(streamingDir);

            string u2netpSrc = precision == ModelPrecision.INT8_QDQ
                ? "u2netp_int8_qdq.onnx"
                : "u2netp.onnx";

            (string src, string dst)[] models =
            {
                ($"triposr_part1_{suffix}.onnx", "triposr_part1.onnx"),
                ($"triposr_part2_{suffix}.onnx", "triposr_part2.onnx"),
                ($"nerf_decoder_{suffix}.onnx", "nerf_decoder.onnx"),
                (u2netpSrc, "u2netp.onnx"),
            };

            for (int i = 0; i < models.Length; i++)
            {
                var (srcName, dstName) = models[i];
                string srcPath = Path.Combine(RECON_ONNX_DIR, srcName);
                string dstPath = Path.Combine(streamingDir, dstName);

                if (!File.Exists(srcPath))
                {
                    Debug.LogWarning($"[ObjectReconstruction] ONNX not found: {srcPath}");
                    continue;
                }

                EditorUtility.DisplayProgressBar($"Deploying {suffix.ToUpper()} Models",
                    $"Copying {srcName} → {dstName}... ({i + 1}/{models.Length})",
                    (float)i / models.Length);

                File.Copy(srcPath, dstPath, overwrite: true);
                var fi = new FileInfo(dstPath);
                Debug.Log($"[ObjectReconstruction] Deployed {dstName} ← {srcName} " +
                          $"({fi.Length / 1048576} MB)");
            }

            string markerPath = Path.Combine(streamingDir, ".precision");
            File.WriteAllText(markerPath, suffix);

            EditorUtility.ClearProgressBar();
            AssetDatabase.Refresh();
            Debug.Log($"[ObjectReconstruction] {suffix.ToUpper()} model deployment complete");
        }
    }
}
#endif
