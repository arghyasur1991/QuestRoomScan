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
        enum ModelQuality { Full, Pruned13L, Pruned12L }
        enum ModelResolution { Res512, Res384 }

        static readonly string[] PrecisionSuffixes = { "fp32", "fp16", "int8", "int8_qdq" };
        static readonly string[] PrecisionLabels = { "FP32", "FP16", "INT8", "QDQ" };

        static readonly string[] QualityPrefixes = { "triposr", "triposr_pruned13L", "triposr_pruned12L" };
        static readonly string[] QualityLabels = { "Full (16L)", "Pruned 13L", "Pruned 12L" };

        static readonly string[] ResolutionSuffixes = { "", "_res384" };
        static readonly string[] ResolutionLabels = { "512×512", "384×384" };
        static readonly int[] ResolutionValues = { 512, 384 };

        ObjectReconstructionModule _objectReconstruction;
        bool _reconTriplaneShaderAssigned;
        bool _reconSurfaceNetsShaderAssigned;
        bool _reconMarchingCubesShaderAssigned;
        bool _reconPostprocessShaderAssigned;
        bool _reconVertexColorShaderAssigned;
        bool _reconTestImagesAssigned;
        bool _reconOnnxModelsDeployed;
        string _reconDeployedInfo;
        int _reconSelectedQuality;
        int _reconSelectedResolution;
        bool[,,] _reconAvailableVariants = new bool[3, 2, 4];

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
            _reconDeployedInfo = DetectDeployedInfo(streamingDir);

            for (int q = 0; q < 3; q++)
                for (int r = 0; r < 2; r++)
                    for (int p = 0; p < 4; p++)
                        _reconAvailableVariants[q, r, p] = IsVariantAvailable(
                            (ModelQuality)q, (ModelResolution)r, (ModelPrecision)p);
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
                string deployLabel = string.IsNullOrEmpty(_reconDeployedInfo)
                    ? "OK"
                    : $"OK ({_reconDeployedInfo})";
                GUILayout.Label(deployLabel, EditorStyles.boldLabel);
                EditorGUILayout.EndHorizontal();
            }

            bool anyAvailable = false;
            for (int q = 0; q < 3 && !anyAvailable; q++)
                for (int r = 0; r < 2 && !anyAvailable; r++)
                    for (int p = 0; p < 4 && !anyAvailable; p++)
                        anyAvailable = _reconAvailableVariants[q, r, p];

            if (anyAvailable)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField("  Deploy Models to StreamingAssets:",
                    EditorStyles.miniLabel);

                EditorGUILayout.BeginHorizontal();
                GUILayout.Space(20);
                EditorGUILayout.LabelField("Quality:", GUILayout.Width(50));
                _reconSelectedQuality = EditorGUILayout.Popup(
                    _reconSelectedQuality, QualityLabels, GUILayout.Width(120));
                EditorGUILayout.LabelField("Resolution:", GUILayout.Width(65));
                _reconSelectedResolution = EditorGUILayout.Popup(
                    _reconSelectedResolution, ResolutionLabels, GUILayout.Width(80));
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                GUILayout.Space(20);
                for (int i = 0; i < 4; i++)
                {
                    bool available = _reconAvailableVariants[
                        _reconSelectedQuality, _reconSelectedResolution, i];
                    using (new EditorGUI.DisabledScope(!available))
                    {
                        string label = PrecisionLabels[i];
                        if (available)
                        {
                            long sz = GetVariantSizeMB(
                                (ModelQuality)_reconSelectedQuality,
                                (ModelResolution)_reconSelectedResolution,
                                (ModelPrecision)i);
                            if (sz > 0) label += $" ({sz}MB)";
                        }
                        if (GUILayout.Button(label, GUILayout.Height(22)))
                            DeployOnnxModels(
                                (ModelQuality)_reconSelectedQuality,
                                (ModelResolution)_reconSelectedResolution,
                                (ModelPrecision)i);
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
                    $"Run: python TripoSR/export_onnx.py --prune 0 1 2 --resolutions 512 384 --qdq --deploy all\n" +
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

        private static string GetFullPrefix(ModelQuality quality, ModelResolution resolution)
        {
            return QualityPrefixes[(int)quality] + ResolutionSuffixes[(int)resolution];
        }

        private static bool IsVariantAvailable(ModelQuality quality, ModelResolution resolution,
            ModelPrecision precision)
        {
            string prefix = GetFullPrefix(quality, resolution);
            string suffix = PrecisionSuffixes[(int)precision];
            string[] required =
            {
                $"{prefix}_part1_{suffix}.onnx",
                $"{prefix}_part2_{suffix}.onnx",
                $"nerf_decoder_{suffix}.onnx",
                "u2netp.onnx",
            };

            foreach (var name in required)
                if (!File.Exists(Path.Combine(RECON_ONNX_DIR, name))) return false;
            return true;
        }

        private static long GetVariantSizeMB(ModelQuality quality, ModelResolution resolution,
            ModelPrecision precision)
        {
            string prefix = GetFullPrefix(quality, resolution);
            string suffix = PrecisionSuffixes[(int)precision];
            long total = 0;
            string[] parts = { $"{prefix}_part1_{suffix}.onnx", $"{prefix}_part2_{suffix}.onnx" };
            foreach (var name in parts)
            {
                string path = Path.Combine(RECON_ONNX_DIR, name);
                if (File.Exists(path)) total += new FileInfo(path).Length;
            }
            return total / (1024 * 1024);
        }

        private static string DetectDeployedInfo(string streamingDir)
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
            for (int q = QualityPrefixes.Length - 1; q >= 0; q--)
            {
                for (int r = ResolutionSuffixes.Length - 1; r >= 0; r--)
                {
                    for (int p = PrecisionSuffixes.Length - 1; p >= 0; p--)
                    {
                        string prefix = QualityPrefixes[q] + ResolutionSuffixes[r];
                        string srcPath = Path.Combine(RECON_ONNX_DIR,
                            $"{prefix}_part1_{PrecisionSuffixes[p]}.onnx");
                        if (File.Exists(srcPath) && new FileInfo(srcPath).Length == deployedSize)
                        {
                            string resLabel = r > 0 ? $" {ResolutionLabels[r]}" : "";
                            return $"{QualityLabels[q]}{resLabel} / {PrecisionSuffixes[p].ToUpper()}";
                        }
                    }
                }
            }

            return null;
        }

        private static void DeployOnnxModels(ModelQuality quality, ModelResolution resolution,
            ModelPrecision precision)
        {
            string prefix = GetFullPrefix(quality, resolution);
            string suffix = PrecisionSuffixes[(int)precision];
            string streamingDir = Path.Combine(Application.streamingAssetsPath, RECON_STREAMING_DIR);
            Directory.CreateDirectory(streamingDir);

            (string src, string dst)[] models =
            {
                ($"{prefix}_part1_{suffix}.onnx", "triposr_part1.onnx"),
                ($"{prefix}_part2_{suffix}.onnx", "triposr_part2.onnx"),
                ($"nerf_decoder_{suffix}.onnx", "nerf_decoder.onnx"),
                ("u2netp.onnx", "u2netp.onnx"),
            };

            string qualityLabel = QualityLabels[(int)quality];
            string resLabel = (int)resolution > 0 ? $" {ResolutionLabels[(int)resolution]}" : "";
            string displayLabel = $"{qualityLabel}{resLabel}";
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

                EditorUtility.DisplayProgressBar(
                    $"Deploying {displayLabel} {suffix.ToUpper()} Models",
                    $"Copying {srcName} → {dstName}... ({i + 1}/{models.Length})",
                    (float)i / models.Length);

                File.Copy(srcPath, dstPath, overwrite: true);
                var fi = new FileInfo(dstPath);
                Debug.Log($"[ObjectReconstruction] Deployed {dstName} ← {srcName} " +
                          $"({fi.Length / 1048576} MB)");
            }

            string markerPath = Path.Combine(streamingDir, ".precision");
            File.WriteAllText(markerPath, $"{displayLabel} / {suffix}");

            string imageSizePath = Path.Combine(streamingDir, ".imagesize");
            File.WriteAllText(imageSizePath, ResolutionValues[(int)resolution].ToString());

            EditorUtility.ClearProgressBar();
            AssetDatabase.Refresh();
            Debug.Log($"[ObjectReconstruction] {displayLabel} {suffix.ToUpper()} deployment complete");
        }
    }
}
#endif
