#if HAS_ONNXRUNTIME
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Genesis.RoomScan.ObjectReconstruction;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// Editor window for testing the Object Reconstruction pipeline in Edit mode.
    /// Runs rembg, TripoSR forward, and mesh extraction without entering Play mode.
    /// ORT inference naturally runs on background threads — no throttling needed.
    /// </summary>
    public class ObjectReconstructionTestWindow : EditorWindow
    {
        [MenuItem("RoomScan/Object Reconstruction Test")]
        public static void ShowWindow()
        {
            GetWindow<ObjectReconstructionTestWindow>("Reconstruction Test");
        }

        [SerializeField] private Texture2D _testImage;
        [SerializeField] private int _gridResolution = 128;
        [SerializeField] private MeshAlgorithm _meshAlgorithm = MeshAlgorithm.MarchingCubes;
        [SerializeField] private ExecutionProvider _executionProvider = ExecutionProvider.CPU;
        [SerializeField] private int _densitySmoothPasses = 0;
        [SerializeField] private MeshExtractionBackend _meshExtractionBackend = MeshExtractionBackend.CPU;
        [SerializeField] private int _laplacianSmoothIterations = 0;
        [SerializeField] private float _laplacianSmoothLambda = 0.5f;

        private ComputeShader _triplaneShader;
        private ComputeShader _surfaceNetsShader;
        private ComputeShader _marchingCubesShader;
        private Shader _vertexColorShader;
        private Shader _projectedTextureShader;
        private ComputeShader _postprocessShader;

        private string _status = "Ready";
        private float _progress;
        private bool _running;
        private CancellationTokenSource _cts;

        private Mesh _resultMesh;
        private GameObject _previewObj;

        private string _timingLog = "";
        private Stopwatch _stepSw;

        // Keyframe-based reconstruction
        private string _keyframeDir = "";
        private System.Collections.Generic.List<DetectionEntry> _detections;
        private System.Collections.Generic.Dictionary<int, KeyframeEntry> _frames;
        private int _selectedDetectionIdx = -1;
        private Texture2D _croppedPreview;
        private Vector2 _keyframeScroll;

        private const string SHADER_DIR = "Packages/com.genesis.roomscan/Runtime.ObjectReconstruction/Shaders/";

        private void OnEnable()
        {
            _triplaneShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                SHADER_DIR + "TriplaneGridSample.compute");
            _surfaceNetsShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                SHADER_DIR + "DensitySurfaceNets.compute");
            _marchingCubesShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                SHADER_DIR + "DensityMarchingCubes.compute");
            _postprocessShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                SHADER_DIR + "DecoderPostprocess.compute");
            _vertexColorShader = AssetDatabase.LoadAssetAtPath<Shader>(
                SHADER_DIR + "VertexColor.shader");
            _projectedTextureShader = AssetDatabase.LoadAssetAtPath<Shader>(
                SHADER_DIR + "ProjectedTexture.shader");

            InstallEditModeYield();
        }

        private void OnDisable()
        {
            Cancel();
            CleanupPreview();
            AsyncHelper.EditModeYield = null;
        }

        private static void InstallEditModeYield()
        {
            if (Application.isPlaying) return;

            AsyncHelper.EditModeYield = () =>
            {
                var tcs = new TaskCompletionSource<bool>();
                EditorApplication.delayCall += () => tcs.SetResult(true);
                return tcs.Task;
            };
        }

        private void OnGUI()
        {
            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Object Reconstruction Pipeline Test", EditorStyles.boldLabel);
            EditorGUILayout.Space(4);

            DrawModelStatus();
            EditorGUILayout.Space(4);

            _testImage = (Texture2D)EditorGUILayout.ObjectField(
                "Test Image", _testImage, typeof(Texture2D), false);

            _gridResolution = EditorGUILayout.IntPopup("Grid Resolution",
                _gridResolution, new[] { "64 (fast)", "128 (balanced)", "256 (quality)" },
                new[] { 64, 128, 256 });

            _meshAlgorithm = (MeshAlgorithm)EditorGUILayout.EnumPopup("Mesh Algorithm", _meshAlgorithm);

            _executionProvider = (ExecutionProvider)EditorGUILayout.EnumPopup(
                "Execution Provider", _executionProvider);

            _densitySmoothPasses = EditorGUILayout.IntSlider(
                "Density Smooth Passes", _densitySmoothPasses, 0, 3);

            _laplacianSmoothIterations = EditorGUILayout.IntSlider(
                "Laplacian Smooth Iters", _laplacianSmoothIterations, 0, 5);
            if (_laplacianSmoothIterations > 0)
                _laplacianSmoothLambda = EditorGUILayout.Slider(
                    "Laplacian Lambda", _laplacianSmoothLambda, 0.1f, 0.9f);

            _meshExtractionBackend = (MeshExtractionBackend)EditorGUILayout.EnumPopup(
                "Mesh Extraction", _meshExtractionBackend);

            EditorGUILayout.Space(4);
            DrawShaderStatus();
            EditorGUILayout.Space(8);

            using (new EditorGUI.DisabledScope(_running))
            {
                if (GUILayout.Button("Run Full Pipeline", GUILayout.Height(30)))
                    RunPipeline();

                EditorGUILayout.Space(2);

                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Test Rembg Only"))
                        RunRembgOnly();
                    if (GUILayout.Button("Test Preprocess Only"))
                        RunPreprocessOnly();
                }

                EditorGUILayout.Space(2);
                if (GUILayout.Button("Run from Preprocessed PNG (bypass rembg+composite)"))
                    RunFromPreprocessedPng();
            }

            if (_running)
            {
                EditorGUILayout.Space(4);
                var rect = EditorGUILayout.GetControlRect(false, 20);
                EditorGUI.ProgressBar(rect, _progress, "");

                if (GUILayout.Button("Cancel"))
                    Cancel();
            }

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            EditorGUILayout.SelectableLabel(_status, EditorStyles.wordWrappedLabel,
                GUILayout.MinHeight(40), GUILayout.ExpandHeight(true));

            if (!string.IsNullOrEmpty(_timingLog))
            {
                EditorGUILayout.Space(8);
                EditorGUILayout.LabelField("Timing", EditorStyles.boldLabel);
                EditorGUILayout.HelpBox(_timingLog, MessageType.Info);
            }

            if (_resultMesh != null)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField("Result",
                    $"{_resultMesh.vertexCount} verts, {_resultMesh.triangles.Length / 3} tris");

                if (GUILayout.Button("Clear Preview"))
                    CleanupPreview();
            }

            // ── Keyframe-based reconstruction ──
            EditorGUILayout.Space(12);
            EditorGUILayout.LabelField("Reconstruct from Detection Keyframe", EditorStyles.boldLabel);
            EditorGUILayout.Space(4);

            using (new EditorGUILayout.HorizontalScope())
            {
                _keyframeDir = EditorGUILayout.TextField("Keyframe Directory", _keyframeDir);
                if (GUILayout.Button("Browse", GUILayout.Width(60)))
                {
                    string dir = EditorUtility.OpenFolderPanel("Select Keyframe Directory", _keyframeDir, "");
                    if (!string.IsNullOrEmpty(dir)) _keyframeDir = dir;
                }
            }

            using (new EditorGUI.DisabledScope(string.IsNullOrEmpty(_keyframeDir) || _running))
            {
                if (GUILayout.Button("Load Detections"))
                    LoadDetections();
            }

            if (_detections != null && _detections.Count > 0)
            {
                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField($"{_detections.Count} detection(s) found:", EditorStyles.miniLabel);

                _keyframeScroll = EditorGUILayout.BeginScrollView(_keyframeScroll, GUILayout.MaxHeight(150));
                for (int i = 0; i < _detections.Count; i++)
                {
                    var d = _detections[i];
                    bool selected = i == _selectedDetectionIdx;
                    string label = $"[{i}] {d.label} (conf={d.confidence:F2}, bbox={d.bbox.width:F0}x{d.bbox.height:F0})";
                    if (GUILayout.Toggle(selected, label, EditorStyles.radioButton) && !selected)
                    {
                        _selectedDetectionIdx = i;
                        LoadCroppedPreview(i);
                    }
                }
                EditorGUILayout.EndScrollView();

                if (_croppedPreview != null)
                {
                    EditorGUILayout.Space(4);
                    EditorGUILayout.LabelField("Cropped Preview:", EditorStyles.miniLabel);
                    var previewRect = GUILayoutUtility.GetRect(128, 128, GUILayout.ExpandWidth(false));
                    EditorGUI.DrawPreviewTexture(previewRect, _croppedPreview, null, ScaleMode.ScaleToFit);
                }

                EditorGUILayout.Space(4);
                using (new EditorGUI.DisabledScope(_selectedDetectionIdx < 0 || _running))
                {
                    if (GUILayout.Button("Reconstruct Selected Detection", GUILayout.Height(28)))
                        RunFromKeyframe();
                }
            }
            else if (_detections != null)
            {
                EditorGUILayout.HelpBox("No detections found in the specified directory.", MessageType.Warning);
            }
        }

        private void DrawModelStatus()
        {
            string onnxDir = Path.Combine(Application.streamingAssetsPath, "ObjectReconstruction");
            bool hasRembg = File.Exists(Path.Combine(onnxDir, "u2netp.onnx"));
            bool hasPart1 = File.Exists(Path.Combine(onnxDir, "triposr_part1.onnx"));
            bool hasPart2 = File.Exists(Path.Combine(onnxDir, "triposr_part2.onnx"));
            bool hasDecoder = File.Exists(Path.Combine(onnxDir, "nerf_decoder.onnx"));

            EditorGUILayout.LabelField("Models (.onnx)", EditorStyles.boldLabel);
            StatusLabel("  u2netp.onnx", hasRembg);
            StatusLabel("  triposr_part1.onnx", hasPart1);
            StatusLabel("  triposr_part2.onnx", hasPart2);
            StatusLabel("  nerf_decoder.onnx", hasDecoder);

            if (!hasRembg || !hasPart1 || !hasPart2 || !hasDecoder)
                EditorGUILayout.HelpBox("Run 'Deploy Models' in RoomScan Setup Wizard first.", MessageType.Warning);
        }

        private void DrawShaderStatus()
        {
            bool ok = _triplaneShader != null && _surfaceNetsShader != null
                && _marchingCubesShader != null && _postprocessShader != null
                && _vertexColorShader != null && _projectedTextureShader != null;
            if (!ok)
                EditorGUILayout.HelpBox("Shaders not found at expected package path.", MessageType.Error);
        }

        private static void StatusLabel(string label, bool ok)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label);
            GUILayout.Label(ok ? "OK" : "MISSING",
                ok ? EditorStyles.boldLabel : EditorStyles.miniLabel);
            EditorGUILayout.EndHorizontal();
        }

        private async void RunPipeline()
        {
            if (_testImage == null)
            {
                EditorUtility.DisplayDialog("Missing Image", "Assign a test image first.", "OK");
                return;
            }

            var module = FindOrCreateModule();
            if (module == null) return;

            SyncModuleConfig(module);

            _running = true;
            _cts = new CancellationTokenSource();
            AsyncHelper.SuppressYields = true;
            _timingLog = "";
            _stepSw = Stopwatch.StartNew();
            var totalSw = Stopwatch.StartNew();
            string _prevStep = null;

            // Maps status strings to approximate progress for the bar
            float ProgressFor(string s)
            {
                if (s.Contains("Loading")) return 0.1f;
                if (s.Contains("background")) return 0.25f;
                if (s.Contains("reconstruction") || s.Contains("Reconstruction")) return 0.45f;
                if (s.Contains("mesh") || s.Contains("Mesh")) return 0.7f;
                if (s.Contains("Done")) return 1f;
                return _progress;
            }

            void OnStatus(string status)
            {
                if (_prevStep != null)
                    AppendTiming($"{_prevStep} {_stepSw.ElapsedMilliseconds:F0}ms");
                _prevStep = status;
                _stepSw.Restart();
                SetStatus(status, ProgressFor(status));
            }

            module.StatusChanged += OnStatus;
            try
            {
                SetStatus("Starting pipeline...", 0.05f);
                var mesh = await module.ReconstructAsync(_testImage, _cts.Token);

                if (_prevStep != null)
                    AppendTiming($"{_prevStep} {_stepSw.ElapsedMilliseconds:F0}ms");

                totalSw.Stop();
                AppendTiming($"--- TOTAL: {totalSw.ElapsedMilliseconds:F0}ms ---");

                if (mesh != null)
                {
                    var projTex = module.LastPreprocessedImage;
                    ShowMeshPreview(mesh, module.CreateMaterial(projTex));
                    SetStatus($"Done! {mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris", 1f);
                    Debug.Log($"[ReconstructionTest] Pipeline complete in {totalSw.ElapsedMilliseconds}ms, " +
                        $"projectedTex={projTex != null}");
                }
                else
                {
                    SetStatus("Pipeline returned no mesh", 0f);
                }
            }
            catch (OperationCanceledException)
            {
                SetStatus("Cancelled", 0f);
            }
            catch (Exception e)
            {
                SetStatus($"Error: {e.Message}", 0f);
                Debug.LogException(e);
            }
            finally
            {
                module.StatusChanged -= OnStatus;
                AsyncHelper.SuppressYields = false;
                _running = false;
                _cts?.Dispose();
                _cts = null;
                Repaint();
            } 
        }

        private async void RunRembgOnly()
        {
            if (!Validate()) return;

            _running = true;
            _cts = new CancellationTokenSource();
            AsyncHelper.SuppressYields = true;
            _timingLog = "";
            OrtRembgModel rembg = null;

            try
            {
                rembg = new OrtRembgModel();

                SetStatus("Loading u2netp...", 0.1f);
                var sw = Stopwatch.StartNew();
                await rembg.LoadAsync(_executionProvider, false, _cts.Token);
                AppendTiming($"Load u2netp: {sw.ElapsedMilliseconds:F0}ms");

                SetStatus("Running rembg inference...", 0.4f);
                var readable = MakeReadableEditor(_testImage);
                sw.Restart();
                var mask = await rembg.InferAsync(readable, _cts.Token);
                AppendTiming($"Rembg inference: {sw.ElapsedMilliseconds:F0}ms");

                int maskW = 320, maskH = 320;
                Debug.Log($"[ReconstructionTest] Rembg mask: {maskW}x{maskH}");

                string debugDir = Path.Combine(Application.dataPath, "../debug_reconstruction");
                Directory.CreateDirectory(debugDir);
                SaveMaskAsPng(mask, maskW, maskH,
                    Path.Combine(debugDir, "unity_rembg_mask.png"));
                AppendTiming($"Mask saved to {debugDir}");

                if (readable != _testImage) DestroyImmediate(readable);
                SetStatus("Rembg OK!", 1f);
            }
            catch (Exception e)
            {
                SetStatus($"Error: {e.Message}", 0f);
                Debug.LogException(e);
            }
            finally
            {
                AsyncHelper.SuppressYields = false;
                rembg?.Dispose();
                _running = false;
                _cts?.Dispose();
                _cts = null;
                Repaint();
            }
        }

        private async void RunPreprocessOnly()
        {
            if (!Validate()) return;

            _running = true;
            _cts = new CancellationTokenSource();
            AsyncHelper.SuppressYields = true;
            _timingLog = "";
            ReconstructionPipeline pipeline = null;

            try
            {
                pipeline = CreatePipeline();

                SetStatus("Loading models...", 0.1f);
                await pipeline.LoadModelsAsync(_cts.Token);

                SetStatus("Running preprocess...", 0.4f);
                var sw = Stopwatch.StartNew();
                var result = await pipeline.PreprocessAsync(_testImage, _cts.Token);
                AppendTiming($"Preprocess: {sw.ElapsedMilliseconds:F0}ms");

                Debug.Log($"[ReconstructionTest] Preprocessed NCHW array length: {result.Length}");

                string debugDir = Path.Combine(Application.dataPath, "../debug_reconstruction");
                Directory.CreateDirectory(debugDir);
                SaveNCHWAsPng(result, 512, 512, Path.Combine(debugDir, "unity_rembg_composite.png"));
                AppendTiming($"Composite saved to {debugDir}");

                SetStatus("Preprocess OK!", 1f);
            }
            catch (Exception e)
            {
                SetStatus($"Error: {e.Message}", 0f);
                Debug.LogException(e);
            }
            finally
            {
                AsyncHelper.SuppressYields = false;
                pipeline?.Dispose();
                _running = false;
                _cts?.Dispose();
                _cts = null;
                Repaint();
            }
        }

        private async void RunFromPreprocessedPng()
        {
            string pngPath = EditorUtility.OpenFilePanel(
                "Select preprocessed composite PNG", "", "png");
            if (string.IsNullOrEmpty(pngPath)) return;

            if (_triplaneShader == null || _surfaceNetsShader == null
                || _marchingCubesShader == null || _postprocessShader == null)
            {
                EditorUtility.DisplayDialog("Missing Shaders", "Compute shaders not found.", "OK");
                return;
            }

            _running = true;
            _cts = new CancellationTokenSource();
            AsyncHelper.SuppressYields = true;
            _timingLog = "";
            var totalSw = Stopwatch.StartNew();
            ReconstructionPipeline pipeline = null;

            try
            {
                pipeline = CreatePipeline();

                SetStatus("Loading models (triposr + decoder)...", 0.1f);
                var sw = Stopwatch.StartNew();
                await pipeline.LoadModelsAsync(_cts.Token);
                AppendTiming($"Load models: {sw.ElapsedMilliseconds:F0}ms");

                int imgSize = pipeline.ModelImageSize;

                var pngBytes = File.ReadAllBytes(pngPath);
                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                tex.LoadImage(pngBytes);
                AppendTiming($"Loaded PNG: {tex.width}x{tex.height} from {Path.GetFileName(pngPath)}");

                var resized = new Texture2D(imgSize, imgSize, TextureFormat.RGB24, false);
                var rt = RenderTexture.GetTemporary(imgSize, imgSize, 0, RenderTextureFormat.ARGB32);
                Graphics.Blit(tex, rt);
                RenderTexture.active = rt;
                resized.ReadPixels(new Rect(0, 0, imgSize, imgSize), 0, 0);
                resized.Apply();
                RenderTexture.active = null;
                RenderTexture.ReleaseTemporary(rt);
                DestroyImmediate(tex);

                float[] preprocessed = ImagePreprocessor.TextureToNCHW(resized, imgSize);
                DestroyImmediate(resized);
                AppendTiming($"Converted to NCHW float[] [1,3,{imgSize},{imgSize}]");

                SetStatus("Running TripoSR forward pass...", 0.35f);
                sw.Restart();
                await pipeline.RunForwardAsync(preprocessed, _cts.Token);
                AppendTiming($"Forward pass: {sw.ElapsedMilliseconds:F0}ms");

                SetStatus("Extracting mesh + vertex colors...", 0.60f);
                sw.Restart();
                var mesh = await pipeline.ExtractMeshAsync(_cts.Token);
                AppendTiming($"Mesh extraction: {sw.ElapsedMilliseconds:F0}ms");

                totalSw.Stop();
                AppendTiming($"--- TOTAL: {totalSw.ElapsedMilliseconds:F0}ms ---");

                ShowMeshPreview(mesh);
                SetStatus($"Done! {mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris", 1f);
            }
            catch (OperationCanceledException)
            {
                SetStatus("Cancelled", 0f);
            }
            catch (Exception e)
            {
                SetStatus($"Error: {e.Message}", 0f);
                Debug.LogException(e);
            }
            finally
            {
                AsyncHelper.SuppressYields = false;
                pipeline?.Dispose();
                _running = false;
                _cts?.Dispose();
                _cts = null;
                Repaint();
            }
        }

        private ObjectReconstructionModule FindOrCreateModule()
        {
            var module = FindAnyObjectByType<ObjectReconstructionModule>();
            if (module != null) return module;

            var go = new GameObject("[ReconstructionTest] Module") { hideFlags = HideFlags.DontSave };
            module = go.AddComponent<ObjectReconstructionModule>();

            module.triplaneGridSampleShader = _triplaneShader;
            module.densitySurfaceNetsShader = _surfaceNetsShader;
            module.densityMarchingCubesShader = _marchingCubesShader;
            module.decoderPostprocessShader = _postprocessShader;
            module.vertexColorShader = _vertexColorShader;
            module.projectedTextureShader = _projectedTextureShader;

            return module;
        }

        private void SyncModuleConfig(ObjectReconstructionModule module)
        {
            var so = new SerializedObject(module);
            so.FindProperty("gridResolution").intValue = _gridResolution;
            so.FindProperty("meshAlgorithm").intValue = (int)_meshAlgorithm;
            so.FindProperty("executionProvider").intValue = (int)_executionProvider;
            so.FindProperty("densitySmoothPasses").intValue = _densitySmoothPasses;
            so.FindProperty("laplacianSmoothIterations").intValue = _laplacianSmoothIterations;
            so.FindProperty("laplacianSmoothLambda").floatValue = _laplacianSmoothLambda;
            so.FindProperty("meshExtractionBackend").intValue = (int)_meshExtractionBackend;

            var mobileProp = so.FindProperty("mobileOptimized");
            bool savedMobile = mobileProp.boolValue;
            mobileProp.boolValue = false;

            so.ApplyModifiedPropertiesWithoutUndo();
            module.ResetPipeline();

            // Restore serialized value so we don't dirty the scene
            so.Update();
            so.FindProperty("mobileOptimized").boolValue = savedMobile;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        private ReconstructionPipeline CreatePipeline()
        {
            return new ReconstructionPipeline(
                gridResolution: _gridResolution,
                densityThreshold: 25f,
                triplaneShader: _triplaneShader,
                surfaceNetsShader: _surfaceNetsShader,
                marchingCubesShader: _marchingCubesShader,
                postprocessShader: _postprocessShader,
                meshAlgorithm: _meshAlgorithm,
                preloadModels: true,
                executionProvider: _executionProvider,
                densitySmoothPasses: _densitySmoothPasses,
                meshBackend: _meshExtractionBackend,
                laplacianSmoothIterations: _laplacianSmoothIterations,
                laplacianSmoothLambda: _laplacianSmoothLambda);
        }

        private bool Validate()
        {
            if (_testImage == null)
            {
                EditorUtility.DisplayDialog("Missing Image", "Assign a test image first.", "OK");
                return false;
            }
            if (_triplaneShader == null || _surfaceNetsShader == null
                || _marchingCubesShader == null || _postprocessShader == null)
            {
                EditorUtility.DisplayDialog("Missing Shaders", "Compute shaders not found.", "OK");
                return false;
            }
            return true;
        }

        private static Texture2D MakeReadableEditor(Texture2D src)
        {
            if (src.isReadable) return src;
            var rt = RenderTexture.GetTemporary(src.width, src.height, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(src, rt);
            RenderTexture.active = rt;
            var copy = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false);
            copy.ReadPixels(new Rect(0, 0, src.width, src.height), 0, 0);
            copy.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
            return copy;
        }

        private void ShowMeshPreview(Mesh mesh, Material material = null)
        {
            CleanupPreview();
            _resultMesh = mesh;

            _previewObj = new GameObject("[ReconstructionTest] Preview");
            _previewObj.hideFlags = HideFlags.DontSave;
            var mf = _previewObj.AddComponent<MeshFilter>();
            var mr = _previewObj.AddComponent<MeshRenderer>();
            mf.sharedMesh = mesh;

            if (material != null)
            {
                mr.sharedMaterial = material;
            }
            else
            {
                var shader = _vertexColorShader
                             ?? Shader.Find("Universal Render Pipeline/Lit")
                             ?? Shader.Find("Standard");
                mr.sharedMaterial = new Material(shader);
            }

            _previewObj.transform.localScale = Vector3.one * 0.5f;

            Selection.activeGameObject = _previewObj;
            SceneView.lastActiveSceneView?.FrameSelected();
        }

        private void CleanupPreview()
        {
            if (_previewObj != null)
                DestroyImmediate(_previewObj);
            _previewObj = null;
            _resultMesh = null;
        }

        private void Cancel()
        {
            _cts?.Cancel();
        }

        private void SetStatus(string msg, float pct)
        {
            _status = msg;
            _progress = pct;
            Repaint();
        }

        private void AppendTiming(string line)
        {
            if (!string.IsNullOrEmpty(_timingLog)) _timingLog += "\n";
            _timingLog += line;
            Repaint();
        }

        private static void SaveNCHWAsPng(float[] data, int w, int h, string path)
        {
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            var pixels = new Color32[w * h];
            for (int py = 0; py < h; py++)
            for (int px = 0; px < w; px++)
            {
                int ty = h - 1 - py;
                float r = Mathf.Clamp01(data[0 * h * w + py * w + px]);
                float g = Mathf.Clamp01(data[1 * h * w + py * w + px]);
                float b = Mathf.Clamp01(data[2 * h * w + py * w + px]);
                pixels[ty * w + px] = new Color32(
                    (byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 255);
            }
            tex.SetPixels32(pixels);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            DestroyImmediate(tex);
            Debug.Log($"[ReconstructionTest] Saved composite PNG: {path} ({w}x{h})");
        }

        private static void SaveMaskAsPng(float[] data, int w, int h, string path)
        {
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            var pixels = new Color32[w * h];
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int ty = h - 1 - y;
                byte v = (byte)(Mathf.Clamp01(data[y * w + x]) * 255);
                pixels[ty * w + x] = new Color32(v, v, v, 255);
            }
            tex.SetPixels32(pixels);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            DestroyImmediate(tex);
            Debug.Log($"[ReconstructionTest] Saved mask PNG: {path} ({w}x{h})");
        }

        // ── Keyframe-based reconstruction ──

        private void LoadDetections()
        {
            _detections = KeyframeLoader.LoadDetections(_keyframeDir);
            _frames = KeyframeLoader.LoadFrames(_keyframeDir);
            _selectedDetectionIdx = -1;
            if (_croppedPreview != null) DestroyImmediate(_croppedPreview);
            _croppedPreview = null;

            if (_detections.Count > 0)
                SetStatus($"Loaded {_detections.Count} detections, {_frames.Count} frames", 0);
            else
                SetStatus("No detections found in directory", 0);
        }

        private async void LoadCroppedPreview(int idx)
        {
            if (_detections == null || idx < 0 || idx >= _detections.Count) return;
            if (_croppedPreview != null) DestroyImmediate(_croppedPreview);

            _croppedPreview = await KeyframeLoader.LoadAndCropAsync(
                _keyframeDir, _detections[idx], denoise: false);
            Repaint();
        }

        private async void RunFromKeyframe()
        {
            if (_detections == null || _selectedDetectionIdx < 0 || _running) return;
            var detection = _detections[_selectedDetectionIdx];

            _running = true;
            _cts = new CancellationTokenSource();
            _timingLog = "";
            var totalSw = Stopwatch.StartNew();

            try
            {
                SetStatus("Loading and cropping keyframe...", 0.1f);
                var cropped = await KeyframeLoader.LoadAndCropAsync(
                    _keyframeDir, detection, denoise: true);

                if (cropped == null)
                {
                    SetStatus("Failed to load/crop keyframe image", 0);
                    return;
                }

                AppendTiming($"Crop+denoise: {totalSw.ElapsedMilliseconds}ms " +
                    $"({cropped.width}x{cropped.height})");

                SetStatus("Running reconstruction pipeline...", 0.3f);

                var module = FindOrCreateModule();
                SyncModuleConfig(module);

                var mesh = await module.ReconstructAsync(cropped, _cts.Token);
                DestroyImmediate(cropped);

                totalSw.Stop();
                AppendTiming($"--- TOTAL: {totalSw.ElapsedMilliseconds}ms ---");

                if (mesh != null)
                {
                    var projTex = module.LastPreprocessedImage;
                    ShowMeshPreview(mesh, module.CreateMaterial(projTex));
                    SetStatus($"Done! {mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris " +
                        $"(label={detection.label})", 1f);
                }
                else
                {
                    SetStatus("Reconstruction returned null", 0);
                }
            }
            catch (OperationCanceledException) { SetStatus("Cancelled", 0); }
            catch (Exception e)
            {
                SetStatus($"Error: {e.Message}", 0);
                Debug.LogException(e);
            }
            finally
            {
                _running = false;
                _cts?.Dispose();
                _cts = null;
            }
        }
    }
}
#endif
