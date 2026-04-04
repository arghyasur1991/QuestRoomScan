#if HAS_AI_INFERENCE
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

        private ComputeShader _triplaneShader;
        private ComputeShader _surfaceNetsShader;
        private ComputeShader _marchingCubesShader;
        private Shader _vertexColorShader;

        private string _status = "Ready";
        private float _progress;
        private bool _running;
        private CancellationTokenSource _cts;

        private Mesh _resultMesh;
        private GameObject _previewObj;

        private string _timingLog = "";

        private ComputeShader _postprocessShader;

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

            InstallEditModeYield();
        }

        private void OnDisable()
        {
            Cancel();
            CleanupPreview();
            AsyncHelper.EditModeYield = null;
        }

        /// <summary>
        /// Hooks AsyncHelper.EditModeYield to use EditorApplication.delayCall,
        /// which genuinely yields to the editor's repaint/input loop.
        /// </summary>
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
        }

        private void DrawModelStatus()
        {
            string sentisDir = Path.Combine(Application.streamingAssetsPath, "ObjectReconstruction");
            bool hasRembg = File.Exists(Path.Combine(sentisDir, "u2netp.sentis"));
            bool hasTriposr = File.Exists(Path.Combine(sentisDir, "triposr.sentis"));
            bool hasDecoder = File.Exists(Path.Combine(sentisDir, "nerf_decoder.sentis"));

            EditorGUILayout.LabelField("Models", EditorStyles.boldLabel);
            StatusLabel("  u2netp.sentis", hasRembg);
            StatusLabel("  triposr.sentis", hasTriposr);
            StatusLabel("  nerf_decoder.sentis", hasDecoder);

            if (!hasRembg || !hasTriposr || !hasDecoder)
                EditorGUILayout.HelpBox("Run 'Convert Models' in RoomScan Setup Wizard first.", MessageType.Warning);
        }

        private void DrawShaderStatus()
        {
            bool ok = _triplaneShader != null && _surfaceNetsShader != null
                && _marchingCubesShader != null && _postprocessShader != null
                && _vertexColorShader != null;
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
            if (!Validate()) return;

            _running = true;
            _cts = new CancellationTokenSource();
            _timingLog = "";
            var totalSw = Stopwatch.StartNew();
            ReconstructionPipeline pipeline = null;

            try
            {
                pipeline = CreatePipeline();

                SetStatus("Loading models...", 0.05f);
                var sw = Stopwatch.StartNew();
                await pipeline.LoadModelsAsync(_cts.Token);
                float loadMs = sw.ElapsedMilliseconds;
                AppendTiming($"Load models: {loadMs:F0}ms");

                SetStatus("Removing background (rembg)...", 0.15f);
                sw.Restart();
                var preprocessed = await pipeline.PreprocessAsync(_testImage, _cts.Token);
                float rembgMs = sw.ElapsedMilliseconds;
                AppendTiming($"Preprocess (rembg + composite): {rembgMs:F0}ms");

                SetStatus("Running TripoSR forward pass...", 0.35f);
                sw.Restart();
                await pipeline.RunForwardAsync(preprocessed, _cts.Token);
                preprocessed.Dispose();
                float forwardMs = sw.ElapsedMilliseconds;
                AppendTiming($"Forward pass: {forwardMs:F0}ms");

                SetStatus("Extracting mesh + vertex colors...", 0.60f);
                sw.Restart();
                var mesh = await pipeline.ExtractMeshAsync(_cts.Token);
                float meshMs = sw.ElapsedMilliseconds;
                AppendTiming($"Mesh + vertex color extraction: {meshMs:F0}ms");

                totalSw.Stop();
                AppendTiming($"--- TOTAL: {totalSw.ElapsedMilliseconds:F0}ms ---");

                ShowMeshPreview(mesh);
                SetStatus($"Done! {mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris", 1f);
                Debug.Log($"[ReconstructionTest] Pipeline complete in {totalSw.ElapsedMilliseconds}ms");
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
                pipeline?.Dispose();
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
            _timingLog = "";
            RembgModel rembg = null;

            try
            {
                rembg = new RembgModel();

                SetStatus("Loading u2netp...", 0.1f);
                var sw = Stopwatch.StartNew();
                await rembg.LoadAsync(_cts.Token);
                AppendTiming($"Load u2netp: {sw.ElapsedMilliseconds:F0}ms");

                SetStatus("Running rembg inference...", 0.4f);
                var readable = MakeReadableEditor(_testImage);
                sw.Restart();
                var mask = await rembg.InferAsync(readable, _cts.Token);
                AppendTiming($"Rembg inference: {sw.ElapsedMilliseconds:F0}ms");

                int maskW = mask.shape[3];
                int maskH = mask.shape[2];
                var maskData = mask.DownloadToArray();
                Debug.Log($"[ReconstructionTest] Rembg mask: {maskW}x{maskH}");

                string debugDir = Path.Combine(Application.dataPath, "../debug_reconstruction");
                Directory.CreateDirectory(debugDir);
                SaveMaskAsPng(maskData, maskW, maskH,
                    Path.Combine(debugDir, "unity_rembg_mask.png"));
                AppendTiming($"Mask saved to {debugDir}");

                mask.Dispose();
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

                Debug.Log($"[ReconstructionTest] Preprocessed tensor shape: {result.shape}");

                string debugDir = Path.Combine(Application.dataPath, "../debug_reconstruction");
                Directory.CreateDirectory(debugDir);
                SaveTensorAsCompositePng(result, Path.Combine(debugDir, "unity_rembg_composite.png"));
                AppendTiming($"Composite saved to {debugDir}");

                result.Dispose();
                SetStatus("Preprocess OK!", 1f);
            }
            catch (Exception e)
            {
                SetStatus($"Error: {e.Message}", 0f);
                Debug.LogException(e);
            }
            finally
            {
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
                "Select 512x512 preprocessed composite PNG", "", "png");
            if (string.IsNullOrEmpty(pngPath)) return;

            if (_triplaneShader == null || _surfaceNetsShader == null
                || _marchingCubesShader == null || _postprocessShader == null)
            {
                EditorUtility.DisplayDialog("Missing Shaders", "Compute shaders not found.", "OK");
                return;
            }

            _running = true;
            _cts = new CancellationTokenSource();
            _timingLog = "";
            var totalSw = Stopwatch.StartNew();
            ReconstructionPipeline pipeline = null;

            try
            {
                // Load PNG as texture
                var pngBytes = File.ReadAllBytes(pngPath);
                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                tex.LoadImage(pngBytes);
                AppendTiming($"Loaded PNG: {tex.width}x{tex.height} from {Path.GetFileName(pngPath)}");

                if (tex.width != 512 || tex.height != 512)
                {
                    DestroyImmediate(tex);
                    SetStatus($"Error: image must be 512x512 (got {tex.width}x{tex.height})", 0f);
                    return;
                }

                // Convert to NCHW tensor [1,3,512,512] — same as pipeline.PreprocessAsync output
                var tensor = new Unity.InferenceEngine.Tensor<float>(
                    new Unity.InferenceEngine.TensorShape(1, 3, 512, 512));
                Unity.InferenceEngine.TextureConverter.ToTensor(tex, tensor,
                    new Unity.InferenceEngine.TextureTransform());
                DestroyImmediate(tex);
                AppendTiming("Converted to tensor [1,3,512,512]");

                pipeline = CreatePipeline();

                SetStatus("Loading models (triposr + decoder)...", 0.1f);
                var sw = Stopwatch.StartNew();
                await pipeline.LoadModelsAsync(_cts.Token);
                AppendTiming($"Load models: {sw.ElapsedMilliseconds:F0}ms");

                SetStatus("Running TripoSR forward pass...", 0.35f);
                sw.Restart();
                await pipeline.RunForwardAsync(tensor, _cts.Token);
                tensor.Dispose();
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
                pipeline?.Dispose();
                _running = false;
                _cts?.Dispose();
                _cts = null;
                Repaint();
            }
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
                meshAlgorithm: _meshAlgorithm);
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

        private void ShowMeshPreview(Mesh mesh)
        {
            CleanupPreview();
            _resultMesh = mesh;

            _previewObj = new GameObject("[ReconstructionTest] Preview");
            _previewObj.hideFlags = HideFlags.DontSave;
            var mf = _previewObj.AddComponent<MeshFilter>();
            var mr = _previewObj.AddComponent<MeshRenderer>();
            mf.sharedMesh = mesh;

            var shader = _vertexColorShader
                         ?? Shader.Find("Universal Render Pipeline/Lit")
                         ?? Shader.Find("Standard");
            mr.sharedMaterial = new Material(shader);
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

        private static void SaveTensorAsCompositePng(Unity.InferenceEngine.Tensor<float> tensor, string path)
        {
            int h = tensor.shape[2];
            int w = tensor.shape[3];
            var data = tensor.DownloadToArray();
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
            System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
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
            System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
            DestroyImmediate(tex);
            Debug.Log($"[ReconstructionTest] Saved mask PNG: {path} ({w}x{h})");
        }

    }
}
#endif
