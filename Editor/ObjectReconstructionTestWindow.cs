#if HAS_AI_INFERENCE
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Genesis.RoomScan.ObjectReconstruction;
using Unity.InferenceEngine;
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

        private ComputeShader _triplaneShader;
        private ComputeShader _surfaceNetsShader;

        private string _status = "Ready";
        private float _progress;
        private bool _running;
        private CancellationTokenSource _cts;

        private Mesh _resultMesh;
        private GameObject _previewObj;

        private string _timingLog = "";

        private const string SHADER_DIR = "Packages/com.genesis.roomscan/Runtime.ObjectReconstruction/Shaders/";

        private void OnEnable()
        {
            _triplaneShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                SHADER_DIR + "TriplaneGridSample.compute");
            _surfaceNetsShader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                SHADER_DIR + "DensitySurfaceNets.compute");
        }

        private void OnDisable()
        {
            Cancel();
            CleanupPreview();
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
            bool hasTriposr = File.Exists(Path.Combine(sentisDir, "triposr_uint8.sentis"));
            bool hasDecoder = File.Exists(Path.Combine(sentisDir, "nerf_decoder.sentis"));

            EditorGUILayout.LabelField("Models", EditorStyles.boldLabel);
            StatusLabel("  u2netp.sentis", hasRembg);
            StatusLabel("  triposr_uint8.sentis", hasTriposr);
            StatusLabel("  nerf_decoder.sentis", hasDecoder);

            if (!hasRembg || !hasTriposr || !hasDecoder)
                EditorGUILayout.HelpBox("Run 'Convert Models' in RoomScan Setup Wizard first.", MessageType.Warning);
        }

        private void DrawShaderStatus()
        {
            bool ok = _triplaneShader != null && _surfaceNetsShader != null;
            if (!ok)
                EditorGUILayout.HelpBox("Compute shaders not found at expected package path.", MessageType.Error);
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
                var sceneCodes = await pipeline.RunForwardAsync(preprocessed, _cts.Token);
                preprocessed.Dispose();
                float forwardMs = sw.ElapsedMilliseconds;
                AppendTiming($"Forward pass: {forwardMs:F0}ms");

                SetStatus("Extracting mesh (surface nets)...", 0.70f);
                sw.Restart();
                var mesh = await pipeline.ExtractMeshAsync(sceneCodes, _cts.Token);
                sceneCodes.Dispose();
                float meshMs = sw.ElapsedMilliseconds;
                AppendTiming($"Mesh extraction: {meshMs:F0}ms");

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

                var maskData = mask.DownloadToArray();
                int maskW = mask.shape[3];
                int maskH = mask.shape[2];
                Debug.Log($"[ReconstructionTest] Rembg mask: {maskW}x{maskH}, " +
                    $"range=[{Mathf.Min(maskData)}, {Mathf.Max(maskData)}]");

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

        private ReconstructionPipeline CreatePipeline()
        {
            return new ReconstructionPipeline(
                forwardLayersPerFrame: 10,
                decoderLayersPerFrame: 20,
                gridSampleChunksPerFrame: 15,
                gridResolution: 128,
                densityThreshold: 25f,
                triplaneShader: _triplaneShader,
                surfaceNetsShader: _surfaceNetsShader);
        }

        private bool Validate()
        {
            if (_testImage == null)
            {
                EditorUtility.DisplayDialog("Missing Image", "Assign a test image first.", "OK");
                return false;
            }
            if (_triplaneShader == null || _surfaceNetsShader == null)
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

            var shader = Shader.Find("Universal Render Pipeline/Lit")
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

        private static float Min(float[] arr)
        {
            float m = float.MaxValue;
            foreach (var v in arr) if (v < m) m = v;
            return m;
        }

        private static float Max(float[] arr)
        {
            float m = float.MinValue;
            foreach (var v in arr) if (v > m) m = v;
            return m;
        }
    }
}
#endif
