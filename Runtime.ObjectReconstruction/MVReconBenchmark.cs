#if HAS_ONNXRUNTIME
using System.IO;
using System.Linq;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Drop-in benchmark for multi-view reconstruction on Quest.
    /// Attach to any GameObject, set runOnStart=true for automated benchmarking.
    /// All timing is logged via Logger.Info, visible in logcat.
    /// </summary>
    internal class MVReconBenchmark : MonoBehaviour
    {
        [Tooltip("Run benchmark automatically on Start")]
        public bool runOnStart = true;

        [Tooltip("Which test UID to benchmark (empty = first available)")]
        public string testUid = "";

        [Tooltip("Density threshold for mesh extraction")]
        [Range(0.1f, 0.95f)]
        public float threshold = 0.5f;

        [Tooltip("Execution provider: CPU, XNNPACK, CoreML, etc.")]
        public ExecutionProvider executionProvider = ExecutionProvider.XNNPACK;

        [Tooltip("Use mobile-optimized ORT session options")]
        public bool mobileOptimized = true;

        private ObjectReconstructionModule _module;
        private bool _benchmarkDone;

        private void Start()
        {
            if (runOnStart)
                RunBenchmark();
        }

        public async void RunBenchmark()
        {
            if (_benchmarkDone) return;

            _module = GetComponent<ObjectReconstructionModule>();
            if (_module == null)
                _module = gameObject.AddComponent<ObjectReconstructionModule>();

            string uid = testUid;
            if (string.IsNullOrEmpty(uid))
            {
                string testDir = Path.Combine(Application.streamingAssetsPath,
                    "ObjectReconstruction", "MVReconTest");
                if (Directory.Exists(testDir))
                {
                    var dirs = Directory.GetDirectories(testDir);
                    if (dirs.Length > 0)
                        uid = Path.GetFileName(dirs[0]);
                }
            }

            if (string.IsNullOrEmpty(uid))
            {
                Logger.Error("[MVReconBenchmark] No test data found");
                return;
            }

            Logger.Info($"[MVReconBenchmark] Starting benchmark: uid={uid} " +
                $"EP={executionProvider} mobile={mobileOptimized} threshold={threshold}");

            var mesh = await _module.TestMVReconAsync(uid, threshold);

            if (mesh != null && mesh.vertexCount > 0)
            {
                Logger.Info($"[MVReconBenchmark] Result: {mesh.vertexCount} verts, " +
                    $"{mesh.triangles.Length / 3} tris");

                var go = new GameObject("MVRecon_Result");
                var mf = go.AddComponent<MeshFilter>();
                mf.sharedMesh = mesh;
                var mr = go.AddComponent<MeshRenderer>();
                mr.sharedMaterial = new Material(
                    Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard"));
                go.transform.localScale = Vector3.one * 0.3f;
                go.transform.position = Camera.main != null
                    ? Camera.main.transform.position + Camera.main.transform.forward * 0.5f
                    : Vector3.zero;
            }
            else
            {
                Logger.Warning("[MVReconBenchmark] Empty mesh produced");
            }

            _benchmarkDone = true;
        }
    }
}
#endif
