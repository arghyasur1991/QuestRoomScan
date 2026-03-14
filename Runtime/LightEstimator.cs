using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    [Serializable]
    public class EstimatedLight
    {
        public Vector3 Position;
        public Vector3 Normal;
        public Color Color;
        public float Intensity;
        public float Confidence;
        public bool Frozen;
        public Light UnityLight;
    }

    public class LightEstimator : MonoBehaviour
    {
        public static LightEstimator Instance { get; private set; }

        [Header("Compute")]
        [SerializeField] public ComputeShader lightDetectionCompute;

        [Header("Detection")]
        [SerializeField, Tooltip("Luminance threshold for bright voxel candidates (after undoing exposure).")]
        [Range(0.3f, 1f)] private float brightThreshold = 0.7f;
        [SerializeField, Tooltip("Minimum color volume alpha to consider a voxel observed.")]
        [Range(0.05f, 0.5f)] private float minVoxelWeight = 0.15f;
        [SerializeField, Tooltip("Mesh cycles between detection runs.")]
        private int detectionIntervalCycles = 3;

        [Header("Clustering")]
        [SerializeField, Tooltip("Spatial radius for merging bright voxels into a single light.")]
        [Range(0.1f, 1f)] private float mergeRadius = 0.3f;
        [SerializeField, Tooltip("Maximum simultaneous detected lights.")]
        [Range(1, 24)] private int maxLights = 12;

        [Header("Position Freezing")]
        [SerializeField, Tooltip("Detection cycles to confirm a light before freezing its position.")]
        [Range(1, 20)] private int freezeConfidence = 5;

        [Header("Intensity Dynamics")]
        [SerializeField, Tooltip("Seconds to fade intensity when a light source disappears.")]
        [Range(0.5f, 10f)] private float fadeSeconds = 2f;

        [Header("Unity Lights")]
        [SerializeField, Tooltip("Default range for placed Unity lights.")]
        [Range(1f, 30f)] private float lightRange = 8f;
        [SerializeField, Tooltip("Multiplier from luminance to Unity light intensity.")]
        [Range(0.5f, 20f)] private float intensityScale = 2f;

        public ReadOnlyCollection<EstimatedLight> Lights => _lightsReadOnly;
        public Color AmbientColor { get; private set; } = Color.gray;

        public event Action LightsUpdated;
        public event Action<EstimatedLight> LightAdded;
        public event Action<EstimatedLight> LightRemoved;

        private const int MaxCandidates = 1024;
        private const int CandidateStride = 48; // 12 floats per candidate (float3+float3+float+float3+float = 44 → pad to 48 for GPU alignment)

        private int _kClearBuffers, _kDetectBright, _kComputeAmbient;
        private GraphicsBuffer _candidateBuffer;
        private GraphicsBuffer _candidateCountBuffer;
        private GraphicsBuffer _ambientAccumBuffer;

        private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ID_ColorVolume = Shader.PropertyToID("_ColorVolume");
        private static readonly int ID_VoxCount = Shader.PropertyToID("_VoxCount");
        private static readonly int ID_VoxSize = Shader.PropertyToID("_VoxSize");
        private static readonly int ID_CameraExposure = Shader.PropertyToID("_CameraExposure");
        private static readonly int ID_BrightThreshold = Shader.PropertyToID("_BrightThreshold");
        private static readonly int ID_MinVoxelWeight = Shader.PropertyToID("_MinVoxelWeight");
        private static readonly int ID_MaxCandidates = Shader.PropertyToID("_MaxCandidates");
        private static readonly int ID_Candidates = Shader.PropertyToID("_Candidates");
        private static readonly int ID_CandidateCount = Shader.PropertyToID("_CandidateCount");
        private static readonly int ID_AmbientAccumInt = Shader.PropertyToID("_AmbientAccumInt");

        private readonly List<EstimatedLight> _lights = new();
        private ReadOnlyCollection<EstimatedLight> _lightsReadOnly;
        private readonly List<LightCluster> _clusters = new();
        private GameObject _lightContainer;

        private int _meshCyclesSinceDetection;
        private volatile bool _readbackPending;
        private float _lastDetectionTime;

        private void Awake()
        {
            Instance = this;
            _lightsReadOnly = _lights.AsReadOnly();
        }

        private void Start()
        {
            if (lightDetectionCompute == null)
            {
                Debug.LogError("[LightEstimator] lightDetectionCompute not assigned");
                enabled = false;
                return;
            }

            _kClearBuffers = lightDetectionCompute.FindKernel("ClearBuffers");
            _kDetectBright = lightDetectionCompute.FindKernel("DetectBrightVoxels");
            _kComputeAmbient = lightDetectionCompute.FindKernel("ComputeAmbient");

            _candidateBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxCandidates, CandidateStride);
            _candidateCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, 4);
            _ambientAccumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 4, 4);

            _lightContainer = new GameObject("[EstimatedLights]");
            _lightContainer.transform.SetParent(transform);

            Debug.Log("[LightEstimator] Initialized");
        }

        private void OnDestroy()
        {
            _candidateBuffer?.Dispose();
            _candidateCountBuffer?.Dispose();
            _ambientAccumBuffer?.Dispose();
            if (_lightContainer != null) Destroy(_lightContainer);
        }

        /// <summary>
        /// Called by RoomScanner after each mesh extraction cycle.
        /// </summary>
        public void OnMeshCycleComplete()
        {
            _meshCyclesSinceDetection++;
            if (_meshCyclesSinceDetection < detectionIntervalCycles) return;
            if (_readbackPending) return;
            _meshCyclesSinceDetection = 0;

            var vol = VolumeIntegrator.Instance;
            if (vol == null || vol.Volume == null || vol.ColorVolume == null) return;

            DispatchDetection(vol);
        }

        private void DispatchDetection(VolumeIntegrator vol)
        {
            var cs = lightDetectionCompute;
            var voxCount = vol.VoxelCount;

            // Clear
            cs.SetBuffer(_kClearBuffers, ID_CandidateCount, _candidateCountBuffer);
            cs.SetBuffer(_kClearBuffers, ID_AmbientAccumInt, _ambientAccumBuffer);
            cs.Dispatch(_kClearBuffers, 1, 1, 1);

            // Shared params
            cs.SetInts(ID_VoxCount, voxCount.x, voxCount.y, voxCount.z);
            cs.SetFloat(ID_VoxSize, vol.VoxelSize);
            cs.SetFloat(ID_CameraExposure, vol.CameraExposure);
            cs.SetFloat(ID_BrightThreshold, brightThreshold);
            cs.SetFloat(ID_MinVoxelWeight, minVoxelWeight);
            cs.SetInt(ID_MaxCandidates, MaxCandidates);

            // DetectBrightVoxels
            cs.SetTexture(_kDetectBright, ID_TsdfVolume, vol.Volume);
            cs.SetTexture(_kDetectBright, ID_ColorVolume, vol.ColorVolume);
            cs.SetBuffer(_kDetectBright, ID_Candidates, _candidateBuffer);
            cs.SetBuffer(_kDetectBright, ID_CandidateCount, _candidateCountBuffer);

            int gx = CeilDiv(voxCount.x, 4);
            int gy = CeilDiv(voxCount.y, 4);
            int gz = CeilDiv(voxCount.z, 4);
            cs.Dispatch(_kDetectBright, gx, gy, gz);

            // ComputeAmbient
            cs.SetTexture(_kComputeAmbient, ID_ColorVolume, vol.ColorVolume);
            cs.SetBuffer(_kComputeAmbient, ID_AmbientAccumInt, _ambientAccumBuffer);
            cs.Dispatch(_kComputeAmbient, gx, gy, gz);

            // Async readback both buffers
            _readbackPending = true;
            _lastDetectionTime = Time.time;

            AsyncGPUReadback.Request(_candidateCountBuffer, OnCountReadback);
        }

        private void OnCountReadback(AsyncGPUReadbackRequest req)
        {
            if (req.hasError)
            {
                _readbackPending = false;
                return;
            }

            int count = Mathf.Min(req.GetData<int>()[0], MaxCandidates);

            if (count == 0)
            {
                // Still read ambient
                AsyncGPUReadback.Request(_ambientAccumBuffer, r => OnAmbientReadback(r, null, 0));
                return;
            }

            int byteCount = count * CandidateStride;
            AsyncGPUReadback.Request(_candidateBuffer, byteCount, 0, r => OnCandidateReadback(r, count));
        }

        private void OnCandidateReadback(AsyncGPUReadbackRequest req, int count)
        {
            if (req.hasError)
            {
                _readbackPending = false;
                return;
            }

            var raw = req.GetData<CandidateRaw>();
            var candidates = new CandidateRaw[count];
            for (int i = 0; i < count; i++)
                candidates[i] = raw[i];

            AsyncGPUReadback.Request(_ambientAccumBuffer, r => OnAmbientReadback(r, candidates, count));
        }

        private void OnAmbientReadback(AsyncGPUReadbackRequest req, CandidateRaw[] candidates, int count)
        {
            if (!req.hasError)
            {
                var accum = req.GetData<uint>();
                float totalW = accum[3] / 65536f;
                if (totalW > 0.01f)
                {
                    float r = (accum[0] / 65536f) / totalW;
                    float g = (accum[1] / 65536f) / totalW;
                    float b = (accum[2] / 65536f) / totalW;
                    AmbientColor = new Color(r, g, b, 1f);
                    RenderSettings.ambientLight = AmbientColor;
                }
            }

            if (candidates != null && count > 0)
            {
                float dt = Time.time - _lastDetectionTime;
                Task.Run(() => ClusterCandidates(candidates, count, dt));
            }
            else
            {
                FadeUnmatchedClusters(Time.time - _lastDetectionTime);
                ApplyLights();
                _readbackPending = false;
            }
        }

        private void ClusterCandidates(CandidateRaw[] candidates, int count, float dt)
        {
            try
            {
                // Sort by luminance descending
                Array.Sort(candidates, 0, count,
                    Comparer<CandidateRaw>.Create((a, b) => b.luminance.CompareTo(a.luminance)));

                bool[] matched = new bool[_clusters.Count];
                float mergeR2 = mergeRadius * mergeRadius;

                for (int i = 0; i < count; i++)
                {
                    var c = candidates[i];
                    var pos = new Vector3(c.posX, c.posY, c.posZ);
                    var col = new Color(c.colR, c.colG, c.colB, 1f);

                    int bestIdx = -1;
                    float bestDist2 = mergeR2;

                    for (int j = 0; j < _clusters.Count; j++)
                    {
                        float d2 = (pos - _clusters[j].Position).sqrMagnitude;
                        if (d2 < bestDist2)
                        {
                            bestDist2 = d2;
                            bestIdx = j;
                        }
                    }

                    if (bestIdx >= 0)
                    {
                        matched[bestIdx] = true;
                        var cl = _clusters[bestIdx];
                        cl.CandidateCount++;
                        cl.LuminanceSum += c.luminance;
                        cl.ColorSum += col;

                        if (!cl.Frozen)
                        {
                            float w = c.luminance / (cl.LuminanceSum + 0.001f);
                            cl.Position = Vector3.Lerp(cl.Position, pos, w);
                            cl.Normal = Vector3.Lerp(cl.Normal,
                                new Vector3(c.nrmX, c.nrmY, c.nrmZ), w).normalized;
                        }
                    }
                    else if (_clusters.Count < maxLights)
                    {
                        _clusters.Add(new LightCluster
                        {
                            Position = pos,
                            Normal = new Vector3(c.nrmX, c.nrmY, c.nrmZ),
                            CandidateCount = 1,
                            LuminanceSum = c.luminance,
                            ColorSum = col,
                            Confidence = 0,
                            Frozen = false,
                            MissedCycles = 0
                        });
                    }
                }

                // Finalize clusters
                for (int j = 0; j < _clusters.Count; j++)
                {
                    var cl = _clusters[j];
                    if (matched.Length > j && matched[j])
                    {
                        cl.Confidence = Mathf.Min(cl.Confidence + 1, freezeConfidence + 10);
                        if (cl.Confidence >= freezeConfidence) cl.Frozen = true;
                        cl.Intensity = (cl.LuminanceSum / Mathf.Max(cl.CandidateCount, 1)) * intensityScale;
                        cl.Color = cl.ColorSum / Mathf.Max(cl.CandidateCount, 1);
                        cl.MissedCycles = 0;
                    }
                    else
                    {
                        cl.MissedCycles++;
                        float fadeRate = dt / Mathf.Max(fadeSeconds, 0.1f);
                        cl.Intensity = Mathf.Max(cl.Intensity - cl.Intensity * fadeRate, 0f);
                    }

                    // Reset per-cycle accumulators
                    cl.CandidateCount = 0;
                    cl.LuminanceSum = 0f;
                    cl.ColorSum = Color.black;
                }

                // Remove dead unfrozen lights (zero intensity, not frozen, many misses)
                _clusters.RemoveAll(cl => !cl.Frozen && cl.MissedCycles > 10 && cl.Intensity < 0.01f);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[LightEstimator] Clustering failed: {ex.Message}");
            }

            // Back to main thread
            UnityMainThreadDispatcher.Enqueue(() =>
            {
                ApplyLights();
                _readbackPending = false;
            });
        }

        private void FadeUnmatchedClusters(float dt)
        {
            float fadeRate = dt / Mathf.Max(fadeSeconds, 0.1f);
            for (int j = 0; j < _clusters.Count; j++)
            {
                var cl = _clusters[j];
                cl.MissedCycles++;
                cl.Intensity = Mathf.Max(cl.Intensity - cl.Intensity * fadeRate, 0f);
            }
            _clusters.RemoveAll(cl => !cl.Frozen && cl.MissedCycles > 10 && cl.Intensity < 0.01f);
        }

        private void ApplyLights()
        {
            // Sync _lights list with _clusters
            while (_lights.Count > _clusters.Count)
            {
                var removed = _lights[_lights.Count - 1];
                if (removed.UnityLight != null)
                    Destroy(removed.UnityLight.gameObject);
                _lights.RemoveAt(_lights.Count - 1);
                LightRemoved?.Invoke(removed);
            }

            for (int i = 0; i < _clusters.Count; i++)
            {
                var cl = _clusters[i];

                EstimatedLight el;
                if (i < _lights.Count)
                {
                    el = _lights[i];
                }
                else
                {
                    var go = new GameObject($"EstLight_{i}");
                    go.transform.SetParent(_lightContainer.transform);
                    var light = go.AddComponent<Light>();
                    light.type = LightType.Point;
                    light.shadows = LightShadows.Soft;
                    light.renderMode = LightRenderMode.Auto;

                    el = new EstimatedLight { UnityLight = light };
                    _lights.Add(el);
                    LightAdded?.Invoke(el);
                }

                el.Position = cl.Position;
                el.Normal = cl.Normal;
                el.Color = cl.Color;
                el.Intensity = cl.Intensity;
                el.Confidence = cl.Confidence;
                el.Frozen = cl.Frozen;

                if (el.UnityLight != null)
                {
                    el.UnityLight.transform.position = cl.Position;
                    el.UnityLight.color = cl.Color;
                    el.UnityLight.intensity = cl.Intensity;
                    el.UnityLight.range = lightRange;
                    el.UnityLight.enabled = cl.Intensity > 0.01f;

                    // Ceiling lights (normal pointing down) → spot aimed down
                    if (cl.Normal.y < -0.5f)
                    {
                        el.UnityLight.type = LightType.Spot;
                        el.UnityLight.spotAngle = 120f;
                        el.UnityLight.transform.rotation = Quaternion.LookRotation(Vector3.down);
                    }
                    else
                    {
                        el.UnityLight.type = LightType.Point;
                    }
                }
            }

            LightsUpdated?.Invoke();
        }

        private static int CeilDiv(int a, int b) => (a + b - 1) / b;

        private struct CandidateRaw
        {
            public float posX, posY, posZ;
            public float luminance;
            public float colR, colG, colB;
            public float _pad0;
            public float nrmX, nrmY, nrmZ;
            public float _pad1;
        }

        private class LightCluster
        {
            public Vector3 Position;
            public Vector3 Normal;
            public Color Color;
            public float Intensity;
            public float Confidence;
            public bool Frozen;
            public int MissedCycles;
            public int CandidateCount;
            public float LuminanceSum;
            public Color ColorSum;
        }
    }

    /// <summary>
    /// Simple main-thread dispatcher for callbacks from background threads.
    /// </summary>
    internal static class UnityMainThreadDispatcher
    {
        private static readonly Queue<Action> _queue = new();
        private static MonoBehaviour _runner;

        public static void Enqueue(Action action)
        {
            lock (_queue) _queue.Enqueue(action);
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Init()
        {
            var go = new GameObject("[MainThreadDispatcher]");
            go.hideFlags = HideFlags.HideAndDontSave;
            _runner = go.AddComponent<DispatcherRunner>();
            UnityEngine.Object.DontDestroyOnLoad(go);
        }

        private class DispatcherRunner : MonoBehaviour
        {
            private void Update()
            {
                lock (_queue)
                {
                    while (_queue.Count > 0)
                        _queue.Dequeue()?.Invoke();
                }
            }
        }
    }
}
