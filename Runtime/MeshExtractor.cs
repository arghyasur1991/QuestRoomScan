using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Orchestrates GPU Surface Nets mesh extraction from the TSDF volume.
    /// Dispatches GPUSurfaceNets compute shaders and manages the GPUMeshRenderer.
    /// </summary>
    public class MeshExtractor : MonoBehaviour
    {
        public static MeshExtractor Instance { get; private set; }

        [Header("Mesh Smoothing")]
        [SerializeField, Tooltip("Post-extraction vertex smoothing iterations. 0 = disabled.")]
        [Range(0, 8)] private int meshSmoothIterations = 1;
        [SerializeField, Tooltip("Laplacian blend strength per iteration.")]
        [Range(0.1f, 1f)] private float meshSmoothLambda = 0.33f;
        [SerializeField, Tooltip("HC back-projection strength to prevent volume shrinkage.")]
        [Range(0f, 1f)] private float meshSmoothBeta = 0.5f;

        [Header("Temporal Stability")]
        [SerializeField, Tooltip("Alpha for large displacements (fast convergence).")]
        [Range(0.1f, 1f)] private float temporalAlphaMax = 0.85f;
        [SerializeField, Tooltip("Alpha for long-stable vertices (strong resistance to change).")]
        [Range(0.01f, 0.5f)] private float temporalAlphaMin = 0.1f;
        [SerializeField, Tooltip("How quickly alpha decays from max to min as vertex stabilizes.")]
        [Range(0.01f, 1f)] private float temporalDecayRate = 0.15f;
        [SerializeField, Tooltip("Displacement threshold (meters) to consider a vertex still converging.")]
        [Range(0.001f, 0.02f)] private float convergenceThreshold = 0.005f;
        [SerializeField, Tooltip("Position changes below this (meters) are suppressed entirely.")]
        [Range(0f, 0.01f)] private float temporalDeadzone = 0.001f;

        [Header("Rendering")]
        [SerializeField] private Material scanMeshMaterial;

        [Header("Compute")]
        [SerializeField] public ComputeShader surfaceNetsCompute;
        [SerializeField, Tooltip("Max vertex fraction of total voxels (0.01-0.10).")]
        [Range(0.01f, 0.10f)] private float gpuVertexBudgetPercent = 0.05f;

        private GPUSurfaceNets _gpuSurfaceNets;
        private GPUMeshRenderer _gpuRenderer;
        private int _extractCount;

        public GPUSurfaceNets GpuSurfaceNets => _gpuSurfaceNets;

        private VolumeIntegrator _volume;

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            _volume = VolumeIntegrator.Instance;
            if (_volume == null)
                throw new Exception("[RoomScan] VolumeIntegrator not found");

            if (surfaceNetsCompute == null)
                throw new Exception("[RoomScan] surfaceNetsCompute not assigned on MeshExtractor");

            var rpAsset = GraphicsSettings.currentRenderPipeline;
            Debug.Log($"[RoomScan] MeshExtractor Start: mat={scanMeshMaterial?.name ?? "NULL"}, " +
                $"shader={scanMeshMaterial?.shader?.name ?? "NULL"}, " +
                $"rp={rpAsset?.name ?? "NULL"}, " +
                $"stereoMode={UnityEngine.XR.XRSettings.stereoRenderingMode}");

            Init();
        }

        private void OnDestroy()
        {
            _gpuSurfaceNets?.Dispose();
            _gpuSurfaceNets = null;
        }

        private void Init()
        {
            _gpuSurfaceNets = new GPUSurfaceNets(surfaceNetsCompute)
            {
                MinMeshWeight = _volume.MinMeshWeight,
                SmoothIterations = meshSmoothIterations,
                SmoothLambda = meshSmoothLambda,
                SmoothBeta = meshSmoothBeta,
                TemporalAlphaMax = temporalAlphaMax,
                TemporalAlphaMin = temporalAlphaMin,
                TemporalDecayRate = temporalDecayRate,
                ConvergenceThreshold = convergenceThreshold,
                TemporalDeadzone = temporalDeadzone
            };
            _gpuSurfaceNets.EnsureBuffers(_volume.VoxelCount, gpuVertexBudgetPercent);

            _gpuRenderer = gameObject.AddComponent<GPUMeshRenderer>();
            _gpuRenderer.GpuMeshMaterial = scanMeshMaterial;
            _gpuRenderer.Initialize(_gpuSurfaceNets, _gpuSurfaceNets.GetVolumeBounds(_volume.VoxelSize));

            Debug.Log($"[RoomScan] GPU Surface Nets initialized: voxels={_volume.VoxelCount}, " +
                      $"voxSize={_volume.VoxelSize}");
        }

        /// <summary>
        /// Run one GPU mesh extraction pass from the current TSDF volume state.
        /// Called by RoomScanner at the configured mesh extraction rate.
        /// </summary>
        public void Extract()
        {
            if (_gpuSurfaceNets == null) return;

            _extractCount++;
            _gpuSurfaceNets.MinMeshWeight = _volume.MinMeshWeight;

            _gpuSurfaceNets.Extract(_volume.Volume, _volume.ColorVolume, _volume.VoxelSize);

            if (_extractCount <= 3 || _extractCount % 50 == 0)
                Debug.Log($"[RoomScan] GPU extraction #{_extractCount}");
        }

        /// <summary>
        /// Dispose GPU resources and reinitialize. Used after loading a saved scan.
        /// </summary>
        public void Reinitialize()
        {
            _gpuSurfaceNets?.Dispose();
            _gpuSurfaceNets = null;
            if (_gpuRenderer != null) Destroy(_gpuRenderer);
            Init();
        }
    }
}
