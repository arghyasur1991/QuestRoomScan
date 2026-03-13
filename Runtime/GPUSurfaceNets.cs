using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class GPUSurfaceNets : IDisposable
    {
        private readonly ComputeShader _compute;

        // Kernel indices
        private readonly int _kClearCounters;
        private readonly int _kClassifyAndEmit;
        private readonly int _kBuildVertexDispatchArgs;
        private readonly int _kInitSmooth;
        private readonly int _kSmoothVertices;
        private readonly int _kApplySmooth;
        private readonly int _kPlaneSnap;
        private readonly int _kTemporalBlend;
        private readonly int _kGenerateIndices;
        private readonly int _kBuildIndirectArgs;

        // GPU buffers
        private GraphicsBuffer _coordVertMap;
        private GraphicsBuffer _vertices;
        private GraphicsBuffer _indices;
        private GraphicsBuffer _counters;
        private GraphicsBuffer _dispatchArgs;
        private GraphicsBuffer _drawIndirectArgs;
        private GraphicsBuffer _smoothPosA;
        private GraphicsBuffer _smoothPosB;
        private GraphicsBuffer _temporalState;

        // Sizing
        private int3 _voxCount;
        private int _totalVoxels;
        private int _maxVertices;
        private int _maxIndices;
        private bool _temporalInitialized;

        // Configurable parameters (set before Extract)
        public float MinMeshWeight { get; set; } = 0.08f;
        public int SmoothIterations { get; set; } = 1;
        public float SmoothLambda { get; set; } = 0.33f;
        public float SmoothBeta { get; set; } = 0.5f;
        public float PlaneSnapThreshold { get; set; } = 0.03f;
        public float TemporalAlphaMax { get; set; } = 0.85f;
        public float TemporalAlphaMin { get; set; } = 0.1f;
        public float TemporalDecayRate { get; set; } = 0.15f;
        public float ConvergenceThreshold { get; set; } = 0.005f;
        public float TemporalDeadzone { get; set; } = 0.001f;

        // Public accessors for rendering
        public GraphicsBuffer VertexBuffer => _vertices;
        public GraphicsBuffer IndexBuffer => _indices;
        public GraphicsBuffer DrawIndirectArgs => _drawIndirectArgs;

        // Shader property IDs
        private static readonly int ID_TsdfVolume = Shader.PropertyToID("_TsdfVolume");
        private static readonly int ID_ColorVolume = Shader.PropertyToID("_ColorVolume");
        private static readonly int ID_VoxCount = Shader.PropertyToID("_VoxCount");
        private static readonly int ID_VoxSize = Shader.PropertyToID("_VoxSize");
        private static readonly int ID_MinWeight = Shader.PropertyToID("_MinWeight");
        private static readonly int ID_TotalVoxels = Shader.PropertyToID("_TotalVoxels");
        private static readonly int ID_MaxVertices = Shader.PropertyToID("_MaxVertices");
        private static readonly int ID_SmoothLambda = Shader.PropertyToID("_SmoothLambda");
        private static readonly int ID_SmoothBeta = Shader.PropertyToID("_SmoothBeta");
        private static readonly int ID_NumPlanes = Shader.PropertyToID("_NumPlanes");
        private static readonly int ID_SnapThreshold = Shader.PropertyToID("_SnapThreshold");
        private static readonly int ID_PlaneNormals = Shader.PropertyToID("_PlaneNormals");
        private static readonly int ID_PlaneParams = Shader.PropertyToID("_PlaneParams");
        private static readonly int ID_TemporalAlphaMax = Shader.PropertyToID("_TemporalAlphaMax");
        private static readonly int ID_TemporalAlphaMin = Shader.PropertyToID("_TemporalAlphaMin");
        private static readonly int ID_TemporalDecayRate = Shader.PropertyToID("_TemporalDecayRate");
        private static readonly int ID_ConvergeThreshold = Shader.PropertyToID("_ConvergeThreshold");
        private static readonly int ID_TemporalDeadzone = Shader.PropertyToID("_TemporalDeadzone");

        private static readonly int ID_CoordVertMap = Shader.PropertyToID("_CoordVertMap");
        private static readonly int ID_Vertices = Shader.PropertyToID("_Vertices");
        private static readonly int ID_Indices = Shader.PropertyToID("_Indices");
        private static readonly int ID_Counters = Shader.PropertyToID("_Counters");
        private static readonly int ID_DispatchArgs = Shader.PropertyToID("_DispatchArgs");
        private static readonly int ID_DrawIndirectArgs = Shader.PropertyToID("_DrawIndirectArgs");
        private static readonly int ID_SmoothPosA = Shader.PropertyToID("_SmoothPosA");
        private static readonly int ID_SmoothPosB = Shader.PropertyToID("_SmoothPosB");
        private static readonly int ID_TemporalState = Shader.PropertyToID("_TemporalState");

        private const int VertexStride = 32;   // sizeof(GPUVertex): float3 + float3 + uint + uint
        private const int Float3Stride = 12;
        private const int Float4Stride = 16;

        public GPUSurfaceNets(ComputeShader compute)
        {
            _compute = compute;

            _kClearCounters = compute.FindKernel("ClearCounters");
            _kClassifyAndEmit = compute.FindKernel("ClassifyAndEmit");
            _kBuildVertexDispatchArgs = compute.FindKernel("BuildVertexDispatchArgs");
            _kInitSmooth = compute.FindKernel("InitSmooth");
            _kSmoothVertices = compute.FindKernel("SmoothVertices");
            _kApplySmooth = compute.FindKernel("ApplySmooth");
            _kPlaneSnap = compute.FindKernel("PlaneSnap");
            _kTemporalBlend = compute.FindKernel("TemporalBlend");
            _kGenerateIndices = compute.FindKernel("GenerateIndices");
            _kBuildIndirectArgs = compute.FindKernel("BuildIndirectArgs");
        }

        public void EnsureBuffers(int3 voxCount, float vertexBudgetPercent = 0.05f)
        {
            int totalVoxels = voxCount.x * voxCount.y * voxCount.z;
            if (_totalVoxels == totalVoxels && _coordVertMap != null)
                return;

            Dispose();

            _voxCount = voxCount;
            _totalVoxels = totalVoxels;
            _maxVertices = Mathf.Max(1024, (int)(totalVoxels * vertexBudgetPercent));
            _maxIndices = _maxVertices * 18;

            _coordVertMap = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVoxels, 4);
            _vertices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, VertexStride);
            _indices = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxIndices, 4);
            _counters = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 2, 4);
            _dispatchArgs = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 3, 4);
            _drawIndirectArgs = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 5, 4);
            _smoothPosA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
            _smoothPosB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _maxVertices, Float3Stride);
            _temporalState = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVoxels, Float4Stride);

            _temporalInitialized = false;

            long totalBytes = (long)totalVoxels * 4          // coordVertMap
                            + (long)_maxVertices * VertexStride  // vertices
                            + (long)_maxIndices * 4           // indices
                            + 2 * 4                            // counters
                            + 3 * 4 + 5 * 4                   // dispatch + draw args
                            + (long)_maxVertices * Float3Stride * 2 // smooth A+B
                            + (long)totalVoxels * Float4Stride;    // temporal state
            Debug.Log($"[GPUSurfaceNets] Allocated buffers: vox={voxCount}, " +
                      $"maxVerts={_maxVertices}, maxIdx={_maxIndices}, " +
                      $"totalGPU={totalBytes / (1024 * 1024)}MB");
        }

        public void InitTemporalState()
        {
            if (_temporalInitialized || _temporalState == null) return;

            var init = new float4[_totalVoxels];
            for (int i = 0; i < _totalVoxels; i++)
                init[i] = new float4(0, 0, 0, -1);
            _temporalState.SetData(init);
            _temporalInitialized = true;
        }

        public void Extract(RenderTexture tsdfVolume, RenderTexture colorVolume,
                            float voxelSize, PlaneData[] planes = null, int numPlanes = 0)
        {
            if (_coordVertMap == null)
                throw new InvalidOperationException("Call EnsureBuffers before Extract");

            if (!_temporalInitialized && TemporalAlphaMax < 1f)
                InitTemporalState();

            SetGlobalParams(voxelSize);
            BindAllBuffers();

            // Textures
            _compute.SetTexture(_kClassifyAndEmit, ID_TsdfVolume, tsdfVolume);
            _compute.SetTexture(_kClassifyAndEmit, ID_ColorVolume, colorVolume);
            _compute.SetTexture(_kGenerateIndices, ID_TsdfVolume, tsdfVolume);

            // 1. Clear counters
            _compute.Dispatch(_kClearCounters, 1, 1, 1);

            // 2. Classify & emit vertices
            int gx = CeilDiv(_voxCount.x, 4);
            int gy = CeilDiv(_voxCount.y, 4);
            int gz = CeilDiv(_voxCount.z, 4);
            _compute.Dispatch(_kClassifyAndEmit, gx, gy, gz);

            // 3. Build dispatch args from vertex count
            _compute.Dispatch(_kBuildVertexDispatchArgs, 1, 1, 1);

            // 4. Smoothing (optional)
            if (SmoothIterations > 0)
            {
                _compute.DispatchIndirect(_kInitSmooth, _dispatchArgs);

                for (int iter = 0; iter < SmoothIterations; iter++)
                {
                    if (iter % 2 == 0)
                    {
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosA);
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosB);
                    }
                    else
                    {
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosB);
                        _compute.SetBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosA);
                    }
                    _compute.DispatchIndirect(_kSmoothVertices, _dispatchArgs);
                }

                // Bind the buffer that has the final result as _SmoothPosA for ApplySmooth
                if (SmoothIterations % 2 == 0)
                    _compute.SetBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosA);
                else
                    _compute.SetBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosB);

                _compute.DispatchIndirect(_kApplySmooth, _dispatchArgs);
            }

            // 5. Plane snap (optional)
            if (planes != null && numPlanes > 0 && PlaneSnapThreshold > 0)
            {
                SetPlaneData(planes, numPlanes);
                _compute.DispatchIndirect(_kPlaneSnap, _dispatchArgs);
            }

            // 6. Temporal blend (optional)
            if (TemporalAlphaMax < 1f)
            {
                _compute.DispatchIndirect(_kTemporalBlend, _dispatchArgs);
            }

            // 7. Generate indices
            _compute.DispatchIndirect(_kGenerateIndices, _dispatchArgs);

            // 8. Build draw indirect args
            _compute.Dispatch(_kBuildIndirectArgs, 1, 1, 1);
        }

        public Bounds GetVolumeBounds(float voxelSize)
        {
            float3 halfExtent = (float3)_voxCount * voxelSize * 0.5f;
            return new Bounds(Vector3.zero, (Vector3)(halfExtent * 2));
        }

        private void SetGlobalParams(float voxelSize)
        {
            _compute.SetInts(ID_VoxCount, _voxCount.x, _voxCount.y, _voxCount.z);
            _compute.SetFloat(ID_VoxSize, voxelSize);
            _compute.SetFloat(ID_MinWeight, MinMeshWeight);
            _compute.SetInt(ID_TotalVoxels, _totalVoxels);
            _compute.SetInt(ID_MaxVertices, _maxVertices);
            _compute.SetFloat(ID_SmoothLambda, SmoothLambda);
            _compute.SetFloat(ID_SmoothBeta, SmoothBeta);
            _compute.SetFloat(ID_SnapThreshold, PlaneSnapThreshold);
            _compute.SetFloat(ID_TemporalAlphaMax, TemporalAlphaMax);
            _compute.SetFloat(ID_TemporalAlphaMin, TemporalAlphaMin);
            _compute.SetFloat(ID_TemporalDecayRate, TemporalDecayRate);
            _compute.SetFloat(ID_ConvergeThreshold, ConvergenceThreshold);
            _compute.SetFloat(ID_TemporalDeadzone, TemporalDeadzone);
        }

        private void BindAllBuffers()
        {
            BindBuffer(_kClearCounters, ID_Counters, _counters);

            BindBuffer(_kClassifyAndEmit, ID_CoordVertMap, _coordVertMap);
            BindBuffer(_kClassifyAndEmit, ID_Vertices, _vertices);
            BindBuffer(_kClassifyAndEmit, ID_Counters, _counters);

            BindBuffer(_kBuildVertexDispatchArgs, ID_Counters, _counters);
            BindBuffer(_kBuildVertexDispatchArgs, ID_DispatchArgs, _dispatchArgs);

            BindBuffer(_kInitSmooth, ID_Vertices, _vertices);
            BindBuffer(_kInitSmooth, ID_SmoothPosA, _smoothPosA);
            BindBuffer(_kInitSmooth, ID_Counters, _counters);

            BindBuffer(_kSmoothVertices, ID_Vertices, _vertices);
            BindBuffer(_kSmoothVertices, ID_CoordVertMap, _coordVertMap);
            BindBuffer(_kSmoothVertices, ID_SmoothPosA, _smoothPosA);
            BindBuffer(_kSmoothVertices, ID_SmoothPosB, _smoothPosB);
            BindBuffer(_kSmoothVertices, ID_Counters, _counters);

            BindBuffer(_kApplySmooth, ID_Vertices, _vertices);
            BindBuffer(_kApplySmooth, ID_SmoothPosA, _smoothPosA);
            BindBuffer(_kApplySmooth, ID_Counters, _counters);

            BindBuffer(_kPlaneSnap, ID_Vertices, _vertices);
            BindBuffer(_kPlaneSnap, ID_Counters, _counters);

            BindBuffer(_kTemporalBlend, ID_Vertices, _vertices);
            BindBuffer(_kTemporalBlend, ID_TemporalState, _temporalState);
            BindBuffer(_kTemporalBlend, ID_Counters, _counters);

            BindBuffer(_kGenerateIndices, ID_Vertices, _vertices);
            BindBuffer(_kGenerateIndices, ID_CoordVertMap, _coordVertMap);
            BindBuffer(_kGenerateIndices, ID_Indices, _indices);
            BindBuffer(_kGenerateIndices, ID_Counters, _counters);

            BindBuffer(_kBuildIndirectArgs, ID_Counters, _counters);
            BindBuffer(_kBuildIndirectArgs, ID_DrawIndirectArgs, _drawIndirectArgs);
        }

        private void BindBuffer(int kernel, int nameID, GraphicsBuffer buffer)
        {
            _compute.SetBuffer(kernel, nameID, buffer);
        }

        private void SetPlaneData(PlaneData[] planes, int count)
        {
            count = Mathf.Min(count, 8);
            _compute.SetInt(ID_NumPlanes, count);

            var normals = new Vector4[8];
            var pars = new Vector4[8];
            for (int i = 0; i < count; i++)
            {
                normals[i] = new Vector4(planes[i].Normal.x, planes[i].Normal.y, planes[i].Normal.z, 0);
                pars[i] = new Vector4(planes[i].Distance, planes[i].Confidence, 0, 0);
            }
            _compute.SetVectorArray(ID_PlaneNormals, normals);
            _compute.SetVectorArray(ID_PlaneParams, pars);
        }

        public void Dispose()
        {
            _coordVertMap?.Release();
            _vertices?.Release();
            _indices?.Release();
            _counters?.Release();
            _dispatchArgs?.Release();
            _drawIndirectArgs?.Release();
            _smoothPosA?.Release();
            _smoothPosB?.Release();
            _temporalState?.Release();

            _coordVertMap = null;
            _vertices = null;
            _indices = null;
            _counters = null;
            _dispatchArgs = null;
            _drawIndirectArgs = null;
            _smoothPosA = null;
            _smoothPosB = null;
            _temporalState = null;

            _totalVoxels = 0;
            _temporalInitialized = false;
        }

        private static int CeilDiv(int a, int b) => (a + b - 1) / b;
    }
}
