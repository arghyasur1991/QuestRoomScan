#if HAS_AI_INFERENCE
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// GPU Surface Nets mesh extraction from a density field.
    /// Dispatches DensitySurfaceNets.compute for all kernels — no CPU fallback.
    /// Accepts a GPU ComputeBuffer for density and uses async readback for results.
    /// </summary>
    internal sealed class DensitySurfaceNets : IMeshExtractor
    {
        private readonly ComputeShader _shader;
        private readonly int _resolution;
        private readonly float _threshold;

        private readonly int _kernelClear;
        private readonly int _kernelClassify;
        private readonly int _kernelBuildVertArgs;
        private readonly int _kernelGenIndices;
        private readonly int _kernelBuildDrawArgs;

        internal DensitySurfaceNets(ComputeShader shader, int resolution, float threshold)
        {
            _shader = shader;
            _resolution = resolution;
            _threshold = threshold;

            _kernelClear = _shader.FindKernel("ClearCounters");
            _kernelClassify = _shader.FindKernel("ClassifyAndEmit");
            _kernelBuildVertArgs = _shader.FindKernel("BuildVertexDispatchArgs");
            _kernelGenIndices = _shader.FindKernel("GenerateIndices");
            _kernelBuildDrawArgs = _shader.FindKernel("BuildIndirectArgs");
        }

        public async Task<Mesh> ExtractAsync(ComputeBuffer densityBuf)
        {
            int res = _resolution;
            int totalVoxels = res * res * res;
            int maxVerts = Mathf.Min(res * res * 10, 2_000_000);
            int maxIndices = maxVerts * 6;

            var coordVertMap = new ComputeBuffer(totalVoxels, sizeof(int));
            var vertexBuf = new ComputeBuffer(maxVerts, GpuVertexSize);
            var indexBuf = new ComputeBuffer(maxIndices, sizeof(uint));
            var counterBuf = new ComputeBuffer(2, sizeof(uint));
            var dispatchArgsBuf = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);
            var drawArgsBuf = new ComputeBuffer(5, sizeof(uint), ComputeBufferType.IndirectArguments);

            try
            {
                _shader.SetInts("_VoxCount", res, res, res);
                _shader.SetFloat("_VoxSize", 1f / (res - 1));
                _shader.SetInt("_MaxVertices", maxVerts);
                _shader.SetFloat("_DensityThreshold", _threshold);
                _shader.SetInt("_TotalVoxels", totalVoxels);

                BindAll(_kernelClear, densityBuf, coordVertMap, vertexBuf, indexBuf,
                    counterBuf, dispatchArgsBuf, drawArgsBuf);
                BindAll(_kernelClassify, densityBuf, coordVertMap, vertexBuf, indexBuf,
                    counterBuf, dispatchArgsBuf, drawArgsBuf);
                BindAll(_kernelBuildVertArgs, densityBuf, coordVertMap, vertexBuf, indexBuf,
                    counterBuf, dispatchArgsBuf, drawArgsBuf);
                BindAll(_kernelGenIndices, densityBuf, coordVertMap, vertexBuf, indexBuf,
                    counterBuf, dispatchArgsBuf, drawArgsBuf);
                BindAll(_kernelBuildDrawArgs, densityBuf, coordVertMap, vertexBuf, indexBuf,
                    counterBuf, dispatchArgsBuf, drawArgsBuf);

                _shader.Dispatch(_kernelClear, 1, 1, 1);

                int groups = Mathf.CeilToInt(res / 4f);
                _shader.Dispatch(_kernelClassify, groups, groups, groups);

                _shader.Dispatch(_kernelBuildVertArgs, 1, 1, 1);
                _shader.DispatchIndirect(_kernelGenIndices, dispatchArgsBuf);
                _shader.Dispatch(_kernelBuildDrawArgs, 1, 1, 1);

                var counters = await AsyncHelper.ReadbackAsync<uint>(counterBuf, 2);
                int vertCount = (int)Mathf.Min(counters[0], maxVerts);
                int idxCount = (int)Mathf.Min(counters[1], maxIndices);

                if (vertCount == 0)
                    return new Mesh();

                var gpuVertsTask = AsyncHelper.ReadbackAsync<GPUVertex>(vertexBuf, vertCount);
                var gpuIndicesTask = AsyncHelper.ReadbackAsync<int>(indexBuf, idxCount);
                await Task.WhenAll(gpuVertsTask, gpuIndicesTask);

                var gpuVerts = gpuVertsTask.Result;
                var gpuIndices = gpuIndicesTask.Result;

                var positions = new Vector3[vertCount];
                var normals = new Vector3[vertCount];
                for (int i = 0; i < vertCount; i++)
                {
                    positions[i] = gpuVerts[i].pos;
                    normals[i] = gpuVerts[i].norm;
                }

                var mesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
                mesh.SetVertices(positions);
                mesh.SetNormals(normals);
                mesh.SetTriangles(gpuIndices, 0);
                mesh.RecalculateBounds();

                return mesh;
            }
            finally
            {
                coordVertMap.Release();
                vertexBuf.Release();
                indexBuf.Release();
                counterBuf.Release();
                dispatchArgsBuf.Release();
                drawArgsBuf.Release();
            }
        }

        private void BindAll(int kernel, ComputeBuffer density, ComputeBuffer coordMap,
            ComputeBuffer verts, ComputeBuffer indices, ComputeBuffer counters,
            ComputeBuffer dispatchArgs, ComputeBuffer drawArgs)
        {
            _shader.SetBuffer(kernel, "_DensityVolume", density);
            _shader.SetBuffer(kernel, "_CoordVertMap", coordMap);
            _shader.SetBuffer(kernel, "_Vertices", verts);
            _shader.SetBuffer(kernel, "_Indices", indices);
            _shader.SetBuffer(kernel, "_Counters", counters);
            _shader.SetBuffer(kernel, "_DispatchArgs", dispatchArgs);
            _shader.SetBuffer(kernel, "_DrawIndirectArgs", drawArgs);
        }

        public void Dispose() { }

        private const int GpuVertexSize = 32;

        [StructLayout(LayoutKind.Sequential)]
        private struct GPUVertex
        {
            public Vector3 pos;
            public Vector3 norm;
            public uint packedColor;
            public uint voxelFlatIdx;
        }
    }
}
#endif
