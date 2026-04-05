#if HAS_ONNXRUNTIME
using System;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    internal interface IMeshExtractor : IDisposable
    {
        /// <param name="densityGPU">Pre-filled density ComputeBuffer on GPU (resolution^3 floats).</param>
        Task<Mesh> ExtractAsync(ComputeBuffer densityGPU);
    }

    internal enum MeshAlgorithm
    {
        MarchingCubes,
        SurfaceNets
    }
}
#endif
