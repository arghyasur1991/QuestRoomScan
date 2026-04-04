#if HAS_AI_INFERENCE
using System;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    internal interface IMeshExtractor : IDisposable
    {
        Task<Mesh> ExtractAsync(float[] density);
    }

    internal enum MeshAlgorithm
    {
        MarchingCubes,
        SurfaceNets
    }
}
#endif
