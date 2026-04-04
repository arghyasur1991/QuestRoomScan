using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Abstraction for single-image 3D reconstruction, implemented in the optional
    /// <c>Genesis.RoomScan.ObjectReconstruction</c> assembly so the core package has no hard
    /// dependency on inference engine packages.
    /// </summary>
    public interface IObjectReconstructionProvider
    {
        string Status { get; }
        bool IsRunning { get; }
        Texture2D[] TestImages { get; }

        Task LoadModelsAsync(CancellationToken ct = default);
        Task ReconstructAsync(Texture2D image, CancellationToken ct = default);
        void ClearMesh();
    }
}
