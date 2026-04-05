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

        /// <summary>
        /// Runs reconstruction and returns the resulting mesh.
        /// The caller is responsible for spawning/positioning the mesh in the scene.
        /// </summary>
        Task<Mesh> ReconstructAsync(Texture2D image, CancellationToken ct = default);

        /// <summary>
        /// Creates a material suitable for rendering the reconstructed mesh (vertex colors).
        /// </summary>
        Material CreateMaterial();

        /// <summary>
        /// Runs rembg only on a test image and returns the mask as a Texture2D for preview.
        /// Useful for quick EP validation (e.g. QNN HTP) without running the full pipeline.
        /// </summary>
        Task<Texture2D> TestRembgAsync(Texture2D image, CancellationToken ct = default);
    }
}
