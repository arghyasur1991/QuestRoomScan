using System;
using System.Threading.Tasks;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Abstraction for Gaussian Splat operations, implemented in the optional
    /// <c>Genesis.RoomScan.GSplat</c> assembly so the core package has no hard
    /// dependency on the GaussianSplatting package.
    /// </summary>
    public interface IGSplatProvider
    {
        bool HasServerTrainedSplats { get; }
        bool RenderVisible { get; set; }
        void ClearSplat();
        void ResetSplatTransform();
        void LoadTrainedPly(byte[] plyData);

        /// <summary>
        /// Runs the full server training pipeline: export, upload, train, download.
        /// </summary>
        Task<byte[]> RunServerTrainingAsync(string keyframeDir, Matrix4x4 keyframeRelocation);

        /// <summary>Server-side atlas enhancement.</summary>
        Task<byte[]> EnhanceAtlasAsync(byte[] pngBytes, int scale, bool inpaint);

        /// <summary>Server-side mesh enhancement.</summary>
        Task<byte[]> EnhanceMeshAsync(byte[] meshBin, int smoothIterations, bool enablePlaneSnap);

        /// <summary>Uploads training data to the server.</summary>
        Task<bool> UploadTrainingDataAsync(Matrix4x4 keyframeRelocation);
    }
}
