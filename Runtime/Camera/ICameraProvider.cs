using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Interface for providing RGB camera frames to the texture projector.
    /// Implement this to plug in custom camera sources (Meta PassthroughCameraAccess,
    /// UXR QuestCamera, etc.). A default WebCamTexture implementation is provided.
    /// </summary>
    public interface ICameraProvider
    {
        bool IsReady { get; }
        Texture CurrentFrame { get; }

        /// <summary>Camera-to-world matrix (extrinsics).</summary>
        Matrix4x4 CameraToWorld { get; }

        /// <summary>Camera projection matrix (intrinsics → projection).</summary>
        Matrix4x4 ProjectionMatrix { get; }

        void StartCapture();
        void StopCapture();
    }
}
