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
        /// <summary>True when the provider has a valid frame available this tick.</summary>
        bool IsReady { get; }

        /// <summary>The most recent camera RGB frame as a GPU texture.</summary>
        Texture CurrentFrame { get; }

        /// <summary>Camera-to-world matrix (extrinsics).</summary>
        Matrix4x4 CameraToWorld { get; }

        /// <summary>Camera projection matrix (intrinsics → projection).</summary>
        Matrix4x4 ProjectionMatrix { get; }

        /// <summary>Begins camera frame acquisition.</summary>
        void StartCapture();

        /// <summary>Stops camera frame acquisition and releases resources.</summary>
        void StopCapture();
    }
}
