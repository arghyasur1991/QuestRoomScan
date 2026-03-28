using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Interface for providing RGB camera frames and intrinsics to the scan pipeline.
    /// Implement this to plug in custom camera sources (Meta PassthroughCameraAccess,
    /// UXR QuestCamera, etc.).
    /// </summary>
    public interface ICameraProvider
    {
        /// <summary>True when the provider has a valid frame available this tick.</summary>
        bool IsReady { get; }

        /// <summary>True when the camera subsystem is actively running (may not have a new frame every tick).</summary>
        bool IsPlaying { get; }

        /// <summary>The most recent camera RGB frame as a GPU texture.</summary>
        Texture CurrentFrame { get; }

        /// <summary>World-space pose of the camera this frame.</summary>
        Pose CameraPose { get; }

        /// <summary>Camera intrinsic focal length in pixels (fx, fy).</summary>
        Vector2 FocalLength { get; }

        /// <summary>Camera intrinsic principal point in pixels (cx, cy).</summary>
        Vector2 PrincipalPoint { get; }

        /// <summary>Native sensor resolution in pixels.</summary>
        Vector2 SensorResolution { get; }

        /// <summary>Actual delivered frame resolution (may differ from sensor resolution).</summary>
        Vector2 CurrentResolution { get; }

        /// <summary>Begins camera frame acquisition.</summary>
        void StartCapture();

        /// <summary>Stops camera frame acquisition and releases resources.</summary>
        void StopCapture();
    }
}
