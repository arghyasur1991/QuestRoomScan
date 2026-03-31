using System;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>Source that originally detected this object.</summary>
    public enum SceneObjectSource : byte
    {
        Unknown = 0,
        MRUK = 1,
        AIDetection = 2,
        Manual = 3
    }

    /// <summary>
    /// A detected object in the scanned room — from MRUK anchors, AI inference,
    /// or manual placement. Serializable for persistence in scan packages.
    /// </summary>
    [Serializable]
    public class SceneObject
    {
        public string id;
        public string label;
        public SceneObjectSource source;
        public SurfaceType surfaceType;
        public float confidence;

        [Header("World Transform")]
        public Vector3 position;
        public Quaternion rotation = Quaternion.identity;
        public Vector3 size = Vector3.one;

        [Header("MRUK-specific")]
        public string mrukLabel;
        public string anchorUuid;

        [Header("AI Detection-specific")]
        public int classId = -1;
        public Rect imageBoundingBox;

        /// <summary>World-space bounding box from position + size.</summary>
        public Bounds WorldBounds => new Bounds(position, size);

        /// <summary>World-space pose from position + rotation.</summary>
        public Pose WorldPose => new Pose(position, rotation);
    }
}
