using System;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Scene API labels the host wants when copying vertical planes.
    /// <see cref="Wall"/> is visible <c>WALL_FACE</c>; <see cref="Screen"/>
    /// is <c>SCREEN</c>. This package copies whatever the host asks for.
    /// </summary>
    [Flags]
    public enum SceneFaceKind
    {
        None = 0,
        Wall = 1 << 0,
        Screen = 1 << 1,
        All = Wall | Screen,
    }

    /// <summary>
    /// One vertical Scene API plane. Hosts query these without taking an
    /// MRUK dependency. The list is already filtered by the
    /// <see cref="SceneFaceKind"/> the host passed.
    /// </summary>
    public readonly struct SceneWallFace
    {
        public readonly Vector3 Center;
        /// <summary>Unit vector from the plane into the room.</summary>
        public readonly Vector3 Inward;
        public readonly float Width;
        public readonly float Height;
        /// <summary>Floor height under this room (metres, world Y).</summary>
        public readonly float FloorY;
        /// <summary>True when the plane also has <c>WALL_ART</c>.</summary>
        public readonly bool PreferAvoid;

        public SceneWallFace(
            Vector3 center,
            Vector3 inward,
            float width,
            float height,
            float floorY,
            bool preferAvoid)
        {
            Center = center;
            Inward = inward;
            Width = width;
            Height = height;
            FloorY = floorY;
            PreferAvoid = preferAvoid;
        }
    }

    /// <summary>
    /// GPU stamp for a <c>SCREEN</c> (TV): a fattened slab whose TSDF is the
    /// analytic plane rather than headset depth. Glass and bounce make the
    /// depth sensor lie; locking the slab to the MRUK plane yields one
    /// planar sheet. Color still comes from the RGB camera.
    /// </summary>
    public readonly struct ScanScreenStamp
    {
        public readonly Vector3 Center;
        /// <summary>Unit vector from the glass into the room.</summary>
        public readonly Vector3 Inward;
        public readonly Vector3 Tangent;
        public readonly Vector3 Bitangent;
        public readonly float HalfWidth;
        public readonly float HalfHeight;
        public readonly float HalfThickness;

        public ScanScreenStamp(
            Vector3 center,
            Vector3 inward,
            Vector3 tangent,
            Vector3 bitangent,
            float halfWidth,
            float halfHeight,
            float halfThickness)
        {
            Center = center;
            Inward = inward;
            Tangent = tangent;
            Bitangent = bitangent;
            HalfWidth = halfWidth;
            HalfHeight = halfHeight;
            HalfThickness = halfThickness;
        }

        /// <summary>
        /// Build a stamp from a vertical plane. Thickness covers a typical
        /// TV volume (~11 cm) plus a little slack; width/height expand for
        /// the bezel so Surface Nets do not leave a depth-noise rim.
        /// </summary>
        public static ScanScreenStamp FromFace(
            SceneWallFace face,
            float halfThickness = 0.13f,
            float expandMetres = 0.03f)
        {
            Vector3 inward = face.Inward.sqrMagnitude > 1e-8f
                ? face.Inward.normalized
                : Vector3.forward;
            Vector3 tangent = Vector3.Cross(Vector3.up, inward);
            if (tangent.sqrMagnitude < 1e-6f)
                tangent = Vector3.Cross(Vector3.right, inward);
            tangent.Normalize();
            Vector3 bitangent = Vector3.Cross(inward, tangent).normalized;
            return new ScanScreenStamp(
                face.Center,
                inward,
                tangent,
                bitangent,
                face.Width * 0.5f + expandMetres,
                face.Height * 0.5f + expandMetres,
                halfThickness);
        }
    }
}
