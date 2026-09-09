using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// One captured pin surface in a Space Setup room: a visible
    /// <c>WALL_FACE</c> or a <c>SCREEN</c> (TV). Hosts pin world-space UI
    /// without taking an MRUK dependency.
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
        /// <summary>True for <c>WALL_ART</c> on a wall — prefer another wall.</summary>
        public readonly bool PreferAvoid;
        /// <summary>True for a <c>SCREEN</c> (TV). Hosts should pin on this
        /// plane and scale to its bounds rather than picking a blank wall.</summary>
        public readonly bool IsScreen;

        public SceneWallFace(
            Vector3 center,
            Vector3 inward,
            float width,
            float height,
            float floorY,
            bool preferAvoid,
            bool isScreen = false)
        {
            Center = center;
            Inward = inward;
            Width = width;
            Height = height;
            FloorY = floorY;
            PreferAvoid = preferAvoid;
            IsScreen = isScreen;
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
        /// Build a stamp from a pin face. Thickness covers a typical TV
        /// volume (~11 cm) plus a little slack; width/height expand for the
        /// bezel so Surface Nets do not leave a depth-noise rim.
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
