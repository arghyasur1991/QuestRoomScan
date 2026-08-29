using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// One captured <c>WALL_FACE</c> in a Space Setup room. Hosts pin
    /// world-space UI without taking an MRUK dependency.
    /// </summary>
    public readonly struct SceneWallFace
    {
        public readonly Vector3 Center;
        /// <summary>Unit vector from the wall into the room.</summary>
        public readonly Vector3 Inward;
        public readonly float Width;
        public readonly float Height;
        /// <summary>Floor height under this room (metres, world Y).</summary>
        public readonly float FloorY;
        /// <summary>True for <c>SCREEN</c> / <c>WALL_ART</c> — prefer another wall.</summary>
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
}
