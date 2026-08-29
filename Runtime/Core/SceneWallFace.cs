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
}
