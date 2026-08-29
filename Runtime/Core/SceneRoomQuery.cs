using System;
using System.Collections.Generic;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Shared "is this world point inside a captured Space Setup room"
    /// test. Native <c>IsPositionInRoom</c> is the floor outline and stays
    /// true a little past a doorway; Horizon Space Setup uses the walls.
    /// One implementation for "any loaded room" (boot) and "this package's
    /// room" (a loaded scan).
    /// </summary>
    static class SceneRoomQuery
    {
        const float OuterWallInsetMetres = 0.08f;

        const MRUKAnchor.SceneLabels OuterWallLabels =
            MRUKAnchor.SceneLabels.WALL_FACE
            | MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE;

        internal static Vector3 HeadsetWorldPosition()
        {
            var rig = UnityEngine.Object.FindAnyObjectByType<OVRCameraRig>(
                FindObjectsInactive.Include);
            if (rig != null && rig.centerEyeAnchor != null)
                return rig.centerEyeAnchor.position;
            var cam = Camera.main;
            return cam != null ? cam.transform.position : Vector3.zero;
        }

        internal static Guid RoomUuid(MRUKRoom room)
        {
            if (room == null || room.Anchor == OVRAnchor.Null)
                return Guid.Empty;
            return room.Anchor.Uuid;
        }

        internal static bool Contains(MRUKRoom room, Vector3 worldPos)
        {
            if (room == null) return false;
            if (!room.IsPositionInRoom(worldPos, testVerticalBounds: true))
                return false;
            return InsideOuterWalls(room, worldPos);
        }

        internal static MRUKRoom FindContaining(IList<MRUKRoom> rooms, Vector3 worldPos)
        {
            if (rooms == null) return null;
            for (int i = 0; i < rooms.Count; i++)
            {
                var room = rooms[i];
                if (Contains(room, worldPos))
                    return room;
            }
            return null;
        }

        internal static MRUKRoom FindByUuid(IList<MRUKRoom> rooms, Guid uuid)
        {
            if (rooms == null || uuid == Guid.Empty) return null;
            for (int i = 0; i < rooms.Count; i++)
            {
                var room = rooms[i];
                if (RoomUuid(room) == uuid)
                    return room;
            }
            return null;
        }

        /// <summary>
        /// Floor-outline <c>IsPositionInRoom</c> does not drop at a doorway.
        /// Wall planes (and invisible faces across openings) do: each plane's
        /// +Z faces into the room, so just outside a door is a negative
        /// half-space even when the floor polygon still contains the head.
        /// </summary>
        static bool InsideOuterWalls(MRUKRoom room, Vector3 worldPos)
        {
            var anchors = room.Anchors;
            if (anchors == null || anchors.Count == 0)
                return true;

            for (int i = 0; i < anchors.Count; i++)
            {
                var a = anchors[i];
                if (a == null) continue;
                if ((a.Label & OuterWallLabels) == 0) continue;

                Vector3 inward = room.GetFacingDirection(a);
                if (inward.sqrMagnitude < 1e-8f)
                    inward = a.transform.forward;
                inward.Normalize();
                if (Vector3.Dot(worldPos - a.transform.position, inward) < OuterWallInsetMetres)
                    return false;
            }

            return true;
        }
    }
}
