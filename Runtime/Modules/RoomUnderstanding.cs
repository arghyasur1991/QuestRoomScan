using System;
using System.Collections.Generic;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Surface type classification for mesh vertices.
    /// </summary>
    public enum SurfaceType : byte
    {
        Unknown   = 0,
        Floor     = 1,
        Ceiling   = 2,
        Wall      = 3,
        Furniture = 4
    }

    /// <summary>
    /// Wraps Meta MRUK APIs to provide semantic room understanding.
    /// Game clients query this instead of MRUK directly.
    /// Falls back to vertex-normal heuristics when MRUK room data is unavailable.
    /// </summary>
    public class RoomUnderstanding : MonoBehaviour, IRoomScanModule
    {
        public string ModuleName => "Room Understanding";
        public void OnModuleInitialize(RoomScanner scanner) { }

        private MRUKRoom _room;
        private MRUK _mruk;
        private bool _subscribedToRoomEvents;

        /// <summary>
        /// Raised when MRUK anchors change (created, updated, or room updated).
        /// RoomScanner subscribes to this to re-populate the SceneObjectRegistry reactively
        /// instead of polling.
        /// </summary>
        public event Action AnchorsChanged;

        // Cached per-vertex classification
        private SurfaceType[] _lastClassification;

        // ─────────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Classify a single world-space position into a <see cref="SurfaceType"/>.
        /// </summary>
        public SurfaceType GetSurfaceType(Vector3 worldPos)
        {
            EnsureRoom();
            if (_room == null) return SurfaceType.Unknown;

            MRUKAnchor best = null;
            float bestDist = float.MaxValue;

            foreach (var anchor in _room.Anchors)
            {
                float d = Vector3.Distance(anchor.transform.position, worldPos);
                if (d < bestDist) { bestDist = d; best = anchor; }
            }

            return best != null ? ClassifyAnchor(best) : SurfaceType.Unknown;
        }

        /// <summary>
        /// Bulk classification: returns a <see cref="SurfaceType"/> per vertex.
        /// When MRUK data is available, each vertex is matched to the nearest
        /// labelled anchor. Otherwise, uses vertex normal heuristics.
        /// </summary>
        public SurfaceType[] GetPerVertexSurfaceTypes(Mesh mesh)
        {
            if (mesh == null) return null;
            var verts = mesh.vertices;
            var normals = mesh.normals;
            var result = new SurfaceType[verts.Length];

            EnsureRoom();
            if (_room != null && _room.Anchors.Count > 0)
                ClassifyFromMRUK(verts, result);
            else
                ClassifyFromNormals(normals, result);

            _lastClassification = result;
            return result;
        }

        /// <summary>Returns the last computed classification, or null.</summary>
        public SurfaceType[] LastClassification => _lastClassification;

        /// <summary>All wall planes from MRUK, or empty if unavailable.</summary>
        public List<Plane> GetWallPlanes()
        {
            var planes = new List<Plane>();
            EnsureRoom();
            if (_room == null) return planes;

            foreach (var anchor in _room.Anchors)
            {
                if (!IsWall(anchor)) continue;
                var t = anchor.transform;
                planes.Add(new Plane(t.forward, t.position));
            }
            return planes;
        }

        /// <summary>Floor plane from MRUK, or a default Y=0 plane.</summary>
        public Plane GetFloorPlane()
        {
            EnsureRoom();
            if (_room != null && _room.FloorAnchors != null && _room.FloorAnchors.Count > 0)
            {
                var ft = _room.FloorAnchors[0].transform;
                return new Plane(ft.up, ft.position);
            }
            return new Plane(Vector3.up, Vector3.zero);
        }

        /// <summary>Bounding boxes for all furniture anchors.</summary>
        public List<Bounds> GetFurnitureBounds()
        {
            var result = new List<Bounds>();
            EnsureRoom();
            if (_room == null) return result;

            foreach (var anchor in _room.Anchors)
            {
                if (!IsFurniture(anchor)) continue;
                if (anchor.VolumeBounds.HasValue)
                    result.Add(anchor.VolumeBounds.Value);
                else if (anchor.PlaneRect.HasValue)
                {
                    var r = anchor.PlaneRect.Value;
                    var center = anchor.transform.TransformPoint(r.center);
                    result.Add(new Bounds(center, new Vector3(r.width, 0.1f, r.height)));
                }
            }
            return result;
        }

        // ─────────────────────────────────────────────────────────────
        //  Scene Object Registry population
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Captures all MRUK anchors as SceneObjects and adds them to the registry.
        /// Each anchor gets a unique ID, full label, pose, bounds, and surface type.
        /// </summary>
        public void PopulateRegistry(SceneObjectRegistry registry)
        {
            if (registry == null) return;
            EnsureRoom();
            if (_room == null || _room.Anchors == null)
            {
                Logger.Warning("[RoomUnderstanding] PopulateRegistry — no MRUK room available");
                return;
            }

            int added = 0;
            int skipped = 0;
            for (int a = 0; a < _room.Anchors.Count; a++)
            {
                var anchor = _room.Anchors[a];
                string label = GetAnchorLabel(anchor);
                if (label == null)
                {
                    skipped++;
                    continue;
                }

                var t = anchor.transform;
                var surfType = ClassifyAnchor(anchor);

                var size = Vector3.one;
                var rot = t.rotation;
                bool hasVolume = anchor.VolumeBounds.HasValue;
                bool hasPlane = anchor.PlaneRect.HasValue;
                if (hasVolume)
                    size = anchor.VolumeBounds.Value.size;
                else if (hasPlane)
                {
                    var r = anchor.PlaneRect.Value;
                    size = new Vector3(r.width, r.height, 0.05f);
                }

                // anchor.transform.position is at the TOP face for volumes;
                // GetAnchorCenter() returns the true geometric center.
                var worldCenter = anchor.GetAnchorCenter();

                // Floor/ceiling anchors have an arbitrary in-plane yaw.
                // Derive the room's horizontal orientation from a wall anchor,
                // then project the PlaneRect corners into that wall-aligned frame
                // so the bounding box aligns with the physical room layout.
                if ((surfType == SurfaceType.Floor || surfType == SurfaceType.Ceiling) && hasPlane)
                {
                    var wallRight = FindWallHorizontalRight();
                    var wallPerp  = Vector3.Cross(Vector3.up, wallRight).normalized;

                    // Rotation: local X→wallRight, local Y→wallPerp, local Z→up (thin)
                    rot = Quaternion.LookRotation(Vector3.up, wallPerp);

                    // Project the anchor's PlaneRect corners into wall-aligned frame
                    var pr = anchor.PlaneRect.Value;
                    var hw = pr.width  * 0.5f;
                    var hh = pr.height * 0.5f;
                    var c0 = t.TransformPoint(new Vector3(-hw, -hh, 0));
                    var c1 = t.TransformPoint(new Vector3( hw, -hh, 0));
                    var c2 = t.TransformPoint(new Vector3( hw,  hh, 0));
                    var c3 = t.TransformPoint(new Vector3(-hw,  hh, 0));

                    float minW = float.MaxValue, maxW = float.MinValue;
                    float minD = float.MaxValue, maxD = float.MinValue;
                    foreach (var corner in new[] { c0, c1, c2, c3 })
                    {
                        var offset = corner - worldCenter;
                        float projW = Vector3.Dot(offset, wallRight);
                        float projD = Vector3.Dot(offset, wallPerp);
                        if (projW < minW) minW = projW; if (projW > maxW) maxW = projW;
                        if (projD < minD) minD = projD; if (projD > maxD) maxD = projD;
                    }

                    size = new Vector3(maxW - minW, maxD - minD, 0.05f);

                    // Re-center to the centroid of the wall-aligned bounding box
                    float offW = (minW + maxW) * 0.5f;
                    float offD = (minD + maxD) * 0.5f;
                    worldCenter += wallRight * offW + wallPerp * offD;
                }

                registry.Add(new SceneObject
                {
                    id = $"mruk_{a}_{label}",
                    label = label,
                    source = SceneObjectSource.MRUK,
                    surfaceType = surfType,
                    confidence = 1f,
                    position = worldCenter,
                    rotation = rot,
                    size = size,
                    mrukLabel = anchor.Label.ToString(),
                    anchorUuid = anchor.Anchor != null ? anchor.Anchor.Uuid.ToString() : ""
                });
                added++;

                Logger.Info($"[RoomUnderstanding] Anchor[{a}]: label={label}, " +
                            $"rawLabel={anchor.Label}, vol={hasVolume}, plane={hasPlane}, " +
                            $"size={size}, rot={rot.eulerAngles}, center={worldCenter}, anchorPos={t.position}");
            }
            Logger.Info($"[RoomUnderstanding] Populated {added} MRUK objects " +
                        $"(from {_room.Anchors.Count} anchors, {skipped} skipped)");
        }

        /// <summary>
        /// Returns a horizontal "right" direction along the first wall anchor found
        /// in the current room, giving us a reliable room-aligned axis.
        /// Falls back to world-right if no wall exists.
        /// </summary>
        private Vector3 FindWallHorizontalRight()
        {
            if (_room != null)
            {
                foreach (var a in _room.Anchors)
                {
                    if (!a.HasAnyLabel(WallLabels)) continue;
                    var right = a.transform.right;
                    right.y = 0;
                    if (right.sqrMagnitude > 0.001f)
                        return right.normalized;
                }
            }
            return Vector3.right;
        }

        private static string GetAnchorLabel(MRUKAnchor anchor)
        {
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.GLOBAL_MESH)) return null;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR)) return "floor";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING)) return "ceiling";
            if (anchor.HasAnyLabel(WallLabels)) return "wall";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.DOOR_FRAME)) return "door";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.WINDOW_FRAME)) return "window";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.TABLE)) return "table";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.COUCH)) return "couch";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.BED)) return "bed";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.LAMP)) return "lamp";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.STORAGE)) return "storage";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.SCREEN)) return "screen";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.PLANT)) return "plant";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.WALL_ART)) return "wall_art";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.OTHER)) return "other";
            return anchor.Label.ToString().ToLowerInvariant();
        }

        // ─────────────────────────────────────────────────────────────
        //  Internals
        // ─────────────────────────────────────────────────────────────

        private void EnsureRoom()
        {
            if (_room != null) return;

            if (_mruk == null)
                _mruk = FindFirstObjectByType<MRUK>();
            if (_mruk == null) return;

            _room = _mruk.GetCurrentRoom();
            if (_room == null && _mruk.Rooms != null && _mruk.Rooms.Count > 0)
                _room = _mruk.Rooms[0];

            SubscribeToRoomEvents();
        }

        private void SubscribeToRoomEvents()
        {
            if (_subscribedToRoomEvents) return;

            if (_mruk != null)
            {
                _mruk.RoomCreatedEvent.AddListener(OnRoomCreatedOrUpdated);
                _mruk.RoomUpdatedEvent.AddListener(OnRoomCreatedOrUpdated);
            }

            if (_room != null)
                _room.AnchorCreatedEvent.AddListener(OnAnchorCreated);

            _subscribedToRoomEvents = true;
        }

        private void UnsubscribeFromRoomEvents()
        {
            if (!_subscribedToRoomEvents) return;

            if (_mruk != null)
            {
                _mruk.RoomCreatedEvent.RemoveListener(OnRoomCreatedOrUpdated);
                _mruk.RoomUpdatedEvent.RemoveListener(OnRoomCreatedOrUpdated);
            }

            if (_room != null)
                _room.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);

            _subscribedToRoomEvents = false;
        }

        private void OnRoomCreatedOrUpdated(MRUKRoom room)
        {
            // Re-subscribe to the new/updated room's anchor events
            if (_room != null)
                _room.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);

            _room = room;
            _room.AnchorCreatedEvent.AddListener(OnAnchorCreated);

            Logger.Info($"[RoomUnderstanding] Room created/updated — {_room.Anchors?.Count ?? 0} anchors");
            AnchorsChanged?.Invoke();
        }

        private void OnAnchorCreated(MRUKAnchor anchor)
        {
            Logger.Info($"[RoomUnderstanding] Anchor created: {anchor.Label}");
            AnchorsChanged?.Invoke();
        }

        /// <summary>Re-query MRUK for the current room (e.g., after scene reload).</summary>
        public void RefreshRoom()
        {
            UnsubscribeFromRoomEvents();
            _room = null;
            EnsureRoom();
        }

        private void OnDestroy()
        {
            UnsubscribeFromRoomEvents();
        }

        private void ClassifyFromMRUK(Vector3[] verts, SurfaceType[] result)
        {
            var anchors = _room.Anchors;
            for (int v = 0; v < verts.Length; v++)
            {
                MRUKAnchor best = null;
                float bestDist = float.MaxValue;
                for (int a = 0; a < anchors.Count; a++)
                {
                    float d = Vector3.SqrMagnitude(anchors[a].transform.position - verts[v]);
                    if (d < bestDist) { bestDist = d; best = anchors[a]; }
                }
                result[v] = best != null ? ClassifyAnchor(best) : SurfaceType.Unknown;
            }
        }

        private static void ClassifyFromNormals(Vector3[] normals, SurfaceType[] result)
        {
            for (int i = 0; i < normals.Length; i++)
            {
                float ny = normals[i].y;
                if (ny > 0.7f) result[i] = SurfaceType.Floor;
                else if (ny < -0.7f) result[i] = SurfaceType.Ceiling;
                else result[i] = SurfaceType.Wall;
            }
        }

        private const MRUKAnchor.SceneLabels WallLabels =
            MRUKAnchor.SceneLabels.WALL_FACE |
            MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE |
            MRUKAnchor.SceneLabels.INNER_WALL_FACE;

        private const MRUKAnchor.SceneLabels FurnitureLabels =
            MRUKAnchor.SceneLabels.TABLE |
            MRUKAnchor.SceneLabels.COUCH |
            MRUKAnchor.SceneLabels.BED |
            MRUKAnchor.SceneLabels.LAMP |
            MRUKAnchor.SceneLabels.STORAGE |
            MRUKAnchor.SceneLabels.SCREEN |
            MRUKAnchor.SceneLabels.PLANT |
            MRUKAnchor.SceneLabels.WALL_ART |
            MRUKAnchor.SceneLabels.OTHER;

        private const MRUKAnchor.SceneLabels StructureLabels =
            MRUKAnchor.SceneLabels.DOOR_FRAME |
            MRUKAnchor.SceneLabels.WINDOW_FRAME;

        private static SurfaceType ClassifyAnchor(MRUKAnchor anchor)
        {
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.GLOBAL_MESH))
                return SurfaceType.Unknown;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR))
                return SurfaceType.Floor;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING))
                return SurfaceType.Ceiling;
            if (anchor.HasAnyLabel(WallLabels))
                return SurfaceType.Wall;
            if (anchor.HasAnyLabel(StructureLabels))
                return SurfaceType.Wall;
            if (anchor.HasAnyLabel(FurnitureLabels))
                return SurfaceType.Furniture;
            return SurfaceType.Furniture;
        }

        private static bool IsWall(MRUKAnchor anchor) => anchor.HasAnyLabel(WallLabels);
        private static bool IsFurniture(MRUKAnchor anchor) => anchor.HasAnyLabel(FurnitureLabels);
    }
}
