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
    /// Occupancy, wall faces, and classification all live here — hosts
    /// should still prefer <see cref="RoomScanSession"/> rather than
    /// taking an MRUK dependency. Falls back to vertex-normal heuristics
    /// when MRUK room data is unavailable.
    /// </summary>
    public class RoomUnderstanding : MonoBehaviour, IRoomScanModule
    {
        public static RoomUnderstanding Instance { get; private set; }

        public string ModuleName => "Room Understanding";
        public void OnModuleInitialize(RoomScanner scanner) { }

        private MRUKRoom _room;
        private MRUK _mruk;
        private bool _subscribedToRoomEvents;

        void Awake()
        {
            Instance = this;
        }

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

        /// <summary>
        /// Floor plane from MRUK, or a default Y=0 plane.
        ///
        /// <para>The normal is the anchor's <c>forward</c>, which is how MRUK
        /// orients a plane anchor — same convention <see cref="GetWallPlanes"/>
        /// uses. A floor anchor is therefore pitched ninety degrees so that its
        /// forward points at the ceiling, and its <c>up</c> lies flat along the
        /// floor. Reading <c>up</c> instead returns a plane standing on edge:
        /// it still passes through the right point, so anything asking whether
        /// a point is on the floor gets a plausible answer, while anything
        /// asking how high the floor is gets the height of the room's origin.
        /// </para>
        /// </summary>
        public Plane GetFloorPlane()
        {
            EnsureRoom();
            if (_room != null && _room.FloorAnchors != null && _room.FloorAnchors.Count > 0)
            {
                var ft = _room.FloorAnchors[0].transform;
                Vector3 normal = ft.forward;
                if (Vector3.Dot(normal, Vector3.up) < 0f) normal = -normal;
                return new Plane(normal, ft.position);
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
        //  Occupancy + wall faces (multi-room; never GetCurrentRoom)
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// True when the headset is inside <b>any</b> loaded captured space
        /// (outer wall planes, including doorway faces). Boot / Space Setup:
        /// any set-up room is enough. A loaded scan is tied to one room —
        /// use <see cref="IsHeadsetInsideRoom"/>. Native
        /// <c>IsPositionInRoom</c> is the floor outline and stays true a
        /// little past a doorway. <c>GetCurrentRoom()</c> is last/first after
        /// you leave — do not use it. Editor returns true.
        /// </summary>
        public bool IsHeadsetInsideAnyRoom()
        {
            if (Application.isEditor) return true;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null || _mruk.Rooms.Count == 0)
                return false;
            return Query.FindContaining(_mruk.Rooms, Query.HeadsetWorldPosition()) != null;
        }

        /// <summary>
        /// True when the headset is inside the captured space with this
        /// Scene API room UUID. False when the UUID is empty, the room is
        /// not in the loaded scene model, or the headset has left that
        /// room — even if another captured room still contains the
        /// headset. Editor returns true.
        /// </summary>
        public bool IsHeadsetInsideRoom(Guid sceneRoomUuid)
        {
            if (Application.isEditor) return true;
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            var room = Query.FindByUuid(_mruk.Rooms, sceneRoomUuid);
            return Query.Contains(room, Query.HeadsetWorldPosition());
        }

        /// <summary>True when a loaded MRUK room still has this Scene API UUID.</summary>
        public bool HasRoom(Guid sceneRoomUuid)
        {
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            return Query.FindByUuid(_mruk.Rooms, sceneRoomUuid) != null;
        }

        /// <summary>
        /// Scene API UUID of the loaded room that contains
        /// <paramref name="worldPos"/> (wall-plane test), or
        /// <see cref="Guid.Empty"/>.
        /// </summary>
        public Guid TryGetRoomUuidAt(Vector3 worldPos)
        {
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null)
                return Guid.Empty;
            return Query.RoomUuid(Query.FindContaining(_mruk.Rooms, worldPos));
        }

        /// <summary>Scene API UUID of the loaded room that contains the headset, or empty.</summary>
        public Guid TryGetRoomUuidContainingHeadset()
            => TryGetRoomUuidAt(Query.HeadsetWorldPosition());

        /// <summary>
        /// Visible <c>WALL_FACE</c> and <c>SCREEN</c> planes of the room that
        /// contains <paramref name="worldPos"/> (not doorway / inner faces).
        /// Returns 0 in the editor and when the point is not inside a
        /// captured room. Clears <paramref name="dest"/>.
        /// </summary>
        public int CopyWallFacesOfRoomContaining(Vector3 worldPos, List<SceneWallFace> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (Application.isEditor) return 0;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyWallFaces(Query.FindContaining(_mruk.Rooms, worldPos), dest);
        }

        /// <summary>
        /// Visible <c>WALL_FACE</c> and <c>SCREEN</c> (TV) planes of the
        /// room that contains the headset. Hosts pin world-space UI to these
        /// without taking an MRUK dependency. A <see cref="SceneWallFace.IsScreen"/>
        /// row is the television — pin on it rather than a blank wall.
        /// </summary>
        public int CopyHeadsetRoomWallFaces(List<SceneWallFace> dest)
            => CopyWallFacesOfRoomContaining(Query.HeadsetWorldPosition(), dest);

        /// <summary>
        /// True when <paramref name="worldPos"/> is inside the captured
        /// space with this Scene API room UUID (floor outline + outer
        /// walls). False when the UUID is empty or the room is not loaded.
        /// </summary>
        public bool Contains(Guid sceneRoomUuid, Vector3 worldPos)
        {
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            return Query.Contains(Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), worldPos);
        }

        /// <summary>
        /// Half-spaces that bound <paramref name="sceneRoomUuid"/> for GPU
        /// TSDF clip: each <c>Vector4(n, w)</c> is outside when
        /// <c>dot(pos, n) &lt; w</c>. Outer walls (including doorway
        /// <c>INVISIBLE_WALL_FACE</c>) use the same 8 cm inset as occupancy.
        /// Floor / ceiling use a 4 cm slack so voxels just below the floor
        /// plane still integrate. Clears <paramref name="dest"/>. Returns 0
        /// when the room is missing.
        /// </summary>
        public int CopyRoomClipPlanes(Guid sceneRoomUuid, List<Vector4> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (sceneRoomUuid == Guid.Empty) return 0;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyRoomClipPlanes(
                Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), dest);
        }

        /// <summary>
        /// <c>SCREEN</c> (TV) stamps for the room. Clears
        /// <paramref name="dest"/>. At most 4. Empty when the UUID is
        /// missing or the room has no television.
        /// </summary>
        public int CopyScreenStamps(Guid sceneRoomUuid, List<ScanScreenStamp> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (sceneRoomUuid == Guid.Empty) return 0;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyScreenStamps(
                Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), dest);
        }

        /// <summary>
        /// Conservative world AABB of the captured room (outer walls including
        /// doorway <c>INVISIBLE_WALL_FACE</c>, floor, ceiling). No 8 cm inset —
        /// GPU clip still uses the inset half-spaces; this box is only an
        /// early-out. False when the UUID is missing.
        /// </summary>
        public bool CopyRoomWorldAabb(Guid sceneRoomUuid, out Vector3 min, out Vector3 max)
        {
            min = max = Vector3.zero;
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            return Query.CopyRoomWorldAabb(
                Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), out min, out max);
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

        void EnsureMruk()
        {
            if (_mruk == null)
                _mruk = FindAnyObjectByType<MRUK>();
        }

        /// <summary>
        /// Cached classification room is the volume that contains the
        /// headset. <c>GetCurrentRoom()</c> is last/first after you leave
        /// and must not be used. Hallway / no rooms → <c>_room</c> is null
        /// (do not fall back to <c>Rooms[0]</c>).
        /// </summary>
        private void EnsureRoom()
        {
            EnsureMruk();
            MRUKRoom next = null;
            if (_mruk != null && _mruk.Rooms != null)
                next = Query.FindContaining(_mruk.Rooms, Query.HeadsetWorldPosition());

            if (next != _room)
            {
                if (_room != null)
                    _room.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);
                _room = next;
                if (_room != null)
                    _room.AnchorCreatedEvent.AddListener(OnAnchorCreated);
            }

            SubscribeToRoomEvents();
        }

        private void SubscribeToRoomEvents()
        {
            if (_subscribedToRoomEvents) return;
            if (_mruk == null) return;

            _mruk.RoomCreatedEvent.AddListener(OnRoomCreatedOrUpdated);
            _mruk.RoomUpdatedEvent.AddListener(OnRoomCreatedOrUpdated);
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
            EnsureRoom();
            Logger.Info($"[RoomUnderstanding] Room created/updated — " +
                        $"headset room anchors={_room?.Anchors?.Count ?? 0} " +
                        $"(event room={room?.Anchors?.Count ?? 0})");
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
            if (Instance == this) Instance = null;
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

        /// <summary>
        /// Shared wall-plane occupancy. Native <c>IsPositionInRoom</c> is the
        /// floor outline; Horizon Space Setup uses the walls.
        /// </summary>
        internal static class Query
        {
            const float OuterWallInsetMetres = 0.08f;

            const MRUKAnchor.SceneLabels OuterWallLabels =
                MRUKAnchor.SceneLabels.WALL_FACE
                | MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE;

            const MRUKAnchor.SceneLabels PinWallAvoidLabels =
                MRUKAnchor.SceneLabels.WALL_ART;

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

            internal static int CopyWallFaces(MRUKRoom room, List<SceneWallFace> dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (room == null || room.Anchors == null) return 0;

                float floorY = FloorY(room);
                for (int i = 0; i < room.Anchors.Count; i++)
                    TryAddPinSurface(room, room.Anchors[i], floorY, dst);

                return dst.Count;
            }

            static void TryAddPinSurface(
                MRUKRoom room, MRUKAnchor a, float floorY, List<SceneWallFace> dst)
            {
                if (a == null) return;

                bool isScreen = a.HasAnyLabel(MRUKAnchor.SceneLabels.SCREEN);
                bool isWall = a.HasAnyLabel(MRUKAnchor.SceneLabels.WALL_FACE)
                    && !a.HasAnyLabel(MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE)
                    && !a.HasAnyLabel(MRUKAnchor.SceneLabels.INNER_WALL_FACE);
                if (!isScreen && !isWall) return;

                Vector3 inward = Inward(room, a);
                if (Vector3.Dot(inward, Vector3.up) > 0.7f
                    || Vector3.Dot(inward, Vector3.up) < -0.7f)
                    return;

                if (!TryPlaneSize(a, out Vector3 center, out float width, out float height))
                    return;

                float min = isScreen ? 0.15f : 0.2f;
                if (width < min || height < min) return;

                bool avoid = !isScreen && a.HasAnyLabel(PinWallAvoidLabels);
                dst.Add(new SceneWallFace(
                    center, inward, width, height, floorY, avoid, isScreen));
            }

            static bool TryPlaneSize(
                MRUKAnchor a, out Vector3 center, out float width, out float height)
            {
                center = default;
                width = 0f;
                height = 0f;

                if (a.PlaneRect.HasValue)
                {
                    var rect = a.PlaneRect.Value;
                    Vector3 worldX = a.transform.TransformVector(new Vector3(rect.width, 0f, 0f));
                    Vector3 worldY = a.transform.TransformVector(new Vector3(0f, rect.height, 0f));
                    float horizX = Vector3.ProjectOnPlane(worldX, Vector3.up).magnitude;
                    float horizY = Vector3.ProjectOnPlane(worldY, Vector3.up).magnitude;
                    width = Mathf.Max(horizX, horizY);
                    height = Mathf.Max(
                        Mathf.Abs(Vector3.Dot(worldX, Vector3.up)),
                        Mathf.Abs(Vector3.Dot(worldY, Vector3.up)));
                    center = a.transform.TransformPoint(rect.center);
                    return true;
                }

                if (!a.VolumeBounds.HasValue) return false;

                var size = a.VolumeBounds.Value.size;
                Vector3 wx = a.transform.TransformVector(new Vector3(size.x, 0f, 0f));
                Vector3 wy = a.transform.TransformVector(new Vector3(0f, size.y, 0f));
                Vector3 wz = a.transform.TransformVector(new Vector3(0f, 0f, size.z));
                float hx = Vector3.ProjectOnPlane(wx, Vector3.up).magnitude;
                float hy = Vector3.ProjectOnPlane(wy, Vector3.up).magnitude;
                float hz = Vector3.ProjectOnPlane(wz, Vector3.up).magnitude;
                width = Mathf.Max(hx, Mathf.Max(hy, hz));
                height = Mathf.Max(
                    Mathf.Abs(Vector3.Dot(wx, Vector3.up)),
                    Mathf.Max(
                        Mathf.Abs(Vector3.Dot(wy, Vector3.up)),
                        Mathf.Abs(Vector3.Dot(wz, Vector3.up))));
                center = a.GetAnchorCenter();
                return true;
            }

            static float FloorY(MRUKRoom room)
            {
                if (room.FloorAnchors != null && room.FloorAnchors.Count > 0)
                {
                    var f = room.FloorAnchors[0];
                    if (f != null) return f.GetAnchorCenter().y;
                }

                return 0f;
            }

            static Vector3 Inward(MRUKRoom room, MRUKAnchor a)
            {
                Vector3 inward = room.GetFacingDirection(a);
                if (inward.sqrMagnitude < 1e-8f)
                    inward = a.transform.forward;
                inward.Normalize();
                return inward;
            }

            static bool InsideOuterWalls(MRUKRoom room, Vector3 worldPos)
            {
                var anchors = room.Anchors;
                if (anchors == null || anchors.Count == 0)
                    return true;

                for (int i = 0; i < anchors.Count; i++)
                {
                    var a = anchors[i];
                    if (a == null) continue;
                    if (!a.HasAnyLabel(OuterWallLabels)) continue;

                    if (Vector3.Dot(worldPos - a.transform.position, Inward(room, a))
                        < OuterWallInsetMetres)
                        return false;
                }

                return true;
            }

            const float FloorCeilingSlackMetres = 0.04f;
            const int MaxRoomClipPlanes = 32;
            const int MaxScreenStamps = 4;

            internal static int CopyRoomClipPlanes(MRUKRoom room, List<Vector4> dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (room == null || room.Anchors == null) return 0;

                for (int i = 0; i < room.Anchors.Count && dst.Count < MaxRoomClipPlanes; i++)
                {
                    var a = room.Anchors[i];
                    if (a == null) continue;
                    if (!a.HasAnyLabel(OuterWallLabels)) continue;

                    Vector3 n = Inward(room, a);
                    float w = Vector3.Dot(a.transform.position, n) + OuterWallInsetMetres;
                    dst.Add(new Vector4(n.x, n.y, n.z, w));
                }

                if (dst.Count < MaxRoomClipPlanes)
                {
                    float fy = FloorY(room);
                    dst.Add(new Vector4(0f, 1f, 0f, fy - FloorCeilingSlackMetres));
                }

                if (dst.Count < MaxRoomClipPlanes
                    && room.CeilingAnchors != null
                    && room.CeilingAnchors.Count > 0
                    && room.CeilingAnchors[0] != null)
                {
                    float cy = room.CeilingAnchors[0].GetAnchorCenter().y;
                    dst.Add(new Vector4(0f, -1f, 0f, -(cy + FloorCeilingSlackMetres)));
                }

                return dst.Count;
            }

            internal static int CopyScreenStamps(MRUKRoom room, List<ScanScreenStamp> dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (room == null) return 0;

                var faces = new List<SceneWallFace>(4);
                CopyWallFaces(room, faces);
                for (int i = 0; i < faces.Count && dst.Count < MaxScreenStamps; i++)
                {
                    if (!faces[i].IsScreen) continue;
                    dst.Add(ScanScreenStamp.FromFace(faces[i]));
                }

                return dst.Count;
            }

            internal static bool CopyRoomWorldAabb(
                MRUKRoom room, out Vector3 min, out Vector3 max)
            {
                min = max = Vector3.zero;
                if (room == null) return false;

                bool any = false;
                Bounds b = default;
                if (room.Anchors != null)
                {
                    for (int i = 0; i < room.Anchors.Count; i++)
                    {
                        var a = room.Anchors[i];
                        if (a == null) continue;
                        if (!a.HasAnyLabel(OuterWallLabels)
                            && !a.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR)
                            && !a.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING))
                            continue;
                        EncapsulateAnchor(a, ref b, ref any);
                    }
                }

                if (!any) return false;
                min = b.min;
                max = b.max;
                return true;
            }

            static void EncapsulateAnchor(MRUKAnchor a, ref Bounds b, ref bool any)
            {
                if (a.PlaneRect.HasValue)
                {
                    var r = a.PlaneRect.Value;
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMin, r.yMin, 0f)), ref b, ref any);
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMax, r.yMin, 0f)), ref b, ref any);
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMin, r.yMax, 0f)), ref b, ref any);
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMax, r.yMax, 0f)), ref b, ref any);
                    return;
                }

                if (!a.VolumeBounds.HasValue) return;
                var vb = a.VolumeBounds.Value;
                Vector3 c = vb.center;
                Vector3 e = vb.extents;
                for (int x = -1; x <= 1; x += 2)
                for (int y = -1; y <= 1; y += 2)
                for (int z = -1; z <= 1; z += 2)
                    EncapsulatePoint(
                        a.transform.TransformPoint(c + Vector3.Scale(e, new Vector3(x, y, z))),
                        ref b, ref any);
            }

            static void EncapsulatePoint(Vector3 p, ref Bounds b, ref bool any)
            {
                if (!any)
                {
                    b = new Bounds(p, Vector3.zero);
                    any = true;
                }
                else
                {
                    b.Encapsulate(p);
                }
            }
        }
    }
}
