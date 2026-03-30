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
        private bool _roomResolved;

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
            if (_room == null || _room.Anchors == null) return;

            for (int a = 0; a < _room.Anchors.Count; a++)
            {
                var anchor = _room.Anchors[a];
                var t = anchor.transform;
                var surfType = ClassifyAnchor(anchor);
                string label = GetAnchorLabel(anchor);

                var size = Vector3.one;
                if (anchor.VolumeBounds.HasValue)
                    size = anchor.VolumeBounds.Value.size;
                else if (anchor.PlaneRect.HasValue)
                {
                    var r = anchor.PlaneRect.Value;
                    size = new Vector3(r.width, 0.05f, r.height);
                }

                registry.Add(new SceneObject
                {
                    id = $"mruk_{a}_{label}",
                    label = label,
                    source = SceneObjectSource.MRUK,
                    surfaceType = surfType,
                    confidence = 1f,
                    position = t.position,
                    rotation = t.rotation,
                    size = size,
                    mrukLabel = anchor.Label.ToString(),
                    anchorUuid = anchor.Anchor != null ? anchor.Anchor.Uuid.ToString() : ""
                });
            }
        }

        private static string GetAnchorLabel(MRUKAnchor anchor)
        {
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR)) return "floor";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING)) return "ceiling";
            if (anchor.HasAnyLabel(WallLabels)) return "wall";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.TABLE)) return "table";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.COUCH)) return "couch";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.BED)) return "bed";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.LAMP)) return "lamp";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.STORAGE)) return "storage";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.SCREEN)) return "screen";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.PLANT)) return "plant";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.OTHER)) return "other";
            return "unknown";
        }

        // ─────────────────────────────────────────────────────────────
        //  Internals
        // ─────────────────────────────────────────────────────────────

        private void EnsureRoom()
        {
            if (_roomResolved) return;
            _roomResolved = true;

            var mruk = FindFirstObjectByType<MRUK>();
            if (mruk == null) return;
            _room = mruk.GetCurrentRoom();
            if (_room == null && mruk.Rooms != null && mruk.Rooms.Count > 0)
                _room = mruk.Rooms[0];
        }

        /// <summary>Re-query MRUK for the current room (e.g., after scene reload).</summary>
        public void RefreshRoom()
        {
            _roomResolved = false;
            EnsureRoom();
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
            MRUKAnchor.SceneLabels.OTHER;

        private static SurfaceType ClassifyAnchor(MRUKAnchor anchor)
        {
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR))
                return SurfaceType.Floor;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING))
                return SurfaceType.Ceiling;
            if (anchor.HasAnyLabel(WallLabels))
                return SurfaceType.Wall;
            if (anchor.HasAnyLabel(FurnitureLabels))
                return SurfaceType.Furniture;
            return SurfaceType.Unknown;
        }

        private static bool IsWall(MRUKAnchor anchor) => anchor.HasAnyLabel(WallLabels);
        private static bool IsFurniture(MRUKAnchor anchor) => anchor.HasAnyLabel(FurnitureLabels);
    }
}
