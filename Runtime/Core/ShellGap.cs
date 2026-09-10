using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>Which captured surface a shell cell or gap belongs to.</summary>
    public enum ShellSurfaceKind : byte
    {
        Wall = 0,
        Floor = 1,
        Ceiling = 2,
        Furniture = 3
    }

    /// <summary>
    /// One boundary loop of the live mesh: a connected run of edges the
    /// surface crosses but that could not be meshed. The scan frontier is the
    /// largest one; a leak in a finished room is a small one.
    /// </summary>
    public readonly struct MeshHole
    {
        /// <summary>World-space centroid of the loop's edge midpoints.</summary>
        public readonly Vector3 Center;
        /// <summary>
        /// Loop length in metres as counted (edges × voxel size). Voxel-edge
        /// loops are ragged, so this runs ~2× the smooth perimeter; use
        /// <see cref="ApproxWidthMetres"/> to picture the hole.
        /// </summary>
        public readonly float PerimeterMetres;
        /// <summary>Boundary edges in the loop.</summary>
        public readonly int Edges;
        /// <summary>Rough across-size of the hole: smooth perimeter ÷ π, with the ragged count halved.</summary>
        public float ApproxWidthMetres => PerimeterMetres * 0.5f / Mathf.PI;

        public MeshHole(Vector3 center, float perimeterMetres, int edges)
        {
            Center = center;
            PerimeterMetres = perimeterMetres;
            Edges = edges;
        }
    }

    /// <summary>
    /// Analytic closure of the live mesh, from its boundary edges. Independent
    /// of any scene model: a watertight surface has no boundary; every
    /// boundary edge not sitting on a clip plane or the volume edge is a hole.
    /// </summary>
    public readonly struct MeshClosure
    {
        /// <summary>
        /// 1 / (1 + OpenBoundaryMetres / (ref × √MeshAreaM2)). 1 for a closed
        /// surface; loops shorter than the hole minimum do not count.
        /// </summary>
        public readonly float Closure;
        /// <summary>Total length of counted hole loops, metres.</summary>
        public readonly float OpenBoundaryMetres;
        /// <summary>Mesh area estimate (quads × voxel²), m².</summary>
        public readonly float MeshAreaM2;
        /// <summary>Hole loops at or above the minimum perimeter.</summary>
        public readonly int HoleCount;
        /// <summary>All boundary edges that are not cuts, including tiny loops.</summary>
        public readonly int HoleEdges;
        /// <summary>Boundary edges on a clip plane / room AABB / volume edge.</summary>
        public readonly int CutEdges;
        /// <summary>Boundary edges the extract recorded (may exceed the analysed capacity).</summary>
        public readonly int OpenEdgesTotal;
        /// <summary>Largest hole loop, or default when none.</summary>
        public readonly MeshHole LargestHole;

        public MeshClosure(float closure, float openBoundaryMetres, float meshAreaM2, int holeCount,
            int holeEdges, int cutEdges, int openEdgesTotal, MeshHole largestHole)
        {
            Closure = closure;
            OpenBoundaryMetres = openBoundaryMetres;
            MeshAreaM2 = meshAreaM2;
            HoleCount = holeCount;
            HoleEdges = holeEdges;
            CutEdges = cutEdges;
            OpenEdgesTotal = openEdgesTotal;
            LargestHole = largestHole;
        }
    }

    /// <summary>
    /// A connected patch of shell cells with no scanned surface in front of
    /// them — a place passthrough would leak through the finished mesh.
    /// Reported largest first by <see cref="RoomScanSession.CopyShellGaps"/>.
    /// </summary>
    public readonly struct ShellGap
    {
        /// <summary>World-space centre of the patch on the captured surface.</summary>
        public readonly Vector3 Center;
        /// <summary>Unit march direction: into the room for walls / floor / ceiling, into the box for furniture.</summary>
        public readonly Vector3 Normal;
        /// <summary>Number of shell cells in the patch.</summary>
        public readonly int Cells;
        /// <summary>Approximate area in square metres (cells × cell²).</summary>
        public readonly float AreaM2;
        public readonly ShellSurfaceKind Kind;

        public ShellGap(Vector3 center, Vector3 normal, int cells, float areaM2, ShellSurfaceKind kind)
        {
            Center = center;
            Normal = normal;
            Cells = cells;
            AreaM2 = areaM2;
            Kind = kind;
        }
    }
}
