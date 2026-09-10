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
    /// A connected patch of shell cells with no scanned surface in front of
    /// them — a place passthrough would leak through the finished mesh.
    /// Reported largest first by <see cref="RoomScanSession.CopyShellGaps"/>.
    /// </summary>
    public readonly struct ShellGap
    {
        /// <summary>World-space centre of the patch on the captured surface.</summary>
        public readonly Vector3 Center;
        /// <summary>Unit normal pointing into the room (outward for furniture).</summary>
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
