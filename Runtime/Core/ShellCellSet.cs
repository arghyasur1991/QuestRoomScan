using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Sample points on the captured room shell (walls, floor, ceiling,
    /// furniture faces). Each cell carries a world position, an inward
    /// march direction (into the room for planes, into the box for
    /// furniture), and a segment along it; the GPU reports whether a scanned
    /// surface lies on the segment, or whether the segment is all observed
    /// air. Fixed-capacity so the per-scan readback path allocates nothing.
    /// </summary>
    public sealed class ShellCellSet
    {
        public const int MaxCells = 16384;
        public const int MaxSurfaces = 96;

        public const byte StateExcluded = 0;
        public const byte StateUncovered = 1;
        public const byte StateCovered = 2;
        /// <summary>
        /// The whole segment is observed free space: nothing is there to
        /// scan (air inside a loose furniture box, glass). Leaves the
        /// denominator like an excluded cell, but is decided by the march
        /// each tick rather than at build time.
        /// </summary>
        public const byte StateEmpty = 3;

        public struct Surface
        {
            public int First;
            public int Nu, Nv;
            public ShellSurfaceKind Kind;
            public Vector3 AxisU;
            public Vector3 AxisV;
            public Vector3 Normal;
        }

        public float CellSize = 0.1f;
        public int CellCount;
        public int UploadCount;
        public int ExcludedCount;
        public int SurfaceCount;
        public int DroppedSurfaces;

        // Per cell (all cells, excluded included so grids stay rectangular).
        public readonly Vector3[] Pos = new Vector3[MaxCells];
        public readonly float[] MarchStart = new float[MaxCells];
        public readonly float[] MarchLength = new float[MaxCells];
        public readonly ushort[] SurfaceIndex = new ushort[MaxCells];
        public readonly ushort[] U = new ushort[MaxCells];
        public readonly ushort[] V = new ushort[MaxCells];
        public readonly byte[] State = new byte[MaxCells];
        /// <summary>Metres along the normal from <see cref="Pos"/> to the hit, NaN when uncovered.</summary>
        public readonly float[] HitOffset = new float[MaxCells];
        /// <summary>Index into the GPU arrays, or -1 for excluded cells.</summary>
        public readonly int[] GpuIndex = new int[MaxCells];

        // Compact upload arrays (non-excluded cells only).
        public readonly Vector4[] GpuPos = new Vector4[MaxCells];
        public readonly Vector4[] GpuNrm = new Vector4[MaxCells];
        public readonly int[] CellOfGpu = new int[MaxCells];

        public readonly Surface[] Surfaces = new Surface[MaxSurfaces];

        public void Clear()
        {
            CellCount = 0;
            UploadCount = 0;
            ExcludedCount = 0;
            SurfaceCount = 0;
            DroppedSurfaces = 0;
        }

        /// <summary>
        /// Reserve a grid. Returns -1 when the set is full; the caller skips
        /// that surface. Furniture is added last so it is dropped first.
        /// </summary>
        public int BeginSurface(ShellSurfaceKind kind, Vector3 axisU, Vector3 axisV, Vector3 normal, int nu, int nv)
        {
            if (nu <= 0 || nv <= 0) return -1;
            if (SurfaceCount >= MaxSurfaces || CellCount + nu * nv > MaxCells)
            {
                DroppedSurfaces++;
                return -1;
            }
            Surfaces[SurfaceCount] = new Surface
            {
                First = CellCount,
                Nu = nu,
                Nv = nv,
                Kind = kind,
                AxisU = axisU,
                AxisV = axisV,
                Normal = normal
            };
            return SurfaceCount++;
        }

        /// <summary>Cells must be added in v-major, u-minor order for the grid reserved by <see cref="BeginSurface"/>.</summary>
        public void AddCell(int surface, int u, int v, Vector3 pos, float marchStart, float marchLength, bool excluded)
        {
            int i = CellCount++;
            Pos[i] = pos;
            MarchStart[i] = marchStart;
            MarchLength[i] = marchLength;
            SurfaceIndex[i] = (ushort)surface;
            U[i] = (ushort)u;
            V[i] = (ushort)v;
            HitOffset[i] = float.NaN;
            if (excluded)
            {
                State[i] = StateExcluded;
                GpuIndex[i] = -1;
                ExcludedCount++;
                return;
            }
            State[i] = StateUncovered;
            var s = Surfaces[surface];
            int g = UploadCount++;
            GpuIndex[i] = g;
            CellOfGpu[g] = i;
            GpuPos[g] = new Vector4(pos.x, pos.y, pos.z, marchStart);
            GpuNrm[g] = new Vector4(s.Normal.x, s.Normal.y, s.Normal.z, marchLength);
        }

        public int CellIndex(int surface, int u, int v)
        {
            var s = Surfaces[surface];
            if (u < 0 || v < 0 || u >= s.Nu || v >= s.Nv) return -1;
            return s.First + v * s.Nu + u;
        }
    }
}
