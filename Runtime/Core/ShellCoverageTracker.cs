using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Decodes the shell-coverage readback into per-cell state, clusters
    /// uncovered cells into gaps, and decides which gaps the auto-fill may
    /// close. Runs on the main thread at the coverage tick (~1 Hz) with
    /// preallocated storage — no allocations after construction.
    /// </summary>
    internal sealed class ShellCoverageTracker
    {
        const int MaxClusters = 256;
        const int MaxGaps = 64;
        const int FillMinAgeTicks = 2;
        const int FillMinNeighbors = 4;

        struct Cluster
        {
            public int Cells;
            public Vector3 Sum;
            public int Surface;
            public ShellSurfaceKind Kind;
            public int UMin, UMax, VMin, VMax;
            public bool TouchesExcluded;
            public int NbCount;
            public float NbSum, NbMin, NbMax;
            public int MinAge;
            public Vector3 BoxMin, BoxMax;
        }

        readonly ShellCellSet _cells;
        readonly int[] _clusterId = new int[ShellCellSet.MaxCells];
        readonly int[] _stack = new int[ShellCellSet.MaxCells];
        readonly byte[] _uncoveredTicks = new byte[ShellCellSet.MaxCells];
        readonly Cluster[] _clusters = new Cluster[MaxClusters];
        readonly int[] _order = new int[MaxClusters];
        readonly ShellGap[] _gaps = new ShellGap[MaxGaps];
        readonly VolumeIntegrator.ShellFillRequest[] _fills = new VolumeIntegrator.ShellFillRequest[32];
        int _clusterCount;
        int _gapCount;

        public bool Available { get; private set; }
        /// <summary>Covered / (uploaded − observed-empty). 0 until the first readback of a scan.</summary>
        public float Coverage { get; private set; }
        public int Uploaded { get; private set; }
        public int Covered { get; private set; }
        /// <summary>Cells the march found to be observed air this tick; not in the denominator.</summary>
        public int Empty { get; private set; }
        public int Excluded => _cells.ExcludedCount;
        public int GapCount { get; private set; }
        public ShellGap LargestGap { get; private set; }
        public int FillsApplied { get; private set; }

        public ShellCoverageTracker(ShellCellSet cells)
        {
            _cells = cells;
        }

        /// <summary>
        /// Call after the cell set is rebuilt (scan start, anchors changed).
        /// Keeps the last coverage figure when the tracker was already live,
        /// so a mid-scan anchors-changed rebuild does not read as 0 % for a
        /// tick; <see cref="Disable"/> between scans is what zeroes it.
        /// </summary>
        public void Reset()
        {
            bool wasLive = Available;
            Available = _cells.UploadCount > 0;
            if (!wasLive)
            {
                Coverage = 0f;
                Covered = 0;
                Empty = 0;
                FillsApplied = 0;
            }
            Uploaded = _cells.UploadCount;
            GapCount = 0;
            _gapCount = 0;
            LargestGap = default;
            System.Array.Clear(_uncoveredTicks, 0, _cells.CellCount);
        }

        public void Disable()
        {
            Available = false;
            Coverage = 0f;
            Covered = 0;
            Empty = 0;
            GapCount = 0;
            _gapCount = 0;
            LargestGap = default;
        }

        public int CopyGaps(List<ShellGap> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (!Available) return 0;
            int n = Mathf.Min(_gapCount, 8);
            for (int i = 0; i < n; i++) dest.Add(_gaps[i]);
            return n;
        }

        /// <summary>
        /// Decode one readback, cluster, and (optionally) dispatch fills.
        /// <paramref name="result"/> holds one word per uploaded cell.
        /// </summary>
        public void Update(uint[] result, int count, VolumeIntegrator vi, float voxelSize)
        {
            if (!Available || count <= 0) return;
            count = Mathf.Min(count, _cells.UploadCount);

            int covered = 0;
            int empty = 0;
            for (int g = 0; g < count; g++)
            {
                int c = _cells.CellOfGpu[g];
                uint w = result[g];
                if ((w & 1u) != 0)
                {
                    covered++;
                    _cells.State[c] = ShellCellSet.StateCovered;
                    int step = (int)((w >> 8) & 0xFFu);
                    _cells.HitOffset[c] = _cells.MarchStart[c] + step * voxelSize;
                    _uncoveredTicks[c] = 0;
                }
                else if ((w & 2u) != 0)
                {
                    empty++;
                    _cells.State[c] = ShellCellSet.StateEmpty;
                    _cells.HitOffset[c] = float.NaN;
                    _uncoveredTicks[c] = 0;
                }
                else
                {
                    _cells.State[c] = ShellCellSet.StateUncovered;
                    _cells.HitOffset[c] = float.NaN;
                    if (_uncoveredTicks[c] < 255) _uncoveredTicks[c]++;
                }
            }
            Covered = covered;
            Empty = empty;
            Uploaded = count;
            int required = count - empty;
            Coverage = required > 0 ? (float)covered / required : 1f;

            BuildClusters();
            PublishGaps();

            if (vi != null && (vi.AutoFillShellGaps || vi.CloseFurnitureHoles))
                FillsApplied += vi.ApplyShellFills(_fills, CollectFills(vi));
        }

        void BuildClusters()
        {
            int n = _cells.CellCount;
            for (int i = 0; i < n; i++) _clusterId[i] = -1;
            _clusterCount = 0;

            for (int seed = 0; seed < n && _clusterCount < MaxClusters; seed++)
            {
                if (_cells.State[seed] != ShellCellSet.StateUncovered || _clusterId[seed] >= 0)
                    continue;

                int id = _clusterCount++;
                var cl = new Cluster
                {
                    Surface = _cells.SurfaceIndex[seed],
                    Kind = _cells.Surfaces[_cells.SurfaceIndex[seed]].Kind,
                    UMin = int.MaxValue, UMax = int.MinValue,
                    VMin = int.MaxValue, VMax = int.MinValue,
                    NbMin = float.MaxValue, NbMax = float.MinValue,
                    MinAge = int.MaxValue,
                    BoxMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue),
                    BoxMax = new Vector3(float.MinValue, float.MinValue, float.MinValue)
                };

                int sp = 0;
                _stack[sp++] = seed;
                _clusterId[seed] = id;
                while (sp > 0)
                {
                    int c = _stack[--sp];
                    cl.Cells++;
                    cl.Sum += _cells.Pos[c];
                    cl.BoxMin = Vector3.Min(cl.BoxMin, _cells.Pos[c]);
                    cl.BoxMax = Vector3.Max(cl.BoxMax, _cells.Pos[c]);
                    int u = _cells.U[c], v = _cells.V[c];
                    if (u < cl.UMin) cl.UMin = u;
                    if (u > cl.UMax) cl.UMax = u;
                    if (v < cl.VMin) cl.VMin = v;
                    if (v > cl.VMax) cl.VMax = v;
                    int age = _uncoveredTicks[c];
                    if (age < cl.MinAge) cl.MinAge = age;

                    Visit(cl.Surface, u - 1, v, id, ref cl, ref sp);
                    Visit(cl.Surface, u + 1, v, id, ref cl, ref sp);
                    Visit(cl.Surface, u, v - 1, id, ref cl, ref sp);
                    Visit(cl.Surface, u, v + 1, id, ref cl, ref sp);
                }
                _clusters[id] = cl;
            }
        }

        void Visit(int surface, int u, int v, int id, ref Cluster cl, ref int sp)
        {
            int nb = _cells.CellIndex(surface, u, v);
            if (nb < 0) return;
            byte st = _cells.State[nb];
            if (st == ShellCellSet.StateExcluded || st == ShellCellSet.StateEmpty)
            {
                // A jamb, a window, or observed air: never extend a plane over it.
                cl.TouchesExcluded = true;
                return;
            }
            if (st == ShellCellSet.StateCovered)
            {
                float h = _cells.HitOffset[nb];
                if (!float.IsNaN(h))
                {
                    cl.NbCount++;
                    cl.NbSum += h;
                    if (h < cl.NbMin) cl.NbMin = h;
                    if (h > cl.NbMax) cl.NbMax = h;
                }
                return;
            }
            if (_clusterId[nb] >= 0) return;
            _clusterId[nb] = id;
            _stack[sp++] = nb;
        }

        void PublishGaps()
        {
            for (int i = 0; i < _clusterCount; i++) _order[i] = i;
            // Insertion sort by Cells desc; cluster counts are small.
            for (int i = 1; i < _clusterCount; i++)
            {
                int key = _order[i];
                int j = i - 1;
                while (j >= 0 && _clusters[_order[j]].Cells < _clusters[key].Cells)
                {
                    _order[j + 1] = _order[j];
                    j--;
                }
                _order[j + 1] = key;
            }

            float cellArea = _cells.CellSize * _cells.CellSize;
            int gaps = 0;
            _gapCount = 0;
            for (int i = 0; i < _clusterCount; i++)
            {
                var cl = _clusters[_order[i]];
                if (cl.Cells < 2) continue;
                gaps++;
                if (_gapCount < MaxGaps)
                {
                    var s = _cells.Surfaces[cl.Surface];
                    _gaps[_gapCount++] = new ShellGap(
                        cl.Sum / cl.Cells, s.Normal, cl.Cells, cl.Cells * cellArea, cl.Kind);
                }
            }
            GapCount = gaps;
            LargestGap = _gapCount > 0 ? _gaps[0] : default;
        }

        int CollectFills(VolumeIntegrator vi)
        {
            int maxFills = Mathf.Min(vi.FillMaxPerTick, _fills.Length);
            int maxCells = vi.FillMaxCells;
            float spreadMax = vi.FillNeighborSpreadMax;
            float half = _cells.CellSize * 0.5f;
            int n = 0;

            for (int i = 0; i < _clusterCount && n < maxFills; i++)
            {
                var cl = _clusters[i];
                if (cl.Cells > maxCells || cl.MinAge < FillMinAgeTicks) continue;

                if (cl.Kind == ShellSurfaceKind.Furniture)
                {
                    // The real surface sits somewhere along the march (a couch
                    // seat is well below its box top), so the close box is
                    // placed where the covered neighbours found it. No covered
                    // neighbour means nothing to close against.
                    if (!vi.CloseFurnitureHoles || cl.NbCount < FillMinNeighbors) continue;
                    if (cl.NbMax - cl.NbMin > spreadMax * 2f) continue;
                    Vector3 shift = _cells.Surfaces[cl.Surface].Normal * (cl.NbSum / cl.NbCount);
                    var pad = Vector3.one * 0.15f;
                    _fills[n++] = new VolumeIntegrator.ShellFillRequest
                    {
                        Close = true,
                        BoxMin = Vector3.Min(cl.BoxMin + shift, cl.BoxMax + shift) - pad,
                        BoxMax = Vector3.Max(cl.BoxMin + shift, cl.BoxMax + shift) + pad
                    };
                    continue;
                }

                if (!vi.AutoFillShellGaps) continue;
                if (cl.TouchesExcluded || cl.NbCount < FillMinNeighbors) continue;
                if (cl.NbMax - cl.NbMin > spreadMax) continue;

                var s = _cells.Surfaces[cl.Surface];
                float nbMean = cl.NbSum / cl.NbCount;
                int cMin = _cells.CellIndex(cl.Surface, cl.UMin, cl.VMin);
                int cMax = _cells.CellIndex(cl.Surface, cl.UMax, cl.VMax);
                if (cMin < 0 || cMax < 0) continue;
                Vector3 center = (_cells.Pos[cMin] + _cells.Pos[cMax]) * 0.5f + s.Normal * nbMean;
                _fills[n++] = new VolumeIntegrator.ShellFillRequest
                {
                    Close = false,
                    Center = center,
                    Inward = s.Normal,
                    Axis = s.AxisU,
                    Bitangent = s.AxisV,
                    HalfW = (cl.UMax - cl.UMin + 1) * half,
                    HalfH = (cl.VMax - cl.VMin + 1) * half
                };
            }
            return n;
        }
    }
}
