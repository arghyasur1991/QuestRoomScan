using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Orchestrates the forward rasterization pipeline:
    /// ProjectSH → TileSort (Scatter + PrefixSum + BitonicSort) → Rasterize.
    /// All on GPU, no CPU readback.
    /// </summary>
    public class GSplatForwardPass : IDisposable
    {
        readonly ComputeShader _projectSH;
        readonly ComputeShader _tileSort;
        readonly ComputeShader _rasterize;

        readonly int _kProjectSHForward;
        readonly int _kScatterToBins;
        readonly int _kPrefixSum;
        readonly int _kBitonicSort;
        readonly int _kRasterizeForward;

        // Scratch buffers (shared across sectors, sized for worst case)
        GraphicsBuffer _xys, _depths, _conics, _numTilesHit, _colors, _aabb, _radii;
        GraphicsBuffer _scatterCounters, _overflowFlag;
        GraphicsBuffer _preallocBins;
        GraphicsBuffer _tileCounts, _tileOffsets;
        GraphicsBuffer _gaussianIdsOut, _packedXYOpac, _packedConic, _packedRGB, _tileBins;
        GraphicsBuffer _finalTs, _finalIndex;
        RenderTexture _outImg;

        int _maxPoints;
        int _imgW, _imgH;
        uint3 _tileBounds;
        int _numTiles;

        static readonly int ID_NumPoints = Shader.PropertyToID("_NumPoints");
        static readonly int ID_GlobScale = Shader.PropertyToID("_GlobScale");
        static readonly int ID_ClipThresh = Shader.PropertyToID("_ClipThresh");
        static readonly int ID_Degree = Shader.PropertyToID("_Degree");
        static readonly int ID_DegreesToUse = Shader.PropertyToID("_DegreesToUse");
        static readonly int ID_ImgSize = Shader.PropertyToID("_ImgSize");
        static readonly int ID_TileBounds = Shader.PropertyToID("_TileBounds");
        static readonly int ID_Intrinsics = Shader.PropertyToID("_Intrinsics");
        static readonly int ID_CamPos = Shader.PropertyToID("_CamPos");
        static readonly int ID_ViewMat = Shader.PropertyToID("_ViewMat");
        static readonly int ID_ProjMat = Shader.PropertyToID("_ProjMat");
        static readonly int ID_NumTiles = Shader.PropertyToID("_NumTiles");
        static readonly int ID_Background = Shader.PropertyToID("_Background");

        public RenderTexture OutputImage => _outImg;
        public GraphicsBuffer FinalTs => _finalTs;
        public GraphicsBuffer FinalIndex => _finalIndex;
        public GraphicsBuffer PackedXYOpac => _packedXYOpac;
        public GraphicsBuffer PackedConic => _packedConic;
        public GraphicsBuffer PackedRGB => _packedRGB;
        public GraphicsBuffer TileBins => _tileBins;
        public GraphicsBuffer GaussianIdsOut => _gaussianIdsOut;
        public GraphicsBuffer XYs => _xys;
        public GraphicsBuffer Conics => _conics;
        public GraphicsBuffer Depths => _depths;
        public GraphicsBuffer Radii => _radii;
        public GraphicsBuffer AABB => _aabb;
        public GraphicsBuffer Colors => _colors;

        public GSplatForwardPass(ComputeShader projectSH, ComputeShader tileSort,
                                 ComputeShader rasterize, int maxPoints, int imgW, int imgH)
        {
            _projectSH = projectSH;
            _tileSort = tileSort;
            _rasterize = rasterize;

            _kProjectSHForward = projectSH.FindKernel("ProjectSHForward");
            _kScatterToBins = tileSort.FindKernel("ScatterToBins");
            _kPrefixSum = tileSort.FindKernel("PrefixSumTileCounts");
            _kBitonicSort = tileSort.FindKernel("BitonicSortPerTile");
            _kRasterizeForward = rasterize.FindKernel("RasterizeForward");

            Resize(maxPoints, imgW, imgH);
        }

        public void Resize(int maxPoints, int imgW, int imgH)
        {
            if (_maxPoints == maxPoints && _imgW == imgW && _imgH == imgH)
                return;

            Dispose();

            _maxPoints = maxPoints;
            _imgW = imgW;
            _imgH = imgH;
            _tileBounds = new uint3(
                (uint)((imgW + 15) / 16),
                (uint)((imgH + 15) / 16),
                1
            );
            _numTiles = (int)(_tileBounds.x * _tileBounds.y);

            const GraphicsBuffer.Target s = GraphicsBuffer.Target.Structured;

            _xys = new GraphicsBuffer(s, maxPoints * 2, 4);
            _depths = new GraphicsBuffer(s, maxPoints, 4);
            _radii = new GraphicsBuffer(s, maxPoints, 4);
            _conics = new GraphicsBuffer(s, maxPoints * 3, 4);
            _numTilesHit = new GraphicsBuffer(s, maxPoints, 4);
            _colors = new GraphicsBuffer(s, maxPoints * 3, 4);
            _aabb = new GraphicsBuffer(s, maxPoints * 2, 4);

            _scatterCounters = new GraphicsBuffer(s, _numTiles, 4);
            _overflowFlag = new GraphicsBuffer(s, 1, 4);

            int maxIntersections = _numTiles * 1024;
            _preallocBins = new GraphicsBuffer(s, _numTiles * 1024, 8);

            _tileCounts = new GraphicsBuffer(s, _numTiles, 4);
            _tileOffsets = new GraphicsBuffer(s, _numTiles, 4);

            int estIntersections = Mathf.Min(maxPoints * 4, maxIntersections);
            _gaussianIdsOut = new GraphicsBuffer(s, estIntersections, 4);
            _packedXYOpac = new GraphicsBuffer(s, estIntersections * 3, 4);
            _packedConic = new GraphicsBuffer(s, estIntersections * 3, 4);
            _packedRGB = new GraphicsBuffer(s, estIntersections * 3, 4);
            _tileBins = new GraphicsBuffer(s, _numTiles * 2, 4);

            int numPixels = imgW * imgH;
            _finalTs = new GraphicsBuffer(s, numPixels, 4);
            _finalIndex = new GraphicsBuffer(s, numPixels, 4);

            _outImg = new RenderTexture(imgW, imgH, 0, GraphicsFormat.R16G16B16A16_SFloat)
            {
                enableRandomWrite = true,
                filterMode = FilterMode.Point
            };
            _outImg.Create();
        }

        /// <summary>
        /// Execute the full forward pass. All parameters stay on GPU.
        /// </summary>
        public void Execute(GSplatBuffers gaussians, Matrix4x4 viewMat, Matrix4x4 projMat,
                            float fx, float fy, float cx, float cy,
                            Vector3 camPos, int shDegree, int shDegreesToUse,
                            float globScale = 1f, float clipThresh = 0.01f,
                            Color? background = null)
        {
            int N = gaussians.CurrentCount;
            if (N == 0) return;

            Color bg = background ?? Color.black;

            // ---- 1. ProjectSH Forward ----
            _projectSH.SetInt(ID_NumPoints, N);
            _projectSH.SetFloat(ID_GlobScale, globScale);
            _projectSH.SetFloat(ID_ClipThresh, clipThresh);
            _projectSH.SetInt(ID_Degree, shDegree);
            _projectSH.SetInt(ID_DegreesToUse, shDegreesToUse);
            _projectSH.SetInts(ID_ImgSize, _imgW, _imgH);
            _projectSH.SetInts(ID_TileBounds, (int)_tileBounds.x, (int)_tileBounds.y, (int)_tileBounds.z);
            _projectSH.SetVector(ID_Intrinsics, new Vector4(fx, fy, cx, cy));
            _projectSH.SetVector(ID_CamPos, camPos);
            _projectSH.SetMatrix(ID_ViewMat, viewMat);
            _projectSH.SetMatrix(ID_ProjMat, projMat);

            _projectSH.SetBuffer(_kProjectSHForward, "_Means3D", gaussians.Means);
            _projectSH.SetBuffer(_kProjectSHForward, "_Scales", gaussians.Scales);
            _projectSH.SetBuffer(_kProjectSHForward, "_Quats", gaussians.Quats);
            _projectSH.SetBuffer(_kProjectSHForward, "_FeaturesDC", gaussians.FeaturesDC);
            _projectSH.SetBuffer(_kProjectSHForward, "_FeaturesRest", gaussians.FeaturesRest);
            _projectSH.SetBuffer(_kProjectSHForward, "_XYs", _xys);
            _projectSH.SetBuffer(_kProjectSHForward, "_Depths", _depths);
            _projectSH.SetBuffer(_kProjectSHForward, "_Radii", _radii);
            _projectSH.SetBuffer(_kProjectSHForward, "_Conics", _conics);
            _projectSH.SetBuffer(_kProjectSHForward, "_NumTilesHit", _numTilesHit);
            _projectSH.SetBuffer(_kProjectSHForward, "_Colors", _colors);
            _projectSH.SetBuffer(_kProjectSHForward, "_AABB", _aabb);

            _projectSH.Dispatch(_kProjectSHForward, CeilDiv(N, 256), 1, 1);

            // ---- 2. Tile Sort ----
            // Clear scatter counters + overflow flag
            ZeroBufGPU(_scatterCounters);
            ZeroBufGPU(_overflowFlag);

            // 2a. Scatter to bins
            _tileSort.SetInt(ID_NumPoints, N);
            _tileSort.SetInts(ID_TileBounds, (int)_tileBounds.x, (int)_tileBounds.y, (int)_tileBounds.z);
            _tileSort.SetBuffer(_kScatterToBins, "_XYs", _xys);
            _tileSort.SetBuffer(_kScatterToBins, "_Depths", _depths);
            _tileSort.SetBuffer(_kScatterToBins, "_Radii", _radii);
            _tileSort.SetBuffer(_kScatterToBins, "_AABB", _aabb);
            _tileSort.SetBuffer(_kScatterToBins, "_ScatterCounters", _scatterCounters);
            _tileSort.SetBuffer(_kScatterToBins, "_PreallocBins", _preallocBins);
            _tileSort.SetBuffer(_kScatterToBins, "_OverflowFlag", _overflowFlag);
            _tileSort.Dispatch(_kScatterToBins, CeilDiv(N, 256), 1, 1);

            // 2b. Prefix sum (single-threaded for small tile counts)
            _tileSort.SetInt(ID_NumTiles, _numTiles);
            _tileSort.SetBuffer(_kPrefixSum, "_TileCounts", _scatterCounters);
            _tileSort.SetBuffer(_kPrefixSum, "_TileOffsets", _tileOffsets);
            _tileSort.Dispatch(_kPrefixSum, 1, 1, 1);

            // 2c. Bitonic sort per tile
            _tileSort.SetInt(ID_NumTiles, _numTiles);
            _tileSort.SetBuffer(_kBitonicSort, "_SortTileCounts", _scatterCounters);
            _tileSort.SetBuffer(_kBitonicSort, "_SortTileOffsets", _tileOffsets);
            _tileSort.SetBuffer(_kBitonicSort, "_SortPreallocBins", _preallocBins);
            _tileSort.SetBuffer(_kBitonicSort, "_SortXYs", _xys);
            _tileSort.SetBuffer(_kBitonicSort, "_SortConics", _conics);
            _tileSort.SetBuffer(_kBitonicSort, "_SortColors", _colors);
            _tileSort.SetBuffer(_kBitonicSort, "_Opacities", gaussians.Opacities);
            _tileSort.SetBuffer(_kBitonicSort, "_GaussianIdsOut", _gaussianIdsOut);
            _tileSort.SetBuffer(_kBitonicSort, "_PackedXYOpac", _packedXYOpac);
            _tileSort.SetBuffer(_kBitonicSort, "_PackedConic", _packedConic);
            _tileSort.SetBuffer(_kBitonicSort, "_PackedRGB", _packedRGB);
            _tileSort.SetBuffer(_kBitonicSort, "_TileBins", _tileBins);
            _tileSort.Dispatch(_kBitonicSort, _numTiles, 1, 1);

            // ---- 3. Rasterize Forward ----
            _rasterize.SetInts(ID_TileBounds, (int)_tileBounds.x, (int)_tileBounds.y, (int)_tileBounds.z);
            _rasterize.SetInts(ID_ImgSize, _imgW, _imgH);
            _rasterize.SetVector(ID_Background, new Vector4(bg.r, bg.g, bg.b, 0));
            _rasterize.SetBuffer(_kRasterizeForward, "_TileBins", _tileBins);
            _rasterize.SetBuffer(_kRasterizeForward, "_PackedXYOpac", _packedXYOpac);
            _rasterize.SetBuffer(_kRasterizeForward, "_PackedConic", _packedConic);
            _rasterize.SetBuffer(_kRasterizeForward, "_PackedRGB", _packedRGB);
            _rasterize.SetBuffer(_kRasterizeForward, "_FinalTs", _finalTs);
            _rasterize.SetBuffer(_kRasterizeForward, "_FinalIndex", _finalIndex);
            _rasterize.SetTexture(_kRasterizeForward, "_OutImg", _outImg);

            int rGroupsX = CeilDiv(_imgW, 8);
            int rGroupsY = CeilDiv(_imgH, 8);
            _rasterize.Dispatch(_kRasterizeForward, rGroupsX, rGroupsY, 1);
        }

        public void Dispose()
        {
            _xys?.Release(); _depths?.Release(); _radii?.Release();
            _conics?.Release(); _numTilesHit?.Release(); _colors?.Release(); _aabb?.Release();
            _scatterCounters?.Release(); _overflowFlag?.Release();
            _preallocBins?.Release();
            _tileCounts?.Release(); _tileOffsets?.Release();
            _gaussianIdsOut?.Release(); _packedXYOpac?.Release();
            _packedConic?.Release(); _packedRGB?.Release(); _tileBins?.Release();
            _finalTs?.Release(); _finalIndex?.Release();
            if (_outImg) UnityEngine.Object.Destroy(_outImg);

            _xys = null; _depths = null; _radii = null;
            _conics = null; _numTilesHit = null; _colors = null; _aabb = null;
            _maxPoints = 0;
        }

        static int CeilDiv(int a, int b) => (a + b - 1) / b;

        static void ZeroBufGPU(GraphicsBuffer buf)
        {
            var z = new byte[buf.count * buf.stride];
            buf.SetData(z);
        }
    }
}
