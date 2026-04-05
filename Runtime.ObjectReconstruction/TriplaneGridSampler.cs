#if HAS_ONNXRUNTIME
using System;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Samples triplane features at 3D positions via GPU compute shader.
    /// Two kernels: SampleTriplane (uniform grid) and SampleAtPositions (arbitrary xyz).
    /// Coordinate convention matches PyTorch F.grid_sample(align_corners=False).
    /// </summary>
    internal sealed class TriplaneGridSampler : IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly int _resolution;
        private readonly int _kernelGrid;
        private readonly int _kernelPositions;

        private ComputeBuffer _sceneCodeBuffer;
        private bool _ownsSceneCodeBuffer;
        private int _numPlanes, _channels, _planeH, _planeW, _featureDim;

        internal TriplaneGridSampler(ComputeShader shader, int resolution)
        {
            _shader = shader;
            _resolution = resolution;
            _kernelGrid = _shader.FindKernel("SampleTriplane");
            _kernelPositions = _shader.FindKernel("SampleAtPositions");
        }

        /// <summary>
        /// Copies scene codes from a flat float[] to an owned GPU buffer.
        /// The source array can be discarded after this call.
        /// </summary>
        internal void CacheSceneCodesGPU(float[] data, int numPlanes, int channels, int planeH, int planeW)
        {
            _numPlanes = numPlanes;
            _channels = channels;
            _planeH = planeH;
            _planeW = planeW;
            _featureDim = _numPlanes * _channels;

            if (_ownsSceneCodeBuffer)
                _sceneCodeBuffer?.Release();

            _sceneCodeBuffer = new ComputeBuffer(data.Length, sizeof(float));
            _sceneCodeBuffer.SetData(data);
            _ownsSceneCodeBuffer = true;

            _shader.SetInt("_NumPlanes", _numPlanes);
            _shader.SetInt("_Channels", _channels);
            _shader.SetInt("_PlaneH", _planeH);
            _shader.SetInt("_PlaneW", _planeW);
            _shader.SetInt("_Resolution", _resolution);
        }

        internal int FeatureDim => _featureDim;
        internal int TotalGridPoints => _resolution * _resolution * _resolution;

        /// <summary>
        /// GPU-only grid sampling. Dispatches the compute shader to write features
        /// directly into the provided output buffer. No CPU round-trip.
        /// </summary>
        internal void SampleGridChunkGPU(int startIdx, int count, ComputeBuffer outputBuf)
        {
            _shader.SetInt("_GridOffset", startIdx);
            _shader.SetInt("_TotalPoints", count);
            _shader.SetBuffer(_kernelGrid, "_SceneCodes", _sceneCodeBuffer);
            _shader.SetBuffer(_kernelGrid, "_OutputFeatures", outputBuf);
            _shader.Dispatch(_kernelGrid, (count + 63) / 64, 1, 1);
        }

        /// <summary>
        /// GPU-only position sampling. Reads positions from the provided buffer
        /// (starting at positionOffset) and writes features into the output buffer.
        /// </summary>
        internal void SampleAtPositionsGPU(
            ComputeBuffer positionsBuf, int positionOffset, int count, ComputeBuffer outputBuf)
        {
            _shader.SetInt("_NumPositions", count);
            _shader.SetInt("_PositionOffset", positionOffset);
            _shader.SetBuffer(_kernelPositions, "_SceneCodes", _sceneCodeBuffer);
            _shader.SetBuffer(_kernelPositions, "_Positions", positionsBuf);
            _shader.SetBuffer(_kernelPositions, "_OutputFeatures", outputBuf);
            _shader.Dispatch(_kernelPositions, (count + 63) / 64, 1, 1);
        }

        public void Dispose()
        {
            if (_ownsSceneCodeBuffer)
                _sceneCodeBuffer?.Release();
            _sceneCodeBuffer = null;
        }
    }
}
#endif
