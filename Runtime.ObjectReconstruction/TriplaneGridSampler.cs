#if HAS_AI_INFERENCE
using System;
using Unity.InferenceEngine;
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
        private int _numPlanes, _channels, _planeH, _planeW, _featureDim;

        internal TriplaneGridSampler(ComputeShader shader, int resolution)
        {
            _shader = shader;
            _resolution = resolution;
            _kernelGrid = _shader.FindKernel("SampleTriplane");
            _kernelPositions = _shader.FindKernel("SampleAtPositions");
        }

        internal void CacheSceneCodes(Tensor<float> sceneCodes)
        {
            var data = sceneCodes.DownloadToArray();
            var shape = sceneCodes.shape;
            _numPlanes = shape[1]; // 3
            _channels = shape[2];  // 40
            _planeH = shape[3];    // 64
            _planeW = shape[4];    // 64
            _featureDim = _numPlanes * _channels; // 120

            _sceneCodeBuffer?.Release();
            _sceneCodeBuffer = new ComputeBuffer(data.Length, sizeof(float));
            _sceneCodeBuffer.SetData(data);

            _shader.SetInt("_NumPlanes", _numPlanes);
            _shader.SetInt("_Channels", _channels);
            _shader.SetInt("_PlaneH", _planeH);
            _shader.SetInt("_PlaneW", _planeW);
            _shader.SetInt("_Resolution", _resolution);
        }

        internal int FeatureDim => _featureDim;
        internal int TotalGridPoints => _resolution * _resolution * _resolution;

        /// <summary>
        /// GPU dispatch for a chunk of uniform grid points (Pass 1 density).
        /// </summary>
        internal float[] SampleGridChunk(int startIdx, int count)
        {
            int featureCount = count * _featureDim;
            using var outputBuf = new ComputeBuffer(featureCount, sizeof(float));

            _shader.SetInt("_GridOffset", startIdx);
            _shader.SetInt("_TotalPoints", count);
            _shader.SetBuffer(_kernelGrid, "_SceneCodes", _sceneCodeBuffer);
            _shader.SetBuffer(_kernelGrid, "_OutputFeatures", outputBuf);
            _shader.Dispatch(_kernelGrid, (count + 63) / 64, 1, 1);

            var result = new float[featureCount];
            outputBuf.GetData(result);
            return result;
        }

        /// <summary>
        /// GPU dispatch for arbitrary 3D positions (Pass 2 vertex colors).
        /// </summary>
        internal float[] SampleFeaturesAtPositions(Vector3[] positions)
        {
            int count = positions.Length;
            int featureCount = count * _featureDim;

            var posData = new float[count * 3];
            for (int i = 0; i < count; i++)
            {
                posData[i * 3 + 0] = positions[i].x;
                posData[i * 3 + 1] = positions[i].y;
                posData[i * 3 + 2] = positions[i].z;
            }

            using var posBuf = new ComputeBuffer(count * 3, sizeof(float));
            posBuf.SetData(posData);

            using var outputBuf = new ComputeBuffer(featureCount, sizeof(float));

            _shader.SetInt("_NumPositions", count);
            _shader.SetBuffer(_kernelPositions, "_SceneCodes", _sceneCodeBuffer);
            _shader.SetBuffer(_kernelPositions, "_Positions", posBuf);
            _shader.SetBuffer(_kernelPositions, "_OutputFeatures", outputBuf);
            _shader.Dispatch(_kernelPositions, (count + 63) / 64, 1, 1);

            var result = new float[featureCount];
            outputBuf.GetData(result);
            return result;
        }

        public void Dispose()
        {
            _sceneCodeBuffer?.Release();
            _sceneCodeBuffer = null;
        }
    }
}
#endif
