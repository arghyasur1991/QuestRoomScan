#if HAS_AI_INFERENCE
using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Samples triplane features at 3D positions via bilinear interpolation.
    /// Coordinate convention matches PyTorch F.grid_sample(align_corners=False):
    /// positions in [-0.5, 0.5] map to pixel edges [-0.5, N-0.5].
    /// </summary>
    internal sealed class TriplaneGridSampler : IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly int _resolution;
        private readonly int _kernelSample;
        private ComputeBuffer _queryPoints;
        private ComputeBuffer _outputFeatures;

        private float[] _cachedSceneData;
        private int _numPlanes, _channels, _planeH, _planeW, _featureDim;

        internal TriplaneGridSampler(ComputeShader shader, int resolution)
        {
            _shader = shader;
            _resolution = resolution;

            if (_shader != null)
                _kernelSample = _shader.FindKernel("SampleTriplane");
        }

        /// <summary>
        /// Cache scene codes for multiple sampling passes (grid + vertex color re-query).
        /// </summary>
        internal void CacheSceneCodes(Tensor<float> sceneCodes)
        {
            _cachedSceneData = sceneCodes.DownloadToArray();
            var shape = sceneCodes.shape;
            _numPlanes = shape[1]; // 3
            _channels = shape[2];  // 40
            _planeH = shape[3];    // 64
            _planeW = shape[4];    // 64
            _featureDim = _numPlanes * _channels; // 120
        }

        internal int FeatureDim => _featureDim;
        internal int TotalGridPoints => _resolution * _resolution * _resolution;

        /// <summary>
        /// Samples a chunk of grid points [startIdx, startIdx+count) and returns
        /// a float array of (count * featureDim). Grid uses edge-to-edge positions
        /// matching linspace(-0.5, 0.5, res). Only allocates memory for the chunk.
        /// </summary>
        internal float[] SampleGridChunk(int startIdx, int count)
        {
            if (_cachedSceneData == null)
                throw new InvalidOperationException("Call CacheSceneCodes first");

            int res = _resolution;
            float invResM1 = 1f / (res - 1);
            var features = new float[count * _featureDim];

            for (int i = 0; i < count; i++)
            {
                int flatIdx = startIdx + i;
                int ix = flatIdx % res;
                int iy = (flatIdx / res) % res;
                int iz = flatIdx / (res * res);

                float x = ix * invResM1 - 0.5f;
                float y = iy * invResM1 - 0.5f;
                float z = iz * invResM1 - 0.5f;

                SampleTriplaneAt(x, y, z, _cachedSceneData, features, i * _featureDim);
            }

            return features;
        }

        /// <summary>
        /// Samples triplane features at arbitrary 3D positions (e.g. mesh vertices).
        /// Positions should be in [-0.5, 0.5] coordinate space.
        /// Returns float array of shape (numPositions * featureDim).
        /// </summary>
        internal float[] SampleFeaturesAtPositions(Vector3[] positions)
        {
            if (_cachedSceneData == null)
                throw new System.InvalidOperationException("Call SampleFeatures first to cache scene codes");

            int count = positions.Length;
            var features = new float[count * _featureDim];

            for (int i = 0; i < count; i++)
            {
                var p = positions[i];
                SampleTriplaneAt(p.x, p.y, p.z, _cachedSceneData, features, i * _featureDim);
            }

            return features;
        }

        /// <summary>
        /// Bilinear sampling matching F.grid_sample(align_corners=False).
        /// Position in [-0.5, 0.5] maps to pixel position = pos*N + N/2 - 0.5.
        /// </summary>
        private void SampleTriplaneAt(float x, float y, float z,
            float[] sceneData, float[] output, int outOffset)
        {
            float u0f, u1f, v0f, v1f;
            int u0, u1, v0, v1;
            float fu, fv;

            // XY, XZ, YZ plane coordinates
            float[] uCoords = { x, x, y };
            float[] vCoords = { y, z, z };

            for (int p = 0; p < _numPlanes; p++)
            {
                float uf = uCoords[p] * _planeW + _planeW * 0.5f - 0.5f;
                float vf = vCoords[p] * _planeH + _planeH * 0.5f - 0.5f;

                u0 = Mathf.Clamp(Mathf.FloorToInt(uf), 0, _planeW - 1);
                v0 = Mathf.Clamp(Mathf.FloorToInt(vf), 0, _planeH - 1);
                u1 = Mathf.Min(u0 + 1, _planeW - 1);
                v1 = Mathf.Min(v0 + 1, _planeH - 1);
                fu = Mathf.Clamp01(uf - u0);
                fv = Mathf.Clamp01(vf - v0);

                int planeOffset = p * _channels * _planeH * _planeW;
                for (int c = 0; c < _channels; c++)
                {
                    int chOffset = planeOffset + c * _planeH * _planeW;
                    float val00 = sceneData[chOffset + v0 * _planeW + u0];
                    float val01 = sceneData[chOffset + v1 * _planeW + u0];
                    float val10 = sceneData[chOffset + v0 * _planeW + u1];
                    float val11 = sceneData[chOffset + v1 * _planeW + u1];

                    output[outOffset + p * _channels + c] =
                        val00 * (1 - fu) * (1 - fv) +
                        val10 * fu * (1 - fv) +
                        val01 * (1 - fu) * fv +
                        val11 * fu * fv;
                }
            }
        }

        public void Dispose()
        {
            _cachedSceneData = null;
            _queryPoints?.Release();
            _outputFeatures?.Release();
        }
    }
}
#endif
