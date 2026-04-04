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
        /// Zero heap allocations per call — all work on stack.
        /// </summary>
        private void SampleTriplaneAt(float x, float y, float z,
            float[] sceneData, float[] output, int outOffset)
        {
            int pw = _planeW, ph = _planeH, ch = _channels;
            float halfW = pw * 0.5f - 0.5f;
            float halfH = ph * 0.5f - 0.5f;
            int chPh = ch * ph;
            int planeStride = chPh * pw;

            // Unrolled 3 planes: XY(x,y), XZ(x,z), YZ(y,z) — no heap allocation
            SampleOnePlane(sceneData, output, outOffset,
                x * pw + halfW, y * ph + halfH, 0, pw, ph, ch, planeStride);
            SampleOnePlane(sceneData, output, outOffset + ch,
                x * pw + halfW, z * ph + halfH, planeStride, pw, ph, ch, planeStride);
            SampleOnePlane(sceneData, output, outOffset + ch * 2,
                y * pw + halfW, z * ph + halfH, planeStride * 2, pw, ph, ch, planeStride);
        }

        private static void SampleOnePlane(float[] data, float[] output, int outOff,
            float uf, float vf, int planeOff, int pw, int ph, int ch, int planeStride)
        {
            int u0 = (int)uf;
            int v0 = (int)vf;
            if (u0 < 0) u0 = 0; else if (u0 >= pw) u0 = pw - 1;
            if (v0 < 0) v0 = 0; else if (v0 >= ph) v0 = ph - 1;
            int u1 = u0 < pw - 1 ? u0 + 1 : u0;
            int v1 = v0 < ph - 1 ? v0 + 1 : v0;

            float fu = uf - u0;
            float fv = vf - v0;
            if (fu < 0f) fu = 0f; else if (fu > 1f) fu = 1f;
            if (fv < 0f) fv = 0f; else if (fv > 1f) fv = 1f;

            float w00 = (1f - fu) * (1f - fv);
            float w10 = fu * (1f - fv);
            float w01 = (1f - fu) * fv;
            float w11 = fu * fv;

            int row0 = v0 * pw;
            int row1 = v1 * pw;

            for (int c = 0; c < ch; c++)
            {
                int chOff = planeOff + c * ph * pw;
                output[outOff + c] =
                    data[chOff + row0 + u0] * w00 +
                    data[chOff + row0 + u1] * w10 +
                    data[chOff + row1 + u0] * w01 +
                    data[chOff + row1 + u1] * w11;
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
