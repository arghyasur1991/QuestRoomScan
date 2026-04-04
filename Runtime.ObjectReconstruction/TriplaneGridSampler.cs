#if HAS_AI_INFERENCE
using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Samples triplane features at 3D query points via a GPU compute shader.
    /// For each point in a resolution^3 grid, projects onto XY, XZ, YZ planes
    /// and bilinearly samples features, concatenating to a (N, 3*C) output.
    /// </summary>
    internal sealed class TriplaneGridSampler : IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly int _resolution;
        private readonly int _kernelSample;
        private ComputeBuffer _queryPoints;
        private ComputeBuffer _outputFeatures;

        internal TriplaneGridSampler(ComputeShader shader, int resolution)
        {
            _shader = shader;
            _resolution = resolution;

            if (_shader != null)
                _kernelSample = _shader.FindKernel("SampleTriplane");
        }

        /// <summary>
        /// Samples all three planes of scene_codes (1, 3, C, H, W) at a uniform grid.
        /// Returns a CPU tensor of shape (resolution^3, 3*C).
        /// </summary>
        internal Tensor<float> SampleFeatures(Tensor<float> sceneCodes)
        {
            var sceneData = sceneCodes.DownloadToArray();
            var shape = sceneCodes.shape;

            int numPlanes = shape[1]; // 3
            int channels = shape[2];  // 40
            int planeH = shape[3];    // 64
            int planeW = shape[4];    // 64
            int featureDim = numPlanes * channels; // 120

            int totalPoints = _resolution * _resolution * _resolution;
            var features = new float[totalPoints * featureDim];

            float step = 1f / _resolution;
            float offset = step * 0.5f - 0.5f;

            for (int iz = 0; iz < _resolution; iz++)
            {
                float z = offset + iz * step;
                for (int iy = 0; iy < _resolution; iy++)
                {
                    float y = offset + iy * step;
                    for (int ix = 0; ix < _resolution; ix++)
                    {
                        float x = offset + ix * step;
                        int ptIdx = (iz * _resolution * _resolution + iy * _resolution + ix) * featureDim;

                        float[][] coords = {
                            new[] { x, y },  // XY plane
                            new[] { x, z },  // XZ plane
                            new[] { y, z },  // YZ plane
                        };

                        for (int p = 0; p < numPlanes; p++)
                        {
                            float u = (coords[p][0] + 0.5f) * (planeW - 1);
                            float v = (coords[p][1] + 0.5f) * (planeH - 1);

                            int u0 = Mathf.Clamp(Mathf.FloorToInt(u), 0, planeW - 1);
                            int v0 = Mathf.Clamp(Mathf.FloorToInt(v), 0, planeH - 1);
                            int u1 = Mathf.Min(u0 + 1, planeW - 1);
                            int v1 = Mathf.Min(v0 + 1, planeH - 1);
                            float fu = u - u0;
                            float fv = v - v0;

                            int planeOffset = p * channels * planeH * planeW;
                            for (int c = 0; c < channels; c++)
                            {
                                int chOffset = planeOffset + c * planeH * planeW;
                                float val00 = sceneData[chOffset + v0 * planeW + u0];
                                float val01 = sceneData[chOffset + v1 * planeW + u0];
                                float val10 = sceneData[chOffset + v0 * planeW + u1];
                                float val11 = sceneData[chOffset + v1 * planeW + u1];

                                float val = val00 * (1 - fu) * (1 - fv) +
                                            val10 * fu * (1 - fv) +
                                            val01 * (1 - fu) * fv +
                                            val11 * fu * fv;

                                features[ptIdx + p * channels + c] = val;
                            }
                        }
                    }
                }
            }

            var tensor = new Tensor<float>(new TensorShape(totalPoints, featureDim));
            tensor.Upload(features);
            return tensor;
        }

        public void Dispose()
        {
            _queryPoints?.Release();
            _outputFeatures?.Release();
        }
    }
}
#endif
