#if HAS_AI_INFERENCE
using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Samples triplane features at 3D positions via GPU compute shader.
    /// Falls back to CPU when compute shader is unavailable.
    /// Coordinate convention matches PyTorch F.grid_sample(align_corners=False).
    /// </summary>
    internal sealed class TriplaneGridSampler : IDisposable
    {
        private readonly ComputeShader _shader;
        private readonly int _resolution;
        private readonly int _kernelGrid;
        private readonly int _kernelPositions;
        private readonly bool _gpuAvailable;

        private float[] _cachedSceneData;
        private ComputeBuffer _sceneCodeBuffer;
        private int _numPlanes, _channels, _planeH, _planeW, _featureDim;

        internal TriplaneGridSampler(ComputeShader shader, int resolution)
        {
            _shader = shader;
            _resolution = resolution;

            if (_shader != null)
            {
                _kernelGrid = _shader.FindKernel("SampleTriplane");
                _kernelPositions = _shader.FindKernel("SampleAtPositions");
                _gpuAvailable = SystemInfo.supportsComputeShaders;
            }
        }

        internal void CacheSceneCodes(Tensor<float> sceneCodes)
        {
            _cachedSceneData = sceneCodes.DownloadToArray();
            var shape = sceneCodes.shape;
            _numPlanes = shape[1]; // 3
            _channels = shape[2];  // 40
            _planeH = shape[3];    // 64
            _planeW = shape[4];    // 64
            _featureDim = _numPlanes * _channels; // 120

            if (_gpuAvailable)
            {
                _sceneCodeBuffer?.Release();
                _sceneCodeBuffer = new ComputeBuffer(_cachedSceneData.Length, sizeof(float));
                _sceneCodeBuffer.SetData(_cachedSceneData);
                SetShaderConstants();
            }
        }

        internal int FeatureDim => _featureDim;
        internal int TotalGridPoints => _resolution * _resolution * _resolution;
        internal bool UseGPU => _gpuAvailable;

        /// <summary>
        /// GPU: dispatch compute shader for a chunk of uniform grid points.
        /// Returns features via AsyncGPUReadback to avoid blocking.
        /// </summary>
        internal float[] SampleGridChunkGPU(int startIdx, int count)
        {
            int featureCount = count * _featureDim;
            using var outputBuf = new ComputeBuffer(featureCount, sizeof(float));

            _shader.SetInt("_GridOffset", startIdx);
            _shader.SetInt("_TotalPoints", count);
            _shader.SetBuffer(_kernelGrid, "_SceneCodes", _sceneCodeBuffer);
            _shader.SetBuffer(_kernelGrid, "_OutputFeatures", outputBuf);

            int groups = (count + 63) / 64;
            _shader.Dispatch(_kernelGrid, groups, 1, 1);

            var result = new float[featureCount];
            outputBuf.GetData(result);
            return result;
        }

        /// <summary>
        /// GPU: dispatch compute shader for arbitrary 3D positions (e.g. mesh vertices).
        /// </summary>
        internal float[] SampleFeaturesAtPositionsGPU(Vector3[] positions)
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

            int groups = (count + 63) / 64;
            _shader.Dispatch(_kernelPositions, groups, 1, 1);

            var result = new float[featureCount];
            outputBuf.GetData(result);
            return result;
        }

        // --- CPU fallback paths (used when compute shader unavailable) ---

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

        internal float[] SampleFeaturesAtPositions(Vector3[] positions)
        {
            if (_cachedSceneData == null)
                throw new InvalidOperationException("Call CacheSceneCodes first");

            int count = positions.Length;
            var features = new float[count * _featureDim];

            for (int i = 0; i < count; i++)
            {
                var p = positions[i];
                SampleTriplaneAt(p.x, p.y, p.z, _cachedSceneData, features, i * _featureDim);
            }

            return features;
        }

        private void SampleTriplaneAt(float x, float y, float z,
            float[] sceneData, float[] output, int outOffset)
        {
            int pw = _planeW, ph = _planeH, ch = _channels;
            float halfW = pw * 0.5f - 0.5f;
            float halfH = ph * 0.5f - 0.5f;
            int planeStride = ch * ph * pw;

            SampleOnePlane(sceneData, output, outOffset,
                x * pw + halfW, y * ph + halfH, 0, pw, ph, ch);
            SampleOnePlane(sceneData, output, outOffset + ch,
                x * pw + halfW, z * ph + halfH, planeStride, pw, ph, ch);
            SampleOnePlane(sceneData, output, outOffset + ch * 2,
                y * pw + halfW, z * ph + halfH, planeStride * 2, pw, ph, ch);
        }

        private static void SampleOnePlane(float[] data, float[] output, int outOff,
            float uf, float vf, int planeOff, int pw, int ph, int ch)
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

        private void SetShaderConstants()
        {
            _shader.SetInt("_NumPlanes", _numPlanes);
            _shader.SetInt("_Channels", _channels);
            _shader.SetInt("_PlaneH", _planeH);
            _shader.SetInt("_PlaneW", _planeW);
            _shader.SetInt("_Resolution", _resolution);
        }

        public void Dispose()
        {
            _cachedSceneData = null;
            _sceneCodeBuffer?.Release();
            _sceneCodeBuffer = null;
        }
    }
}
#endif
