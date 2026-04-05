#if HAS_ONNXRUNTIME
using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Wraps the NeRF decoder MLP via ONNX Runtime. Optimized for the hot path:
    /// uses RunPreallocated with reusable input/output float[] buffers to achieve
    /// zero allocation per chunk after initial warmup.
    /// </summary>
    internal sealed class OrtDecoderModel : OrtModelBase
    {
        private const string ModelFileName = "ObjectReconstruction/nerf_decoder.onnx";
        private const int FeatureDim = 120;
        private const int OutputDim = 4;

        private float[] _inputBuffer;
        private float[] _outputBuffer;
        private int _lastChunkSize = -1;
        private string _outputName;

        internal async Task LoadAsync(
            ExecutionProvider ep, bool mobileOptimized, CancellationToken ct)
        {
            await LoadSessionAsync(ModelFileName, ep, mobileOptimized, ct);
            _outputName = _session.OutputMetadata.Keys.GetEnumerator().Current;
            foreach (var key in _session.OutputMetadata.Keys)
            {
                _outputName = key;
                break;
            }
        }

        /// <summary>
        /// Run decoder on a chunk of triplane features.
        /// Caller must consume the returned array before the next call (buffer reuse).
        /// </summary>
        /// <param name="features">Flat float array of sampled triplane features.</param>
        /// <param name="count">Number of points in this chunk.</param>
        /// <returns>float[] of shape [count, 4] — density + RGB.</returns>
        internal async Task<float[]> RunChunkAsync(float[] features, int count)
        {
            if (!IsLoaded)
                throw new InvalidOperationException("OrtDecoderModel not loaded");

            int totalIn = count * FeatureDim;
            if (_inputBuffer == null || _inputBuffer.Length < totalIn)
                _inputBuffer = new float[totalIn];
            Buffer.BlockCopy(features, 0, _inputBuffer, 0, totalIn * sizeof(float));

            var tensor = new DenseTensor<float>(
                _inputBuffer.AsMemory(0, totalIn), new[] { count, FeatureDim });
            LoadInput(tensor);

            if (count != _lastChunkSize)
            {
                UpdateOutputDimensions(_outputName, new[] { count, OutputDim });
                _lastChunkSize = count;
            }

            await RunPreallocated();

            var output = GetPreallocatedOutput<float>();
            int totalOut = count * OutputDim;
            if (_outputBuffer == null || _outputBuffer.Length < totalOut)
                _outputBuffer = new float[totalOut];

            var srcSpan = output.Buffer.Span;
            srcSpan.Slice(0, totalOut).CopyTo(_outputBuffer.AsSpan(0, totalOut));
            return _outputBuffer;
        }
    }
}
#endif
