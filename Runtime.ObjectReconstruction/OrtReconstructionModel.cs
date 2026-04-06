#if HAS_ONNXRUNTIME
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>Per-sub-phase timing breakdown for the forward pass.</summary>
    internal struct ForwardTiming
    {
        internal float Part1LoadMs;
        internal float Part1RunMs;
        internal float Part2LoadMs;
        internal float Part2RunMs;

        public override string ToString() =>
            $"p1Load={Part1LoadMs / 1000f:F1}s p1Run={Part1RunMs / 1000f:F1}s " +
            $"p2Load={Part2LoadMs / 1000f:F1}s p2Run={Part2RunMs / 1000f:F1}s";
    }

    /// <summary>
    /// Wraps the split TripoSR model (Part1 + Part2) via ONNX Runtime.
    /// <list type="bullet">
    /// <item><b>Preloaded</b> (editor): Both sessions kept alive across runs.</item>
    /// <item><b>Sequential</b> (Quest): Each half loaded, executed, and disposed
    ///   independently to minimize peak memory.</item>
    /// </list>
    /// </summary>
    internal sealed class OrtReconstructionModel : IDisposable
    {
        private const string Part1FileName = "ObjectReconstruction/triposr_part1.onnx";
        private const string Part2FileName = "ObjectReconstruction/triposr_part2.onnx";

        private const string EncoderOutputName = "/Reshape_output_0";

        private readonly ExecutionProvider _ep;
        private readonly bool _mobileOptimized;

        private OrtModelBase _part1;
        private OrtModelBase _part2;
        private bool _preloaded;
        private string _hiddenOutputName;
        private int _imageSize = 512;
        internal bool IsLoaded => _preloaded;

        /// <summary>Expected input image resolution (read from ONNX model metadata).</summary>
        internal int ImageSize => _imageSize;

        /// <summary>Timing from the most recent RunAsync call (populated after completion).</summary>
        internal ForwardTiming LastTiming { get; private set; }

        internal OrtReconstructionModel(ExecutionProvider ep, bool mobileOptimized)
        {
            _ep = ep;
            _mobileOptimized = mobileOptimized;
        }

        /// <summary>
        /// Briefly load Part1 to discover the expected image input size, then dispose.
        /// Used in sequential mode where we can't keep sessions alive but still need
        /// the image size for preprocessing.
        /// </summary>
        internal async Task DiscoverImageSizeAsync(CancellationToken ct)
        {
            var temp = new PartModel();
            await temp.LoadAsync(Part1FileName, _ep, _mobileOptimized, ct);
            _imageSize = temp.DiscoverImageSize();
            temp.Dispose();
        }

        /// <summary>Load both sessions and keep alive for reuse (editor path).</summary>
        internal async Task PreloadAsync(CancellationToken ct)
        {
            if (_preloaded) return;

            _part1 = new PartModel();
            await ((PartModel)_part1).LoadAsync(Part1FileName, _ep, _mobileOptimized, ct);
            _hiddenOutputName = ((PartModel)_part1).DiscoverHiddenOutputName();
            _imageSize = ((PartModel)_part1).DiscoverImageSize();

            _part2 = new PartModel();
            await ((PartModel)_part2).LoadAsync(Part2FileName, _ep, _mobileOptimized, ct);

            _preloaded = true;
        }

        /// <returns>Scene codes as float[].</returns>
        internal async Task<float[]> RunAsync(float[] preprocessed, CancellationToken ct)
        {
            return _preloaded
                ? await RunPreloadedAsync(preprocessed, ct)
                : await RunSequentialAsync(preprocessed, ct);
        }

        private async Task<float[]> RunPreloadedAsync(float[] preprocessed, CancellationToken ct)
        {
            var inputTensor = new DenseTensor<float>(preprocessed, new[] { 1, 3, _imageSize, _imageSize });

            var part1 = (PartModel)_part1;
            part1.SetInput(inputTensor);
            await part1.RunPreallocatedPublic();
            ct.ThrowIfCancellationRequested();

            var encoderStates = part1.GetOutputTensor(EncoderOutputName);
            var hiddenStates = part1.GetOutputTensor(_hiddenOutputName);

            var part2 = (PartModel)_part2;
            part2.SetInput(EncoderOutputName, encoderStates);
            part2.SetInput(_hiddenOutputName, hiddenStates);
            await part2.RunPreallocatedPublic();
            ct.ThrowIfCancellationRequested();

            return part2.GetOutputTensor().ToArray();
        }

        private async Task<float[]> RunSequentialAsync(float[] preprocessed, CancellationToken ct)
        {
            var timing = new ForwardTiming();
            var sw = Stopwatch.StartNew();

            DenseTensor<float> encoderStates;
            DenseTensor<float> hiddenStates;
            string hiddenName;

            {
                var part1 = new PartModel();
                await part1.LoadAsync(Part1FileName, _ep, _mobileOptimized, ct);
                hiddenName = part1.DiscoverHiddenOutputName();
                int imageSize = part1.DiscoverImageSize();
                timing.Part1LoadMs = sw.ElapsedMilliseconds;
                Logger.Info($"[TripoSR] Part1 loaded: {timing.Part1LoadMs:F0}ms (imageSize={imageSize})");
                sw.Restart();
                await AsyncHelper.YieldFrame();

                var inputTensor = new DenseTensor<float>(preprocessed, new[] { 1, 3, imageSize, imageSize });
                part1.SetInput(inputTensor);
                await part1.RunPreallocatedPublic();
                ct.ThrowIfCancellationRequested();

                timing.Part1RunMs = sw.ElapsedMilliseconds;
                Logger.Info($"[TripoSR] Part1 infer: {timing.Part1RunMs:F0}ms");
                sw.Restart();

                // Grab references before dispose — managed DenseTensor backing arrays
                // survive _preallocatedOutputs.Clear() since our locals keep them alive
                encoderStates = part1.GetOutputTensor(EncoderOutputName);
                hiddenStates = part1.GetOutputTensor(hiddenName);

                part1.Dispose();
            }
            GC.Collect();
            await AsyncHelper.YieldFrame();

            float[] sceneCodes;
            {
                var part2 = new PartModel();
                await part2.LoadAsync(Part2FileName, _ep, _mobileOptimized, ct);
                timing.Part2LoadMs = sw.ElapsedMilliseconds;
                Logger.Info($"[TripoSR] Part2 loaded: {timing.Part2LoadMs:F0}ms");
                sw.Restart();
                await AsyncHelper.YieldFrame();

                part2.SetInput(EncoderOutputName, encoderStates);
                part2.SetInput(hiddenName, hiddenStates);
                await part2.RunPreallocatedPublic();
                ct.ThrowIfCancellationRequested();

                timing.Part2RunMs = sw.ElapsedMilliseconds;
                Logger.Info($"[TripoSR] Part2 infer: {timing.Part2RunMs:F0}ms");
                sw.Stop();

                sceneCodes = part2.GetOutputTensor().ToArray();
                part2.Dispose();
            }
            GC.Collect();
            await AsyncHelper.YieldFrame();

            LastTiming = timing;
            Logger.Info($"[TripoSR] Forward breakdown: {timing}");

            return sceneCodes;
        }

        public void Dispose()
        {
            _part1?.Dispose();
            _part2?.Dispose();
            _part1 = null;
            _part2 = null;
            _preloaded = false;
        }

        /// <summary>Thin wrapper exposing OrtModelBase methods needed by the reconstruction model.</summary>
        private sealed class PartModel : OrtModelBase
        {
            internal async Task LoadAsync(
                string relativePath, ExecutionProvider ep, bool mobileOptimized, CancellationToken ct)
            {
                await LoadSessionAsync(relativePath, ep, mobileOptimized, ct);
            }

            /// <summary>
            /// Read the expected image input size (H dimension) from the ONNX model metadata.
            /// Returns 512 as fallback for models without explicit static shapes.
            /// </summary>
            internal int DiscoverImageSize()
            {
                if (_session.InputMetadata.TryGetValue("image", out var meta)
                    && meta.Dimensions.Length == 4
                    && meta.Dimensions[2] > 0)
                    return meta.Dimensions[2];
                return 512;
            }

            /// <summary>
            /// Find the Add_2_output_0 tensor name from Part 1's outputs. This varies
            /// by model quality: full 16L splits at blocks.7, pruned 13L/12L at blocks.5.
            /// </summary>
            internal string DiscoverHiddenOutputName()
            {
                foreach (string name in _session.OutputNames)
                    if (name.Contains("Add_2_output_0"))
                        return name;
                throw new InvalidOperationException(
                    "Part1 model has no Add_2_output_0 output — wrong model file?");
            }

            internal void SetInput(DenseTensor<float> tensor)
            {
                LoadInput(0, tensor);
            }

            internal void SetInput(string name, DenseTensor<float> tensor)
            {
                _inputs.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
            }

            internal async Task RunPreallocatedPublic()
            {
                await RunPreallocated();
            }

            internal DenseTensor<float> GetOutputTensor(string name)
            {
                return GetPreallocatedOutput<float>(name);
            }

            internal DenseTensor<float> GetOutputTensor()
            {
                return GetPreallocatedOutput<float>();
            }
        }
    }
}
#endif
