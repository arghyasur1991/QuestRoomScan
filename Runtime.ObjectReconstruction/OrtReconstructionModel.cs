#if HAS_ONNXRUNTIME
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Genesis.RoomScan.ObjectReconstruction
{
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
        private const string HiddenOutputName = "/backbone/transformer_blocks.7/Add_2_output_0";

        private readonly ExecutionProvider _ep;
        private readonly bool _mobileOptimized;

        private OrtModelBase _part1;
        private OrtModelBase _part2;
        private bool _preloaded;

        internal OrtReconstructionModel(ExecutionProvider ep, bool mobileOptimized)
        {
            _ep = ep;
            _mobileOptimized = mobileOptimized;
        }

        /// <summary>Load both sessions and keep alive for reuse (editor path).</summary>
        internal async Task PreloadAsync(CancellationToken ct)
        {
            if (_preloaded) return;

            _part1 = new PartModel();
            await ((PartModel)_part1).LoadAsync(Part1FileName, _ep, _mobileOptimized, ct);

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
            var inputTensor = new DenseTensor<float>(preprocessed, new[] { 1, 3, 512, 512 });

            var part1 = (PartModel)_part1;
            part1.SetInput(inputTensor);
            using var results1 = await part1.RunDisposablePublic();
            ct.ThrowIfCancellationRequested();

            var encoderStates = ExtractTensor(results1, EncoderOutputName);
            var hiddenStates = ExtractTensor(results1, HiddenOutputName);

            var part2 = (PartModel)_part2;
            part2.SetInput(EncoderOutputName, encoderStates);
            part2.SetInput(HiddenOutputName, hiddenStates);
            using var results2 = await part2.RunDisposablePublic();
            ct.ThrowIfCancellationRequested();

            return results2.First().AsTensor<float>().ToArray();
        }

        private async Task<float[]> RunSequentialAsync(float[] preprocessed, CancellationToken ct)
        {
            DenseTensor<float> encoderStates;
            DenseTensor<float> hiddenStates;

            {
                var part1 = new PartModel();
                await part1.LoadAsync(Part1FileName, _ep, _mobileOptimized, ct);
                await AsyncHelper.YieldFrame();

                var inputTensor = new DenseTensor<float>(preprocessed, new[] { 1, 3, 512, 512 });
                part1.SetInput(inputTensor);
                using var results1 = await part1.RunDisposablePublic();
                ct.ThrowIfCancellationRequested();

                encoderStates = CloneTensor(ExtractTensor(results1, EncoderOutputName));
                hiddenStates = CloneTensor(ExtractTensor(results1, HiddenOutputName));

                part1.Dispose();
            }
            await AsyncHelper.YieldFrame();

            float[] sceneCodes;
            {
                var part2 = new PartModel();
                await part2.LoadAsync(Part2FileName, _ep, _mobileOptimized, ct);
                await AsyncHelper.YieldFrame();

                part2.SetInput(EncoderOutputName, encoderStates);
                part2.SetInput(HiddenOutputName, hiddenStates);
                using var results2 = await part2.RunDisposablePublic();
                ct.ThrowIfCancellationRequested();

                sceneCodes = results2.First().AsTensor<float>().ToArray();
                part2.Dispose();
            }
            await AsyncHelper.YieldFrame();

            return sceneCodes;
        }

        private static DenseTensor<float> ExtractTensor(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, string name)
        {
            foreach (var r in results)
                if (r.Name == name)
                    return r.AsTensor<float>() as DenseTensor<float>;
            throw new InvalidOperationException($"Output '{name}' not found in results");
        }

        private static DenseTensor<float> CloneTensor(DenseTensor<float> src)
        {
            var data = src.ToArray();
            return new DenseTensor<float>(data, src.Dimensions.ToArray());
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

            internal void SetInput(DenseTensor<float> tensor)
            {
                LoadInput(0, tensor);
            }

            internal void SetInput(string name, DenseTensor<float> tensor)
            {
                _inputs.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
            }

            internal async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposablePublic()
            {
                return await RunDisposable();
            }
        }
    }
}
#endif
