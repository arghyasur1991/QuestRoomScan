#if HAS_AI_INFERENCE
using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.AIDetection
{
    /// <summary>
    /// YOLO-based object detection via Unity Inference Engine.
    /// Compiles a model graph with built-in GPU NMS (matching Unity's official reference),
    /// runs time-sliced inference to avoid VR frame drops, and returns Detection[] results.
    /// </summary>
    public class YoloDetectionModel : IDetectionModel
    {
        private readonly ModelAsset _modelAsset;
        private readonly TextAsset _classLabelsAsset;
        private readonly BackendType _backend;
        private readonly bool _splitOverFrames;
        private readonly int _layersPerFrame;
        private readonly float _scoreThreshold;
        private readonly float _iouThreshold;

        private Model _rawModel;
        private Worker _worker;
        private Tensor<float> _centersToCorners;
        private string[] _labels;
        private int _inputW, _inputH;
        private bool _disposed;

        public string ModelName => "YOLOv9t";
        public string[] ClassLabels => _labels;
        public bool IsLoaded => _worker != null;

        public YoloDetectionModel(
            ModelAsset modelAsset,
            TextAsset classLabelsAsset,
            BackendType backend = BackendType.GPUCompute,
            bool splitOverFrames = true,
            int layersPerFrame = 22,
            float scoreThreshold = 0.5f,
            int maxDetections = 100,
            float iouThreshold = 0.5f)
        {
            _modelAsset = modelAsset;
            _classLabelsAsset = classLabelsAsset;
            _backend = backend;
            _splitOverFrames = splitOverFrames;
            _layersPerFrame = layersPerFrame;
            _scoreThreshold = scoreThreshold;
            _iouThreshold = iouThreshold;
        }

        public async Task LoadAsync()
        {
            if (_modelAsset == null)
                throw new InvalidOperationException("No model asset assigned");

            // Step 1: Parse the ONNX model (heavy — ~100-300ms on Quest)
            _rawModel = ModelLoader.Load(_modelAsset);
            await Task.Yield();

            var inputShape = _rawModel.inputs[0].shape;
            _inputH = inputShape.Get(2) > 0 ? inputShape.Get(2) : 640;
            _inputW = inputShape.Get(3) > 0 ? inputShape.Get(3) : 640;

            // Quest 3 has a 128MB max compute buffer — YOLO at 640x640 exceeds it.
            // Cap to 320x320 on Android (reduces intermediate buffers by ~4x).
#if UNITY_ANDROID && !UNITY_EDITOR
            const int maxDim = 320;
            if (_inputH > maxDim || _inputW > maxDim)
            {
                _inputH = maxDim;
                _inputW = maxDim;
            }
#endif

            _labels = _classLabelsAsset != null
                ? _classLabelsAsset.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                : Array.Empty<string>();

            // Step 2: Build the functional graph with NMS (moderate cost).
            // Use explicit input shape (may differ from model default on Quest).
            _centersToCorners = new Tensor<float>(new TensorShape(4, 4), new float[]
            {
                1, 0, 1, 0,
                0, 1, 0, 1,
                -0.5f, 0, 0.5f, 0,
                0, -0.5f, 0, 0.5f
            });

            var graph = new FunctionalGraph();
            var input = graph.AddInput(DataType.Float,
                new DynamicTensorShape(1, 3, _inputH, _inputW));
            var modelOutput = Functional.Forward(_rawModel, new[] { input })[0];
            var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..];
            var scores = Functional.ReduceMax(allScores, 0);
            var classIDs = Functional.ArgMax(allScores, 0);
            var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(_centersToCorners));
            var indices = Functional.NMS(boxCorners, scores, _iouThreshold, _scoreThreshold);
            var coords = Functional.IndexSelect(boxCoords, 0, indices);
            var labelIDs = Functional.IndexSelect(classIDs, 0, indices);
            await Task.Yield();

            // Step 3: Compile graph + create worker (heavy — GPU shader compilation)
            var compiled = graph.Compile(coords, labelIDs);
            await Task.Yield();

            _worker = new Worker(compiled, _backend);
            await Task.Yield();
        }

        public async Task<Detection[]> DetectAsync(Texture src, CancellationToken ct = default)
        {
            if (_worker == null || src == null) return Array.Empty<Detection>();

            using var input = new Tensor<float>(new TensorShape(1, 3, _inputH, _inputW));
            TextureConverter.ToTensor(src, input);

            if (_splitOverFrames)
            {
                var it = _worker.ScheduleIterable(input);
                int steps = 0;
                while (it.MoveNext())
                {
                    ct.ThrowIfCancellationRequested();
                    if (++steps % _layersPerFrame == 0) await Task.Yield();
                }
            }
            else
            {
                _worker.Schedule(input);
            }

            // Read post-NMS outputs: coords (N, 4) and labelIDs (N)
            using var coordsCpu = await (_worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndCloneAsync();
            using var labelsCpu = await (_worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndCloneAsync();
            ct.ThrowIfCancellationRequested();

            float scaleX = (float)src.width / _inputW;
            float scaleY = (float)src.height / _inputH;

            int count = coordsCpu.shape[0];
            var results = new Detection[Mathf.Min(count, 200)];

            for (int n = 0; n < results.Length; n++)
            {
                float cx = coordsCpu[n, 0];
                float cy = coordsCpu[n, 1];
                float w = coordsCpu[n, 2];
                float h = coordsCpu[n, 3];
                int classId = labelsCpu[n];

                results[n] = new Detection
                {
                    boundingBox = new Rect(
                        (cx - w * 0.5f) * scaleX,
                        (cy - h * 0.5f) * scaleY,
                        w * scaleX,
                        h * scaleY),
                    classId = classId,
                    label = classId >= 0 && classId < _labels.Length
                        ? _labels[classId] : $"cls_{classId}",
                    confidence = 1f
                };
            }

            return results;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _worker?.Dispose();
            _worker = null;
            _centersToCorners?.Dispose();
            _centersToCorners = null;
        }
    }
}
#endif
