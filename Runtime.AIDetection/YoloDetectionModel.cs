#if HAS_AI_INFERENCE
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

namespace Genesis.RoomScan.AIDetection
{
    /// <summary>
    /// YOLO-based object detection via Unity Inference Engine.
    /// Following Meta's Quest-proven pattern: compile the model graph WITHOUT NMS,
    /// then run NMS in a separate GPU compute shader dispatch. This avoids
    /// Functional.NMS inflating intermediate buffers beyond Quest 3's 128MB limit.
    /// Only the final kept detections (~10-20) are read back to CPU.
    /// </summary>
    public class YoloDetectionModel : IDetectionModel
    {
        private readonly ModelAsset _modelAsset;
        private readonly TextAsset _classLabelsAsset;
        private readonly ComputeShader _nmsShader;
        private readonly BackendType _backend;
        private readonly bool _splitOverFrames;
        private readonly int _layersPerFrame;
        private readonly float _scoreThreshold;
        private readonly float _iouThreshold;
        private readonly int _maxInputResolution;
        private const int MaxKeptBoxes = 200;

        private Model _rawModel;
        private Worker _worker;
        private string[] _labels;
        private int _inputW, _inputH;
        private bool _disposed;

        // GPU NMS resources
        private int _nmsKernel;
        private ComputeBuffer _outCoordsGpu;
        private ComputeBuffer _outLabelIDsGpu;
        private ComputeBuffer _outScoresGpu;
        private ComputeBuffer _countGpu;
        private Vector4[] _coordsReadback;
        private int[] _labelsReadback;
        private float[] _scoresReadback;
        private readonly int[] _countReadback = new int[1];

        public string ModelName => "YOLOv9t";
        public string[] ClassLabels => _labels;
        public bool IsLoaded => _worker != null;

        public YoloDetectionModel(
            ModelAsset modelAsset,
            TextAsset classLabelsAsset,
            ComputeShader nmsShader,
            BackendType backend = BackendType.GPUCompute,
            bool splitOverFrames = true,
            int layersPerFrame = 22,
            float scoreThreshold = 0.5f,
            float iouThreshold = 0.5f,
            int maxInputResolution = 0)
        {
            _modelAsset = modelAsset;
            _classLabelsAsset = classLabelsAsset;
            _nmsShader = nmsShader;
            _backend = backend;
            _splitOverFrames = splitOverFrames;
            _layersPerFrame = layersPerFrame;
            _scoreThreshold = scoreThreshold;
            _iouThreshold = iouThreshold;
            _maxInputResolution = maxInputResolution;
        }

        public async Task LoadAsync()
        {
            if (_modelAsset == null)
                throw new InvalidOperationException("No model asset assigned");

            _rawModel = ModelLoader.Load(_modelAsset);
            await Task.Yield();

            var inputShape = _rawModel.inputs[0].shape;
            _inputH = inputShape.Get(2) > 0 ? inputShape.Get(2) : 640;
            _inputW = inputShape.Get(3) > 0 ? inputShape.Get(3) : 640;

            if (_maxInputResolution > 0)
            {
                _inputH = Mathf.Min(_inputH, _maxInputResolution);
                _inputW = Mathf.Min(_inputW, _maxInputResolution);
            }

            _labels = _classLabelsAsset != null
                ? _classLabelsAsset.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                : Array.Empty<string>();

            // Build graph WITHOUT NMS — output raw tensors for GPU NMS post-processing.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(DataType.Float,
                new DynamicTensorShape(1, 3, _inputH, _inputW));
            var modelOutput = Functional.Forward(_rawModel, new[] { input })[0];
            var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);  // (N, 4) cx,cy,w,h
            var allScores = modelOutput[0, 4.., ..];                    // (classes, N)
            var scores = Functional.ReduceMax(allScores, 0);            // (N)
            var classIDs = Functional.ArgMax(allScores, 0);             // (N)
            await Task.Yield();

            var compiled = graph.Compile(boxCoords, classIDs, scores);
            await Task.Yield();

            _worker = new Worker(compiled, _backend);

            // Init GPU NMS buffers
            InitNmsBuffers();
            await Task.Yield();

            Logger.Info($"[YoloDetectionModel] Loaded — input={_inputW}x{_inputH}, " +
                        $"labels={_labels.Length}, backend={_backend}, " +
                        $"nmsShader={((_nmsShader != null) ? "GPU" : "CPU fallback")}");
        }

        private void InitNmsBuffers()
        {
            if (_nmsShader == null) return;

            _nmsKernel = _nmsShader.FindKernel("RunNMS");
            _outCoordsGpu = new ComputeBuffer(MaxKeptBoxes, sizeof(float) * 4, ComputeBufferType.Append);
            _outLabelIDsGpu = new ComputeBuffer(MaxKeptBoxes, sizeof(int), ComputeBufferType.Append);
            _outScoresGpu = new ComputeBuffer(MaxKeptBoxes, sizeof(float), ComputeBufferType.Append);
            _countGpu = new ComputeBuffer(1, sizeof(int), ComputeBufferType.IndirectArguments);
            _coordsReadback = new Vector4[MaxKeptBoxes];
            _labelsReadback = new int[MaxKeptBoxes];
            _scoresReadback = new float[MaxKeptBoxes];
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

            float scaleX = (float)src.width / _inputW;
            float scaleY = (float)src.height / _inputH;

            if (_nmsShader != null)
                return RunGpuNms(scaleX, scaleY);

            return await RunCpuNmsFallback(scaleX, scaleY, ct);
        }

        private Detection[] RunGpuNms(float scaleX, float scaleY)
        {
            var coordsTensor = _worker.PeekOutput("output_0") as Tensor<float>;
            var classIdsTensor = _worker.PeekOutput("output_1") as Tensor<int>;
            var scoresTensor = _worker.PeekOutput("output_2") as Tensor<float>;

            var coordsPin = ComputeTensorData.Pin(coordsTensor);
            var classIdsPin = ComputeTensorData.Pin(classIdsTensor);
            var scoresPin = ComputeTensorData.Pin(scoresTensor);

            uint numCandidates = (uint)coordsTensor.shape[0];

            _outCoordsGpu.SetCounterValue(0);
            _outLabelIDsGpu.SetCounterValue(0);
            _outScoresGpu.SetCounterValue(0);

            _nmsShader.SetBuffer(_nmsKernel, "inBoxCoords", coordsPin.buffer);
            _nmsShader.SetBuffer(_nmsKernel, "inClassIDs", classIdsPin.buffer);
            _nmsShader.SetBuffer(_nmsKernel, "inScores", scoresPin.buffer);
            _nmsShader.SetBuffer(_nmsKernel, "outCoords", _outCoordsGpu);
            _nmsShader.SetBuffer(_nmsKernel, "outLabelIDs", _outLabelIDsGpu);
            _nmsShader.SetBuffer(_nmsKernel, "outScores", _outScoresGpu);
            _nmsShader.SetFloat("scoreThreshold", _scoreThreshold);
            _nmsShader.SetFloat("iouThreshold", _iouThreshold);
            _nmsShader.SetInt("numCandidates", (int)numCandidates);

            _nmsShader.Dispatch(_nmsKernel, (int)((numCandidates + 63) / 64), 1, 1);

            ComputeBuffer.CopyCount(_outCoordsGpu, _countGpu, 0);
            _countGpu.GetData(_countReadback);
            int boxesFound = Mathf.Min(_countReadback[0], MaxKeptBoxes);

            if (boxesFound == 0) return Array.Empty<Detection>();

            _outCoordsGpu.GetData(_coordsReadback, 0, 0, boxesFound);
            _outLabelIDsGpu.GetData(_labelsReadback, 0, 0, boxesFound);
            _outScoresGpu.GetData(_scoresReadback, 0, 0, boxesFound);

            var results = new Detection[boxesFound];
            for (int n = 0; n < boxesFound; n++)
            {
                var c = _coordsReadback[n];
                int classId = _labelsReadback[n];

                results[n] = new Detection
                {
                    boundingBox = new Rect(
                        (c.x - c.z * 0.5f) * scaleX,
                        (c.y - c.w * 0.5f) * scaleY,
                        c.z * scaleX,
                        c.w * scaleY),
                    classId = classId,
                    label = classId >= 0 && classId < _labels.Length
                        ? _labels[classId] : $"cls_{classId}",
                    confidence = _scoresReadback[n]
                };
            }

            return results;
        }

        private async Task<Detection[]> RunCpuNmsFallback(float scaleX, float scaleY, CancellationToken ct)
        {
            using var coordsCpu = await (_worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndCloneAsync();
            using var classIdsCpu = await (_worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndCloneAsync();
            using var scoresCpu = await (_worker.PeekOutput("output_2") as Tensor<float>).ReadbackAndCloneAsync();
            ct.ThrowIfCancellationRequested();

            int numBoxes = coordsCpu.shape[0];
            var candidates = new List<(Rect box, int classId, float score)>();

            for (int i = 0; i < numBoxes; i++)
            {
                float score = scoresCpu[i];
                if (score < _scoreThreshold) continue;

                float cx = coordsCpu[i, 0], cy = coordsCpu[i, 1];
                float w = coordsCpu[i, 2], h = coordsCpu[i, 3];

                candidates.Add((
                    new Rect((cx - w * 0.5f) * scaleX, (cy - h * 0.5f) * scaleY,
                             w * scaleX, h * scaleY),
                    classIdsCpu[i], score));
            }

            candidates.Sort((a, b) => b.score.CompareTo(a.score));
            var kept = new List<Detection>();
            var suppressed = new bool[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                if (suppressed[i]) continue;
                var (box, classId, score) = candidates[i];
                kept.Add(new Detection
                {
                    boundingBox = box,
                    classId = classId,
                    label = classId >= 0 && classId < _labels.Length
                        ? _labels[classId] : $"cls_{classId}",
                    confidence = score
                });
                if (kept.Count >= MaxKeptBoxes) break;

                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (!suppressed[j] && IoU(box, candidates[j].box) > _iouThreshold)
                        suppressed[j] = true;
                }
            }

            return kept.ToArray();
        }

        private static float IoU(Rect a, Rect b)
        {
            float x1 = Mathf.Max(a.xMin, b.xMin);
            float y1 = Mathf.Max(a.yMin, b.yMin);
            float x2 = Mathf.Min(a.xMax, b.xMax);
            float y2 = Mathf.Min(a.yMax, b.yMax);
            float intersection = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
            float union = a.width * a.height + b.width * b.height - intersection;
            return union > 0 ? intersection / union : 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _worker?.Dispose();
            _worker = null;
            _outCoordsGpu?.Release();
            _outLabelIDsGpu?.Release();
            _outScoresGpu?.Release();
            _countGpu?.Release();
        }
    }
}
#endif
