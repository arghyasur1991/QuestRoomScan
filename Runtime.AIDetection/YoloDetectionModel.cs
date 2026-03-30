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
    /// Loads a .sentis ModelAsset, runs time-sliced GPU inference, and returns
    /// Detection[] results. Uses our own camera pipeline, not Meta's building blocks.
    /// </summary>
    public class YoloDetectionModel : IDetectionModel
    {
        private readonly ModelAsset _modelAsset;
        private readonly TextAsset _classLabelsAsset;
        private readonly BackendType _backend;
        private readonly bool _splitOverFrames;
        private readonly int _layersPerFrame;
        private readonly float _scoreThreshold;
        private readonly int _maxDetections;

        private Model _model;
        private Worker _worker;
        private string[] _labels;
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
            int maxDetections = 100)
        {
            _modelAsset = modelAsset;
            _classLabelsAsset = classLabelsAsset;
            _backend = backend;
            _splitOverFrames = splitOverFrames;
            _layersPerFrame = layersPerFrame;
            _scoreThreshold = scoreThreshold;
            _maxDetections = maxDetections;
        }

        public async Task LoadAsync()
        {
            if (_modelAsset == null)
                throw new InvalidOperationException("No model asset assigned");

            _model = ModelLoader.Load(_modelAsset);
            _worker = new Worker(_model, _backend);

            _labels = _classLabelsAsset != null
                ? _classLabelsAsset.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                : Array.Empty<string>();

            // Warmup pass to avoid first-frame spike
            await Task.Yield();
        }

        public async Task<Detection[]> DetectAsync(Texture src, CancellationToken ct = default)
        {
            if (_worker == null || src == null) return Array.Empty<Detection>();

            var inputShape = _model.inputs[0].shape;
            int inputH = inputShape[2].value;
            int inputW = inputShape[3].value;

            using var input = new Tensor<float>(new TensorShape(1, 3, inputH, inputW));
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

            // Read output tensors — shape depends on model export, handle common YOLO formats
            var output = _worker.PeekOutput(0) as Tensor<float>;
            if (output == null) return Array.Empty<Detection>();

            var cpuOutput = await output.ReadbackAndCloneAsync();
            ct.ThrowIfCancellationRequested();

            float scaleX = (float)src.width / inputW;
            float scaleY = (float)src.height / inputH;

            var results = DecodeDetections(cpuOutput, scaleX, scaleY);
            cpuOutput.Dispose();
            return ApplyNms(results, 0.5f);
        }

        private List<Detection> DecodeDetections(Tensor<float> output, float scaleX, float scaleY)
        {
            var results = new List<Detection>();
            var shape = output.shape;

            // Common YOLO output: [1, numClasses+4, numBoxes] or [1, numBoxes, numClasses+4]
            int dim1 = shape[1];
            int dim2 = shape[2];
            bool transposed = dim1 > dim2;
            int numBoxes = transposed ? dim2 : dim1;
            int numFields = transposed ? dim1 : dim2;
            int numClasses = numFields - 4;
            if (numClasses <= 0) return results;

            for (int b = 0; b < numBoxes && results.Count < _maxDetections; b++)
            {
                float bestScore = 0;
                int bestClass = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = transposed ? output[0, c + 4, b] : output[0, b, c + 4];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                if (bestScore < _scoreThreshold) continue;

                float cx, cy, w, h;
                if (transposed)
                {
                    cx = output[0, 0, b]; cy = output[0, 1, b];
                    w  = output[0, 2, b]; h  = output[0, 3, b];
                }
                else
                {
                    cx = output[0, b, 0]; cy = output[0, b, 1];
                    w  = output[0, b, 2]; h  = output[0, b, 3];
                }

                results.Add(new Detection
                {
                    boundingBox = new Rect(
                        (cx - w * 0.5f) * scaleX,
                        (cy - h * 0.5f) * scaleY,
                        w * scaleX,
                        h * scaleY),
                    classId = bestClass,
                    label = bestClass < _labels.Length ? _labels[bestClass] : $"cls_{bestClass}",
                    confidence = bestScore
                });
            }

            return results;
        }

        /// <summary>Simple CPU non-max suppression — sufficient for ≤100 candidates.</summary>
        private static Detection[] ApplyNms(List<Detection> detections, float iouThreshold)
        {
            detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));
            var kept = new List<Detection>();

            for (int i = 0; i < detections.Count; i++)
            {
                bool suppressed = false;
                for (int j = 0; j < kept.Count; j++)
                {
                    if (IoU(detections[i].boundingBox, kept[j].boundingBox) > iouThreshold)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed) kept.Add(detections[i]);
            }

            return kept.ToArray();
        }

        private static float IoU(Rect a, Rect b)
        {
            float x1 = Mathf.Max(a.xMin, b.xMin);
            float y1 = Mathf.Max(a.yMin, b.yMin);
            float x2 = Mathf.Min(a.xMax, b.xMax);
            float y2 = Mathf.Min(a.yMax, b.yMax);
            float inter = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
            float union = a.width * a.height + b.width * b.height - inter;
            return union > 0 ? inter / union : 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _worker?.Dispose();
            _worker = null;
        }
    }
}
#endif
