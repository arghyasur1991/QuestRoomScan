using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// A single 2D detection from a model inference pass.
    /// </summary>
    [Serializable]
    public struct Detection
    {
        /// <summary>Bounding box in source texture pixel coords (xMin, yMin, width, height).</summary>
        public Rect boundingBox;
        /// <summary>Zero-based class index from the model.</summary>
        public int classId;
        /// <summary>Human-readable label resolved from the model's class list.</summary>
        public string label;
        /// <summary>Confidence score in [0,1].</summary>
        public float confidence;
    }

    /// <summary>
    /// Pluggable abstraction for on-device AI models. Lives in the core assembly
    /// (no inference engine dependency) so SceneObjectRegistry and RoomScanner can
    /// reference it regardless of whether com.unity.ai.inference is installed.
    /// Concrete implementations (e.g. YoloDetectionModel) live in the optional
    /// Runtime.AIDetection assembly.
    /// </summary>
    public interface IDetectionModel : IDisposable
    {
        string ModelName { get; }
        string[] ClassLabels { get; }
        bool IsLoaded { get; }

        Task LoadAsync();
        Task<Detection[]> DetectAsync(Texture src, CancellationToken ct = default);
    }
}
