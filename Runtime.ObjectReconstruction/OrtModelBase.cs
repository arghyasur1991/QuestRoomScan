#if HAS_ONNXRUNTIME
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace Genesis.RoomScan.ObjectReconstruction
{
    /// <summary>
    /// Abstract base for all ORT model wrappers. Provides sequential queue loading
    /// (LiveTalk pattern), mobile-optimized SessionOptions, dual Run modes
    /// (preallocated + disposable), and Spark-TTS _inputs.Clear() lifecycle.
    /// </summary>
    internal abstract class OrtModelBase : IDisposable
    {
        protected InferenceSession _session;
        protected List<string> _inputNames;
        protected readonly List<NamedOnnxValue> _inputs = new();
        protected List<NamedOnnxValue> _preallocatedOutputs;
        private bool _disposed;

        protected bool IsLoaded => _session != null;

        #region Loading

        protected async Task LoadSessionAsync(
            string relativePath, ExecutionProvider ep, bool mobileOptimized, CancellationToken ct)
        {
            string modelPath = await ModelPathResolver.ResolveAsync(relativePath, ct);
            var options = CreateSessionOptions(ep, mobileOptimized);
            string modelName = Path.GetFileNameWithoutExtension(relativePath);

            var task = new Task(() =>
            {
                if (ep == ExecutionProvider.CoreML)
                {
                    try
                    {
                        _session = CreateSessionWithCoreMLRecovery(modelPath, options);
                    }
                    catch (Exception e) when (
                        e.Message.Contains("output_features has no value") ||
                        e.Message.Contains("gemm_input") ||
                        e.Message.Contains("DynamicQuantizeLinear") ||
                        e.Message.Contains("QuantizeLinear"))
                    {
                        Logger.Warning($"[OrtModelBase] CoreML incompatible with {modelName} " +
                                       $"(likely INT8 quantized), falling back to CPU: {e.Message}");
                        var cpuOptions = CreateSessionOptions(ExecutionProvider.CPU, mobileOptimized);
                        _session = new InferenceSession(modelPath, cpuOptions);
                    }
                }
                else
                {
                    _session = new InferenceSession(modelPath, options);
                }
            });
            OrtLoadQueue.Enqueue(task, modelName);
            await task;

            _inputNames = _session.InputMetadata.Keys.ToList();

            _preallocatedOutputs = new List<NamedOnnxValue>();
            foreach (var kvp in _session.OutputMetadata)
            {
                if (kvp.Value.IsTensor && kvp.Value.ElementType == typeof(float))
                {
                    var dims = kvp.Value.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
                    var tensor = new DenseTensor<float>(dims);
                    _preallocatedOutputs.Add(NamedOnnxValue.CreateFromTensor(kvp.Key, tensor));
                }
                else if (kvp.Value.IsTensor && kvp.Value.ElementType == typeof(long))
                {
                    var dims = kvp.Value.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
                    var tensor = new DenseTensor<long>(dims);
                    _preallocatedOutputs.Add(NamedOnnxValue.CreateFromTensor(kvp.Key, tensor));
                }
            }
        }

        private static SessionOptions CreateSessionOptions(ExecutionProvider ep, bool mobileOptimized)
        {
            var options = new SessionOptions
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };

            if (mobileOptimized)
            {
                options.EnableMemoryPattern = false;
                options.EnableCpuMemArena = false;
                options.IntraOpNumThreads = 1;
            }

            switch (ep)
            {
                case ExecutionProvider.NNAPI:
                    options.AppendExecutionProvider_Nnapi(
                        NnapiFlags.NNAPI_FLAG_USE_FP16);
                    break;
                case ExecutionProvider.XNNPACK:
                    options.AppendExecutionProvider("XNNPACK",
                        new Dictionary<string, string>());
                    break;
                case ExecutionProvider.CoreML:
                    ConfigureCoreML(options);
                    break;
            }

            return options;
        }

        private static string _coremlCacheDir;

        private static string GetCoreMLCacheDirectory()
        {
            if (_coremlCacheDir != null) return _coremlCacheDir;
            var dataPath = Application.platform == RuntimePlatform.IPhonePlayer
                ? Application.persistentDataPath
                : Application.dataPath;
            _coremlCacheDir = Path.Combine(dataPath, "Models", "coreml_cache");
            return _coremlCacheDir;
        }

        /// <summary>
        /// Configures CoreML with MLProgram format, GPU compute, model caching,
        /// and cache corruption recovery (LiveTalk pattern).
        /// </summary>
        private static void ConfigureCoreML(SessionOptions options)
        {
            string cacheDir = GetCoreMLCacheDirectory();
            if (!Directory.Exists(cacheDir))
                Directory.CreateDirectory(cacheDir);

            try
            {
                var coremlOptions = new Dictionary<string, string>
                {
                    ["ModelFormat"] = "MLProgram",
                    ["MLComputeUnits"] = "CPUAndGPU",
                    ["RequireStaticInputShapes"] = "0",
                    ["EnableOnSubgraphs"] = "1",
                    ["ModelCacheDirectory"] = cacheDir,
                };
                options.AppendExecutionProvider("CoreML", coremlOptions);
            }
            catch (Exception e)
            {
                Logger.Warning($"[OrtModelBase] CoreML dict config failed ({e.Message}), trying flags fallback");
                try
                {
                    options.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU |
                        CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM |
                        CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                }
                catch (Exception e2)
                {
                    Logger.Warning($"[OrtModelBase] CoreML flags fallback also failed ({e2.Message}), using CPU");
                }
            }
        }

        /// <summary>
        /// Creates session with CoreML cache corruption recovery.
        /// On Manifest.json / cache errors, waits and retries once before giving up.
        /// </summary>
        private static InferenceSession CreateSessionWithCoreMLRecovery(
            string modelPath, SessionOptions options)
        {
            try
            {
                return new InferenceSession(modelPath, options);
            }
            catch (Exception e) when (
                e.Message.Contains("Manifest.json") ||
                e.Message.Contains("coreml_cache") ||
                e.Message.Contains("manifest does not exist"))
            {
                Logger.Warning($"[OrtModelBase] CoreML cache corruption detected, retrying: {e.Message}");
                Thread.Sleep(1000);
                return new InferenceSession(modelPath, options);
            }
        }

        #endregion

        #region Input

        protected void LoadInput<T>(int index, Tensor<T> tensor)
        {
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], tensor));
        }

        protected void LoadInput<T>(Tensor<T> tensor)
        {
            LoadInput(0, tensor);
        }

        #endregion

        #region Inference

        /// <summary>
        /// Hot-path run: writes into preallocated output tensors. Zero allocation per call.
        /// </summary>
        protected async Task RunPreallocated()
        {
            try
            {
                var runOptions = new RunOptions
                {
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                };
                await Task.Run(() => _session.Run(_inputs, _preallocatedOutputs, runOptions));
            }
            finally
            {
                _inputs.Clear();
            }
        }

        /// <summary>
        /// One-shot run: ORT allocates outputs. Caller disposes the returned collection.
        /// </summary>
        protected async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable()
        {
            try
            {
                var runOptions = new RunOptions
                {
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                };
                return await Task.Run(() =>
                    _session.Run(_inputs, _session.OutputNames, runOptions));
            }
            finally
            {
                _inputs.Clear();
            }
        }

        #endregion

        #region Output

        protected DenseTensor<T> GetPreallocatedOutput<T>(string outputName)
        {
            var output = _preallocatedOutputs.FirstOrDefault(o => o.Name == outputName);
            return output?.AsTensor<T>() as DenseTensor<T>;
        }

        protected DenseTensor<T> GetPreallocatedOutput<T>()
        {
            return _preallocatedOutputs[0].AsTensor<T>() as DenseTensor<T>;
        }

        protected void UpdateOutputDimensions(string outputName, int[] newDims)
        {
            int idx = _preallocatedOutputs.FindIndex(o => o.Name == outputName);
            if (idx < 0) return;
            var tensor = new DenseTensor<float>(newDims);
            _preallocatedOutputs[idx] = NamedOnnxValue.CreateFromTensor(outputName, tensor);
        }

        #endregion

        #region Dispose

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                _session?.Dispose();
                _session = null;
                _preallocatedOutputs?.Clear();
                _preallocatedOutputs = null;
                _inputs.Clear();
                _inputNames = null;
            }
            _disposed = true;
        }

        ~OrtModelBase()
        {
            Dispose(false);
        }

        #endregion
    }
}
#endif
