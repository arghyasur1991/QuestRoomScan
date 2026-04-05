#if HAS_ONNXRUNTIME
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
        private static bool _ortLoggingInitialized;
        private static OrtLoggingLevel _ortLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        /// <summary>
        /// Earliest possible runtime init — runs before any SessionOptions can be created,
        /// ensuring OrtEnv is created with our custom callback (Spark-TTS / LiveTalk pattern).
        /// SubsystemRegistration fires before BeforeSceneLoad and AfterSceneLoad.
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void EarlyInitOrtLogging()
        {
            _ortLoggingInitialized = false;
            InitializeOrtLogging(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);
        }

        /// <summary>
        /// Initializes ORT environment with custom Unity logging callback.
        /// Routes ALL ORT internal logs (EP assignments, graph partitioning, QNN decisions)
        /// through Unity's Logger so they appear in logcat with the [RoomScan] tag.
        /// If OrtEnv was already created (e.g. by another package), updates its log level.
        /// </summary>
        internal static void InitializeOrtLogging(OrtLoggingLevel level = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
        {
            _ortLogLevel = level;

            if (_ortLoggingInitialized)
            {
                // Already initialized with callback — just update the level if needed
                if (OrtEnv.IsCreated)
                {
                    try
                    {
                        OrtEnv.Instance().EnvLogLevel = level;
                    }
                    catch (Exception) { /* best effort */ }
                }
                return;
            }
            _ortLoggingInitialized = true;

            if (OrtEnv.IsCreated)
            {
                // OrtEnv was created before us (e.g. by asus4 bootstrap).
                // Can't install callback, but CAN update the log level.
                Logger.Info("[OrtModelBase] OrtEnv already created, updating log level only");
                try
                {
                    OrtEnv.Instance().EnvLogLevel = level;
                    Logger.Info($"[OrtModelBase] OrtEnv log level updated to {level}");
                }
                catch (Exception e)
                {
                    Logger.Warning($"[OrtModelBase] Failed to update OrtEnv log level: {e.Message}");
                }
                return;
            }

            try
            {
                var options = new EnvironmentCreationOptions
                {
                    logLevel = level,
                    logId = "Sentience",
                    loggingFunction = OrtLoggingCallback,
                };
                OrtEnv.CreateInstanceWithOptions(ref options);
                Logger.Info($"[OrtModelBase] ORT env created with custom callback (level={level})");
            }
            catch (Exception e)
            {
                Logger.Warning($"[OrtModelBase] ORT logging init failed: {e.Message}");
            }
        }

        private static void OrtLoggingCallback(
            IntPtr param, OrtLoggingLevel severity, string category,
            string logId, string codeLocation, string message)
        {
            string tag = $"[ORT/{category}]";
            switch (severity)
            {
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE:
                    Logger.Verbose($"{tag} {message}");
                    break;
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO:
                    Logger.Info($"{tag} {message}");
                    break;
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING:
                    Logger.Warning($"{tag} {message}");
                    break;
                default:
                    Logger.Error($"{tag} {message}");
                    break;
            }
        }

        protected InferenceSession _session;
        protected List<string> _inputNames;
        protected readonly List<NamedOnnxValue> _inputs = new();
        protected List<NamedOnnxValue> _preallocatedOutputs;
        private bool _disposed;
        private bool _profilingEnabled;

        internal bool IsLoaded => _session != null;

        #region Loading

        protected async Task LoadSessionAsync(
            string relativePath, ExecutionProvider ep, bool mobileOptimized, CancellationToken ct)
        {
            string modelPath = await ModelPathResolver.ResolveAsync(relativePath, ct);
            string modelName = Path.GetFileNameWithoutExtension(relativePath);

            string modelCacheDir = null;
            if (ep == ExecutionProvider.CoreML)
                modelCacheDir = GetPerModelCacheDir(modelPath, modelName);

            var options = CreateSessionOptions(ep, mobileOptimized, modelCacheDir);

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
                else if (ep == ExecutionProvider.QNN_HTP)
                {
                    try
                    {
                        Logger.Info($"[OrtModelBase] Creating QNN HTP session for {modelName}...");
                        _session = new InferenceSession(modelPath, options);
                        Logger.Info($"[OrtModelBase] QNN HTP session created OK for {modelName}");
                    }
                    catch (Exception e)
                    {
                        Logger.Warning($"[OrtModelBase] QNN HTP rejected {modelName}: {e.Message}");
                        if (e.InnerException != null)
                            Logger.Warning($"[OrtModelBase] QNN inner: {e.InnerException.Message}");
                        Logger.Info($"[OrtModelBase] Falling back to CPU for {modelName}");
                        _profilingEnabled = false;
                        var cpuOptions = CreateSessionOptions(ExecutionProvider.CPU, mobileOptimized);
                        cpuOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
                        _session = new InferenceSession(modelPath, cpuOptions);
                        Logger.Info($"[OrtModelBase] CPU fallback session created for {modelName}");
                    }
                }
                else
                {
                    _session = new InferenceSession(modelPath, options);
                }
            });
            OrtLoadQueue.Enqueue(task, modelName);
            await task;

            _profilingEnabled = (ep == ExecutionProvider.QNN_HTP);
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

        private static SessionOptions CreateSessionOptions(
            ExecutionProvider ep, bool mobileOptimized, string modelCacheDir = null)
        {
            var options = new SessionOptions
            {
                LogSeverityLevel = (ep == ExecutionProvider.QNN_HTP)
                    ? OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
                    : OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
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
                    ConfigureCoreML(options, modelCacheDir);
                    break;
                case ExecutionProvider.QNN_HTP:
                    ConfigureQnn(options);
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
        /// Returns a per-model cache subdirectory keyed by file size. Different model
        /// precisions (FP32/INT8) get isolated cache dirs automatically, preventing
        /// stale compiled-model collisions when models share the same filename.
        /// Also cleans up old cache dirs for the same model name with different sizes.
        /// </summary>
        private static string GetPerModelCacheDir(string modelPath, string modelName)
        {
            long fileSize = new FileInfo(modelPath).Length;
            string baseDir = GetCoreMLCacheDirectory();
            string modelCacheDir = Path.Combine(baseDir, $"{modelName}_{fileSize}");

            if (!Directory.Exists(modelCacheDir))
            {
                if (Directory.Exists(baseDir))
                {
                    foreach (var oldDir in Directory.GetDirectories(baseDir, $"{modelName}_*"))
                    {
                        if (oldDir != modelCacheDir)
                        {
                            Logger.Info($"[OrtModelBase] Removing stale CoreML cache: {Path.GetFileName(oldDir)}");
                            try { Directory.Delete(oldDir, true); }
                            catch (Exception e) { Logger.Warning($"[OrtModelBase] Cleanup failed: {e.Message}"); }
                        }
                    }
                }
                Directory.CreateDirectory(modelCacheDir);
            }

            return modelCacheDir;
        }

        /// <summary>
        /// Configures CoreML with MLProgram format, GPU compute, model caching,
        /// and cache corruption recovery (LiveTalk pattern).
        /// </summary>
        private static void ConfigureCoreML(SessionOptions options, string modelCacheDir = null)
        {
            string cacheDir = modelCacheDir ?? GetCoreMLCacheDirectory();
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

        private static void ConfigureQnn(SessionOptions options)
        {
            // Bump env log level to INFO so QNN EP decisions are visible
            InitializeOrtLogging(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);

            QnnEnvironment.Initialize();

            string backendPath = "libQnnHtp.so";
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!string.IsNullOrEmpty(QnnEnvironment.NativeLibDir))
                backendPath = System.IO.Path.Combine(QnnEnvironment.NativeLibDir, "libQnnHtp.so");
#endif

            var qnnOptions = new Dictionary<string, string>
            {
                ["backend_path"] = backendPath,
                ["htp_performance_mode"] = "burst",
                ["enable_htp_fp16_precision"] = "1",
            };

            string profileDir = Path.Combine(Application.persistentDataPath, "ort_profiles");
            Directory.CreateDirectory(profileDir);
            string profilePrefix = Path.Combine(profileDir, "qnn_");
            options.ProfileOutputPathPrefix = profilePrefix;
            options.EnableProfiling = true;
            Logger.Info($"[OrtModelBase] QNN profiling enabled, prefix={profilePrefix}");

            options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");

            Logger.Info($"[OrtModelBase] QNN HTP backend_path={backendPath} (CPU fallback disabled)");
            options.AppendExecutionProvider("QNN", qnnOptions);
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
                var logLevel = _profilingEnabled
                    ? OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
                    : OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                var runOptions = new RunOptions { LogSeverityLevel = logLevel };
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
                var logLevel = _profilingEnabled
                    ? OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
                    : OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                var runOptions = new RunOptions { LogSeverityLevel = logLevel };
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
                if (_profilingEnabled && _session != null)
                {
                    try
                    {
                        string profileFile = _session.EndProfiling();
                        Logger.Info($"[OrtModelBase] QNN profile written: {profileFile}");
                    }
                    catch (Exception e)
                    {
                        Logger.Warning($"[OrtModelBase] EndProfiling failed: {e.Message}");
                    }
                }
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
