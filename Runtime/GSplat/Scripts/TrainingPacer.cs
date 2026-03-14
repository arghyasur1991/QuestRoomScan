using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Manages training pace: thermal throttling, quality-based termination,
    /// and iteration budgeting per frame.
    /// </summary>
    public class TrainingPacer
    {
        public struct Config
        {
            public float MaxFrameTimeMs;
            public float ThermalThrottleTemp;
            public float QualityThresholdPSNR;
            public float MinLossImprovementRate;
            public int MinIterations;
            public int MaxIterations;
            public int DensifyEvery;
            public int DensifyStartIter;
            public int DensifyStopIter;

            public static Config Default => new()
            {
                MaxFrameTimeMs = 10f,
                ThermalThrottleTemp = 42f,
                QualityThresholdPSNR = 25f,
                MinLossImprovementRate = 0.001f,
                MinIterations = 100,
                MaxIterations = 500,
                DensifyEvery = 100,
                DensifyStartIter = 50,
                DensifyStopIter = 300
            };
        }

        readonly Config _config;
        float _lastLoss = float.MaxValue;
        float _lossEMA = float.MaxValue;
        int _stagnantFrames;
        float _lastFrameTimeMs;

        public int RecommendedIters { get; private set; } = 3;
        public bool ShouldDensify { get; private set; }
        public bool ShouldTerminate { get; private set; }

        public TrainingPacer(Config? config = null)
        {
            _config = config ?? Config.Default;
        }

        /// <summary>
        /// Call each frame before training. Returns how many iterations to run.
        /// </summary>
        public int UpdatePace(int currentIter, float currentLoss, float frameTimeMs, float deviceTempC)
        {
            _lastFrameTimeMs = frameTimeMs;

            // Thermal throttling
            if (deviceTempC > _config.ThermalThrottleTemp)
            {
                RecommendedIters = 0;
                return 0;
            }

            // Frame time budget: reduce iters if we're over budget
            if (frameTimeMs > _config.MaxFrameTimeMs)
                RecommendedIters = Mathf.Max(1, RecommendedIters - 1);
            else if (frameTimeMs < _config.MaxFrameTimeMs * 0.7f)
                RecommendedIters = Mathf.Min(10, RecommendedIters + 1);

            // Quality termination
            ShouldTerminate = false;
            if (currentIter >= _config.MaxIterations)
            {
                ShouldTerminate = true;
            }
            else if (currentIter >= _config.MinIterations)
            {
                _lossEMA = _lossEMA == float.MaxValue
                    ? currentLoss
                    : 0.9f * _lossEMA + 0.1f * currentLoss;

                float improvement = (_lastLoss - currentLoss) / Mathf.Max(1e-8f, _lastLoss);
                if (improvement < _config.MinLossImprovementRate)
                    _stagnantFrames++;
                else
                    _stagnantFrames = 0;

                if (_stagnantFrames > 10)
                    ShouldTerminate = true;
            }

            _lastLoss = currentLoss;

            // Densification schedule
            ShouldDensify = currentIter >= _config.DensifyStartIter &&
                            currentIter <= _config.DensifyStopIter &&
                            currentIter % _config.DensifyEvery == 0;

            return RecommendedIters;
        }

        public void Reset()
        {
            _lastLoss = float.MaxValue;
            _lossEMA = float.MaxValue;
            _stagnantFrames = 0;
            RecommendedIters = 3;
            ShouldDensify = false;
            ShouldTerminate = false;
        }
    }
}
