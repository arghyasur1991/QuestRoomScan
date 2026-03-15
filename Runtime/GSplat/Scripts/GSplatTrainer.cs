using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Runs a training iteration: forward pass → loss → backward pass → optimizer step.
    /// Designed for time-sliced execution (N iters per Update frame).
    /// </summary>
    public class GSplatTrainer : IDisposable
    {
        public struct Config
        {
            public int SHDegree;
            public int SHDegreesToUse;
            public float GlobScale;
            public float ClipThresh;
            public float SSIMWeight;
            public float LRMeans;
            public float LRScales;
            public float LRQuats;
            public float LROpacity;
            public float LRDC;
            public float LRRest;
            public float AdamBeta1;
            public float AdamBeta2;
            public float AdamEps;

            public float DensifyGradThresh;
            public float DensifySizeThresh;
            public float DensifyScreenThresh;
            public float DensifyCullAlpha;
            public float DensifyCullScale;
            public float DensifyCullScreenSize;

            public static Config Default => new()
            {
                SHDegree = 2,
                SHDegreesToUse = 2,
                GlobScale = 1f,
                ClipThresh = 0.01f,
                SSIMWeight = 0.2f,
                LRMeans = 0.00016f,
                LRScales = 0.005f,
                LRQuats = 0.001f,
                LROpacity = 0.05f,
                LRDC = 0.0025f,
                LRRest = 0.000125f,
                AdamBeta1 = 0.9f,
                AdamBeta2 = 0.999f,
                AdamEps = 1e-15f,
                DensifyGradThresh = 0.0002f,
                DensifySizeThresh = 0.01f,
                DensifyScreenThresh = 20f,
                DensifyCullAlpha = 0.005f,
                DensifyCullScale = 0.5f,
                DensifyCullScreenSize = 40f,
            };
        }

        readonly GSplatForwardPass _forward;
        readonly ComputeShader _lossCS;
        readonly ComputeShader _rasterBwdCS;
        readonly ComputeShader _projBwdCS;
        readonly ComputeShader _adamCS;
        readonly ComputeShader _densifyCS;

        readonly int _kLossFwd, _kLossBwd;
        readonly int _kRastBwd;
        readonly int _kProjBwd, _kSHBwd;
        readonly int _kAdam, _kGradStats, _kZero;
        int _kDensifyClassify, _kAppendSplit, _kAppendDup, _kCullClassify;
        int _kPrefixSum, _kCompactScatter, _kCompactCopyBack;

        GraphicsBuffer _lossBuffer;
        GraphicsBuffer _intermediates;
        RenderTexture _vRendered;

        GraphicsBuffer _vxy;
        GraphicsBuffer _vConic;
        GraphicsBuffer _vDepth;

        // Densification scratch buffers
        GraphicsBuffer _splitFlag, _dupFlag, _splitPrefix, _dupPrefix;
        GraphicsBuffer _keepFlag, _keepPrefix;
        GraphicsBuffer _randomSamples;
        GraphicsBuffer _compactTemp;

        readonly Config _config;
        int _step;
        int _imgW, _imgH;
        int _maxPoints;
        readonly int[] _readback1 = new int[1];

        static readonly int ID_ImgSize = Shader.PropertyToID("_ImgSize");
        static readonly int ID_SSIMWeight = Shader.PropertyToID("_SSIMWeight");
        static readonly int ID_InvN = Shader.PropertyToID("_InvN");
        static readonly int ID_NumPoints = Shader.PropertyToID("_NumPoints");
        static readonly int ID_NumPoints2 = Shader.PropertyToID("_NumPoints2");
        static readonly int ID_N = Shader.PropertyToID("_N");
        static readonly int ID_StepSize = Shader.PropertyToID("_StepSize");
        static readonly int ID_Beta1 = Shader.PropertyToID("_Beta1");
        static readonly int ID_Beta2 = Shader.PropertyToID("_Beta2");
        static readonly int ID_Bc2Sqrt = Shader.PropertyToID("_Bc2Sqrt");
        static readonly int ID_Eps = Shader.PropertyToID("_Eps");
        static readonly int ID_InvMaxDim = Shader.PropertyToID("_InvMaxDim");
        static readonly int ID_ZeroCount = Shader.PropertyToID("_ZeroCount");

        public int Step => _step;
        public RenderTexture RenderedImage => _forward.OutputImage;

        public void ResetStep() { _step = 0; }

        public GSplatTrainer(ComputeShader projectSH, ComputeShader tileSort,
                             ComputeShader rasterize, ComputeShader lossCS,
                             ComputeShader rasterBwdCS, ComputeShader projBwdCS,
                             ComputeShader adamCS, ComputeShader densifyCS,
                             int maxPoints, int imgW, int imgH,
                             Config? config = null)
        {
            _config = config ?? Config.Default;
            _imgW = imgW;
            _imgH = imgH;
            _maxPoints = maxPoints;

            _forward = new GSplatForwardPass(projectSH, tileSort, rasterize, maxPoints, imgW, imgH);

            _lossCS = lossCS;
            _kLossFwd = lossCS.FindKernel("LossForward");
            _kLossBwd = lossCS.FindKernel("LossBackward");

            _rasterBwdCS = rasterBwdCS;
            _kRastBwd = rasterBwdCS.FindKernel("RasterizeBackward");

            _projBwdCS = projBwdCS;
            _kProjBwd = projBwdCS.FindKernel("ProjectBackward");
            _kSHBwd = projBwdCS.FindKernel("SHBackwardAdam");

            _adamCS = adamCS;
            _kAdam = adamCS.FindKernel("AdamStep");
            _kGradStats = adamCS.FindKernel("AccumulateGradStats");
            _kZero = adamCS.FindKernel("ZeroBuffer");

            int numPixels = imgW * imgH;
            _lossBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numPixels, 4);
            _intermediates = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numPixels * 15, 4);
            _vRendered = new RenderTexture(imgW, imgH, 0, GraphicsFormat.R32G32B32A32_SFloat)
            {
                enableRandomWrite = true,
                filterMode = FilterMode.Point
            };
            _vRendered.Create();

            const GraphicsBuffer.Target s = GraphicsBuffer.Target.Structured;
            _vxy    = new GraphicsBuffer(s, maxPoints * 2, 4);
            _vConic = new GraphicsBuffer(s, maxPoints * 3, 4);
            _vDepth = new GraphicsBuffer(s, maxPoints, 4);

            // Densification
            _densifyCS = densifyCS;
            if (densifyCS != null)
            {
                _kDensifyClassify = densifyCS.FindKernel("DensifyClassify");
                _kAppendSplit = densifyCS.FindKernel("DensifyAppendSplit");
                _kAppendDup = densifyCS.FindKernel("DensifyAppendDup");
                _kCullClassify = densifyCS.FindKernel("DensifyCullClassify");
                _kPrefixSum = densifyCS.FindKernel("PrefixSum");
                _kCompactScatter = densifyCS.FindKernel("CompactScatter");
                _kCompactCopyBack = densifyCS.FindKernel("CompactCopyBack");

                _splitFlag    = new GraphicsBuffer(s, maxPoints, 4);
                _dupFlag      = new GraphicsBuffer(s, maxPoints, 4);
                _splitPrefix  = new GraphicsBuffer(s, maxPoints, 4);
                _dupPrefix    = new GraphicsBuffer(s, maxPoints, 4);
                _keepFlag     = new GraphicsBuffer(s, maxPoints, 4);
                _keepPrefix   = new GraphicsBuffer(s, maxPoints, 4);
                _randomSamples = new GraphicsBuffer(s, maxPoints * 6, 4);
                int maxStride = Mathf.Max(24, (_config.SHDegree > 0 ? (NumSHBases(_config.SHDegree) - 1) * 3 : 1));
                _compactTemp  = new GraphicsBuffer(s, maxPoints * maxStride, 4);
            }
        }

        static int NumSHBases(int d) => d switch { 0 => 1, 1 => 4, 2 => 9, 3 => 16, _ => 25 };

        /// <summary>
        /// Run one complete training iteration against a keyframe.
        /// </summary>
        public void TrainStep(GSplatBuffers gaussians, Texture keyframe,
                              Matrix4x4 viewMat, Matrix4x4 projMat,
                              float fx, float fy, float cx, float cy,
                              Vector3 camPos)
        {
            int N = gaussians.CurrentCount;
            if (N == 0) return;
            _step++;

            float bc2 = Mathf.Sqrt(1f - Mathf.Pow(_config.AdamBeta2, _step));

            // 1. Forward pass
            _forward.Execute(gaussians, viewMat, projMat, fx, fy, cx, cy,
                             camPos, _config.SHDegree, _config.SHDegreesToUse,
                             _config.GlobScale, _config.ClipThresh);

            // 2. Loss forward
            _lossCS.SetInts(ID_ImgSize, _imgW, _imgH);
            _lossCS.SetFloat(ID_SSIMWeight, _config.SSIMWeight);
            _lossCS.SetTexture(_kLossFwd, "_Rendered", _forward.OutputImage);
            _lossCS.SetTexture(_kLossFwd, "_GroundTruth", keyframe);
            _lossCS.SetBuffer(_kLossFwd, "_Intermediates", _intermediates);
            _lossCS.SetBuffer(_kLossFwd, "_LossSum", _lossBuffer);
            _lossCS.Dispatch(_kLossFwd, CeilDiv(_imgW, 8), CeilDiv(_imgH, 8), 1);

            // 3. Loss backward → dL/d(rendered)
            float invN = 1f / (_imgW * _imgH * 3f);
            _lossCS.SetFloat(ID_InvN, invN);
            _lossCS.SetTexture(_kLossBwd, "_Rendered", _forward.OutputImage);
            _lossCS.SetTexture(_kLossBwd, "_GroundTruth", keyframe);
            _lossCS.SetBuffer(_kLossBwd, "_Intermediates", _intermediates);
            _lossCS.SetTexture(_kLossBwd, "_VRendered", _vRendered);
            _lossCS.Dispatch(_kLossBwd, CeilDiv(_imgW, 8), CeilDiv(_imgH, 8), 1);

            // 4. Zero gradient accumulators + intermediate buffers (GPU dispatch, no CPU alloc)
            ZeroGradsGPU(gaussians);
            ZeroBufGPU(_vxy);
            ZeroBufGPU(_vConic);
            ZeroBufGPU(_vDepth);

            // 5. Rasterize backward → writes to SEPARATE intermediate buffers
            // (_vxy, _vConic, _vDepth) to avoid aliasing with projection backward
            _rasterBwdCS.SetInts("_TileBounds", (int)((uint)(_imgW + 15) / 16), (int)((uint)(_imgH + 15) / 16), 1);
            _rasterBwdCS.SetInts(ID_ImgSize, _imgW, _imgH);
            _rasterBwdCS.SetVector("_Background", Vector4.zero);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_TileBins", _forward.TileBins);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_GaussianIdsSorted", _forward.GaussianIdsOut);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_PackedXYOpac", _forward.PackedXYOpac);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_PackedConic", _forward.PackedConic);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_PackedRGB", _forward.PackedRGB);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_FinalTs", _forward.FinalTs);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_FinalIndex", _forward.FinalIndex);
            _rasterBwdCS.SetTexture(_kRastBwd, "_VOutput", _vRendered);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VXY", _vxy);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VConic", _vConic);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VRGB", gaussians.GradColors);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VOpacity", gaussians.GradOpacities);

            int rGroupsX = CeilDiv(_imgW, 8);
            int rGroupsY = CeilDiv(_imgH, 8);
            _rasterBwdCS.Dispatch(_kRastBwd, rGroupsX, rGroupsY, 1);

            // 6. Projection backward — reads intermediate buffers, writes to final grad buffers
            _projBwdCS.SetInt(ID_NumPoints, N);
            _projBwdCS.SetFloat("_GlobScale", _config.GlobScale);
            _projBwdCS.SetInt("_Degree", _config.SHDegree);
            _projBwdCS.SetInt("_DegreesToUse", _config.SHDegreesToUse);
            _projBwdCS.SetInts(ID_ImgSize, _imgW, _imgH);
            _projBwdCS.SetVector("_Intrinsics", new Vector4(fx, fy, cx, cy));
            _projBwdCS.SetVector("_CamPos", camPos);
            _projBwdCS.SetMatrix("_ViewMat", viewMat);
            _projBwdCS.SetMatrix("_ProjMat", projMat);
            // Kernel 1: ProjectBackward — reads from separate intermediates, writes to grad buffers
            _projBwdCS.SetBuffer(_kProjBwd, "_Means3D", gaussians.Means);
            _projBwdCS.SetBuffer(_kProjBwd, "_Scales", gaussians.Scales);
            _projBwdCS.SetBuffer(_kProjBwd, "_Quats", gaussians.Quats);
            _projBwdCS.SetBuffer(_kProjBwd, "_Radii", _forward.Radii);
            _projBwdCS.SetBuffer(_kProjBwd, "_Conics", _forward.Conics);
            _projBwdCS.SetBuffer(_kProjBwd, "_VXY", _vxy);             // READ from intermediate
            _projBwdCS.SetBuffer(_kProjBwd, "_VDepth", _vDepth);        // READ from intermediate
            _projBwdCS.SetBuffer(_kProjBwd, "_VConic", _vConic);        // READ from intermediate
            _projBwdCS.SetBuffer(_kProjBwd, "_VMean3D", gaussians.GradMeans);  // WRITE to final grad
            _projBwdCS.SetBuffer(_kProjBwd, "_VScale", gaussians.GradScales);  // WRITE to final grad
            _projBwdCS.SetBuffer(_kProjBwd, "_VQuat", gaussians.GradQuats);    // WRITE to final grad
            _projBwdCS.Dispatch(_kProjBwd, CeilDiv(N, 256), 1, 1);

            // Kernel 2: SHBackwardAdam (6 RW: FeaturesDC/Rest + Adam state)
            _projBwdCS.SetBuffer(_kSHBwd, "_Means3D", gaussians.Means);
            _projBwdCS.SetBuffer(_kSHBwd, "_Radii", _forward.Radii);
            _projBwdCS.SetBuffer(_kSHBwd, "_VColors", gaussians.GradColors);
            _projBwdCS.SetBuffer(_kSHBwd, "_FeaturesDC", gaussians.FeaturesDC);
            _projBwdCS.SetBuffer(_kSHBwd, "_FeaturesRest", gaussians.FeaturesRest);
            _projBwdCS.SetBuffer(_kSHBwd, "_DCExpAvg", gaussians.AdamDCM);
            _projBwdCS.SetBuffer(_kSHBwd, "_DCExpAvgSq", gaussians.AdamDCV);
            _projBwdCS.SetBuffer(_kSHBwd, "_RestExpAvg", gaussians.AdamRestM);
            _projBwdCS.SetBuffer(_kSHBwd, "_RestExpAvgSq", gaussians.AdamRestV);
            float dcStep = _config.LRDC / (1f - Mathf.Pow(_config.AdamBeta1, _step));
            float restStep = _config.LRRest / (1f - Mathf.Pow(_config.AdamBeta1, _step));
            _projBwdCS.SetFloat("_DCStepSize", dcStep);
            _projBwdCS.SetFloat("_DCBc2Sqrt", bc2);
            _projBwdCS.SetFloat("_RestStepSize", restStep);
            _projBwdCS.SetFloat("_RestBc2Sqrt", bc2);
            _projBwdCS.SetFloat("_AdamBeta1", _config.AdamBeta1);
            _projBwdCS.SetFloat("_AdamBeta2", _config.AdamBeta2);
            _projBwdCS.SetFloat("_AdamEps", _config.AdamEps);
            _projBwdCS.Dispatch(_kSHBwd, CeilDiv(N, 256), 1, 1);

            // 7. Adam for means, scales, quats, opacity
            RunAdam(gaussians.Means, gaussians.GradMeans,
                    gaussians.AdamMeanM, gaussians.AdamMeanV,
                    N * 3, _config.LRMeans, bc2);

            RunAdam(gaussians.Scales, gaussians.GradScales,
                    gaussians.AdamScaleM, gaussians.AdamScaleV,
                    N * 3, _config.LRScales, bc2);

            RunAdam(gaussians.Quats, gaussians.GradQuats,
                    gaussians.AdamQuatM, gaussians.AdamQuatV,
                    N * 4, _config.LRQuats, bc2);

            RunAdam(gaussians.Opacities, gaussians.GradOpacities,
                    gaussians.AdamOpacM, gaussians.AdamOpacV,
                    N, _config.LROpacity, bc2);

            // 8. Gradient stats for densification
            float invMaxDim = 1f / Mathf.Max(_imgW, _imgH);
            _adamCS.SetInt(ID_NumPoints2, N);
            _adamCS.SetFloat(ID_InvMaxDim, invMaxDim);
            _adamCS.SetBuffer(_kGradStats, "_Radii", _forward.Radii);
            _adamCS.SetBuffer(_kGradStats, "_XYGrad", gaussians.GradMeans);
            _adamCS.SetBuffer(_kGradStats, "_VisCounts", gaussians.VisCounts);
            _adamCS.SetBuffer(_kGradStats, "_XYGradNorm", gaussians.XYGradNorm);
            _adamCS.SetBuffer(_kGradStats, "_Max2DSize", gaussians.Max2DSize);
            _adamCS.Dispatch(_kGradStats, CeilDiv(N, 256), 1, 1);
        }

        void RunAdam(GraphicsBuffer param, GraphicsBuffer grad,
                     GraphicsBuffer m, GraphicsBuffer v, int n, float lr, float bc2)
        {
            float stepSize = lr / (1f - Mathf.Pow(_config.AdamBeta1, _step));
            _adamCS.SetInt(ID_N, n);
            _adamCS.SetFloat(ID_StepSize, stepSize);
            _adamCS.SetFloat(ID_Beta1, _config.AdamBeta1);
            _adamCS.SetFloat(ID_Beta2, _config.AdamBeta2);
            _adamCS.SetFloat(ID_Bc2Sqrt, bc2);
            _adamCS.SetFloat(ID_Eps, _config.AdamEps);
            _adamCS.SetBuffer(_kAdam, "_Params", param);
            _adamCS.SetBuffer(_kAdam, "_Grads", grad);
            _adamCS.SetBuffer(_kAdam, "_ExpAvg", m);
            _adamCS.SetBuffer(_kAdam, "_ExpAvgSq", v);
            _adamCS.Dispatch(_kAdam, CeilDiv(n, 256), 1, 1);
        }

        /// <summary>
        /// Run densification: classify → split/dup → cull → compact.
        /// Returns the new Gaussian count.
        /// </summary>
        public int Densify(GSplatBuffers gaussians)
        {
            if (_densifyCS == null) return gaussians.CurrentCount;

            int N = gaussians.CurrentCount;
            if (N == 0 || N > _maxPoints / 3) return N;

            float halfMaxDim = 0.5f * Mathf.Max(_imgW, _imgH);

            // 1. Classify: split vs dup vs nothing
            _densifyCS.SetInt("_N", N);
            _densifyCS.SetFloat("_GradThresh", _config.DensifyGradThresh);
            _densifyCS.SetFloat("_SizeThresh", _config.DensifySizeThresh);
            _densifyCS.SetFloat("_ScreenThresh", _config.DensifyScreenThresh);
            _densifyCS.SetFloat("_HalfMaxDim", halfMaxDim);
            _densifyCS.SetInt("_CheckScreen", 1);
            _densifyCS.SetBuffer(_kDensifyClassify, "_XYGradNorm", gaussians.XYGradNorm);
            _densifyCS.SetBuffer(_kDensifyClassify, "_VisCounts", gaussians.VisCounts);
            _densifyCS.SetBuffer(_kDensifyClassify, "_Scales", gaussians.Scales);
            _densifyCS.SetBuffer(_kDensifyClassify, "_Max2DSize", gaussians.Max2DSize);
            _densifyCS.SetBuffer(_kDensifyClassify, "_SplitFlag", _splitFlag);
            _densifyCS.SetBuffer(_kDensifyClassify, "_DupFlag", _dupFlag);
            _densifyCS.Dispatch(_kDensifyClassify, CeilDiv(N, 256), 1, 1);

            // 2–3. Prefix sums
            RunPrefixSum(_splitFlag, _splitPrefix, N);
            RunPrefixSum(_dupFlag, _dupPrefix, N);

            // 4. Read back counts (GPU sync — acceptable at densify cadence)
            _splitPrefix.GetData(_readback1, 0, N - 1, 1);
            int splitCount = _readback1[0];
            _dupPrefix.GetData(_readback1, 0, N - 1, 1);
            int dupCount = _readback1[0];

            int nNew = N + 2 * splitCount + dupCount;
            if (nNew > _maxPoints)
            {
                Debug.LogWarning($"[GSplat] Densify overflow ({nNew} > {_maxPoints}), skipping");
                ResetDensifyStats(gaussians);
                return N;
            }

            // 5. Upload random normal samples for split children (Box-Muller)
            if (splitCount > 0)
            {
                int numR = splitCount * 2 * 3;
                var rands = new float[numR];
                for (int i = 0; i < numR; i += 2)
                {
                    float u1 = UnityEngine.Random.Range(1e-6f, 1f);
                    float u2 = UnityEngine.Random.Range(0f, 2f * Mathf.PI);
                    float mag = Mathf.Sqrt(-2f * Mathf.Log(u1));
                    rands[i] = mag * Mathf.Cos(u2);
                    if (i + 1 < numR) rands[i + 1] = mag * Mathf.Sin(u2);
                }
                _randomSamples.SetData(rands, 0, 0, numR);
            }

            // 6. Append split children
            if (splitCount > 0)
            {
                _densifyCS.SetInt("_N", N);
                _densifyCS.SetFloat("_LogSizeFac", Mathf.Log(1.6f));
                _densifyCS.SetBuffer(_kAppendSplit, "_SplitFlag", _splitFlag);
                _densifyCS.SetBuffer(_kAppendSplit, "_SplitPrefix", _splitPrefix);
                _densifyCS.SetBuffer(_kAppendSplit, "_RandomSamples", _randomSamples);
                _densifyCS.SetBuffer(_kAppendSplit, "_MeansBuf", gaussians.Means);
                _densifyCS.SetBuffer(_kAppendSplit, "_ScalesBuf", gaussians.Scales);
                _densifyCS.SetBuffer(_kAppendSplit, "_QuatsBuf", gaussians.Quats);
                _densifyCS.SetBuffer(_kAppendSplit, "_FeaturesDCBuf", gaussians.FeaturesDC);
                _densifyCS.SetBuffer(_kAppendSplit, "_OpacitiesBuf", gaussians.Opacities);
                _densifyCS.Dispatch(_kAppendSplit, CeilDiv(N, 256), 1, 1);
            }

            // 7. Append duplicates
            if (dupCount > 0)
            {
                _densifyCS.SetInt("_N", N);
                _densifyCS.SetInt("_NSplits", splitCount);
                _densifyCS.SetBuffer(_kAppendDup, "_DupFlag", _dupFlag);
                _densifyCS.SetBuffer(_kAppendDup, "_DupPrefix", _dupPrefix);
                _densifyCS.SetBuffer(_kAppendDup, "_MeansBuf", gaussians.Means);
                _densifyCS.SetBuffer(_kAppendDup, "_ScalesBuf", gaussians.Scales);
                _densifyCS.SetBuffer(_kAppendDup, "_QuatsBuf", gaussians.Quats);
                _densifyCS.SetBuffer(_kAppendDup, "_FeaturesDCBuf", gaussians.FeaturesDC);
                _densifyCS.SetBuffer(_kAppendDup, "_OpacitiesBuf", gaussians.Opacities);
                _densifyCS.Dispatch(_kAppendDup, CeilDiv(N, 256), 1, 1);
            }

            // 8. Cull classify
            _densifyCS.SetInt("_NOld", N);
            _densifyCS.SetFloat("_CullAlphaThresh", _config.DensifyCullAlpha);
            _densifyCS.SetFloat("_CullScaleThresh", _config.DensifyCullScale);
            _densifyCS.SetFloat("_CullScreenSize", _config.DensifyCullScreenSize);
            _densifyCS.SetInt("_CheckHuge", 1);
            _densifyCS.SetBuffer(_kCullClassify, "_SplitPrefix", _splitPrefix);
            _densifyCS.SetBuffer(_kCullClassify, "_DupPrefix", _dupPrefix);
            _densifyCS.SetBuffer(_kCullClassify, "_SplitFlag", _splitFlag);
            _densifyCS.SetBuffer(_kCullClassify, "_OpacitiesBuf", gaussians.Opacities);
            _densifyCS.SetBuffer(_kCullClassify, "_ScalesBuf", gaussians.Scales);
            _densifyCS.SetBuffer(_kCullClassify, "_Max2DSize", gaussians.Max2DSize);
            _densifyCS.SetBuffer(_kCullClassify, "_KeepFlag", _keepFlag);
            _densifyCS.Dispatch(_kCullClassify, CeilDiv(nNew, 256), 1, 1);

            // 9. Prefix sum of keep flags
            RunPrefixSum(_keepFlag, _keepPrefix, nNew);

            // 10. Compact each parameter buffer
            CompactBuffer(gaussians.Means, nNew, 3);
            CompactBuffer(gaussians.Scales, nNew, 3);
            CompactBuffer(gaussians.Quats, nNew, 4);
            CompactBuffer(gaussians.Opacities, nNew, 1);
            CompactBuffer(gaussians.FeaturesDC, nNew, 3);
            if (gaussians.SHRestSize > 0)
                CompactBuffer(gaussians.FeaturesRest, nNew, gaussians.SHRestSize);

            // 11. Read back final count
            _keepPrefix.GetData(_readback1, 0, nNew - 1, 1);
            int finalCount = Mathf.Min(_readback1[0], _maxPoints);
            gaussians.CurrentCount = finalCount;

            Debug.Log($"[GSplat] Densify: {N} → {nNew} (split={splitCount} dup={dupCount}) → {finalCount} after cull");

            // 12. Reset Adam state (indices changed) and densify stats
            ResetAdamState(gaussians);
            ResetDensifyStats(gaussians);

            return finalCount;
        }

        public void ResetDensifyStats(GSplatBuffers g)
        {
            ZeroBufGPU(g.VisCounts);
            ZeroBufGPU(g.XYGradNorm);
            ZeroBufGPU(g.Max2DSize);
        }

        void ResetAdamState(GSplatBuffers g)
        {
            ZeroBufGPU(g.AdamMeanM); ZeroBufGPU(g.AdamMeanV);
            ZeroBufGPU(g.AdamScaleM); ZeroBufGPU(g.AdamScaleV);
            ZeroBufGPU(g.AdamQuatM); ZeroBufGPU(g.AdamQuatV);
            ZeroBufGPU(g.AdamOpacM); ZeroBufGPU(g.AdamOpacV);
            ZeroBufGPU(g.AdamDCM); ZeroBufGPU(g.AdamDCV);
            ZeroBufGPU(g.AdamRestM); ZeroBufGPU(g.AdamRestV);
        }

        void RunPrefixSum(GraphicsBuffer input, GraphicsBuffer output, int n)
        {
            _densifyCS.SetInt("_PrefixN", n);
            _densifyCS.SetBuffer(_kPrefixSum, "_PrefixInput", input);
            _densifyCS.SetBuffer(_kPrefixSum, "_PrefixOutput", output);
            _densifyCS.Dispatch(_kPrefixSum, 1, 1, 1);
        }

        void CompactBuffer(GraphicsBuffer buf, int nNew, int stride)
        {
            _densifyCS.SetInt("_CompactN", nNew);
            _densifyCS.SetInt("_CompactStride", stride);
            _densifyCS.SetBuffer(_kCompactScatter, "_CompactSrc", buf);
            _densifyCS.SetBuffer(_kCompactScatter, "_CompactDst", _compactTemp);
            _densifyCS.SetBuffer(_kCompactScatter, "_KeepPrefix", _keepPrefix);
            _densifyCS.SetBuffer(_kCompactScatter, "_KeepFlagRead", _keepFlag);
            _densifyCS.Dispatch(_kCompactScatter, CeilDiv(nNew * stride, 256), 1, 1);

            _densifyCS.SetInt("_CopyStride", stride);
            _densifyCS.SetInt("_LastPrefixIdx", nNew - 1);
            _densifyCS.SetBuffer(_kCompactCopyBack, "_CopySrc", _compactTemp);
            _densifyCS.SetBuffer(_kCompactCopyBack, "_CopyDst", buf);
            _densifyCS.SetBuffer(_kCompactCopyBack, "_KeepPrefix", _keepPrefix);
            _densifyCS.Dispatch(_kCompactCopyBack, CeilDiv(nNew * stride, 256), 1, 1);
        }

        public void Dispose()
        {
            _forward?.Dispose();
            _lossBuffer?.Release();
            _intermediates?.Release();
            _vxy?.Release();
            _vConic?.Release();
            _vDepth?.Release();
            _splitFlag?.Release();
            _dupFlag?.Release();
            _splitPrefix?.Release();
            _dupPrefix?.Release();
            _keepFlag?.Release();
            _keepPrefix?.Release();
            _randomSamples?.Release();
            _compactTemp?.Release();
            if (_vRendered) UnityEngine.Object.Destroy(_vRendered);
        }

        static int CeilDiv(int a, int b) => (a + b - 1) / b;

        void ZeroBufGPU(GraphicsBuffer buf)
        {
            if (buf == null) return;
            int count = buf.count * buf.stride / 4;
            _adamCS.SetInt(ID_ZeroCount, count);
            _adamCS.SetBuffer(_kZero, "_ZeroBuf", buf);
            _adamCS.Dispatch(_kZero, CeilDiv(count, 256), 1, 1);
        }

        void ZeroGradsGPU(GSplatBuffers g)
        {
            ZeroBufGPU(g.GradMeans);
            ZeroBufGPU(g.GradScales);
            ZeroBufGPU(g.GradQuats);
            ZeroBufGPU(g.GradOpacities);
            ZeroBufGPU(g.GradColors);
        }
    }
}
