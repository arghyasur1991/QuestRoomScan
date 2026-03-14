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
            };
        }

        readonly GSplatForwardPass _forward;
        readonly ComputeShader _lossCS;
        readonly ComputeShader _rasterBwdCS;
        readonly ComputeShader _projBwdCS;
        readonly ComputeShader _adamCS;

        readonly int _kLossFwd, _kLossBwd;
        readonly int _kRastBwd;
        readonly int _kProjBwd;
        readonly int _kAdam, _kGradStats, _kZero;

        GraphicsBuffer _lossBuffer;
        GraphicsBuffer _intermediates;
        RenderTexture _vRendered;

        readonly Config _config;
        int _step;
        int _imgW, _imgH;

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

        public GSplatTrainer(ComputeShader projectSH, ComputeShader tileSort,
                             ComputeShader rasterize, ComputeShader lossCS,
                             ComputeShader rasterBwdCS, ComputeShader projBwdCS,
                             ComputeShader adamCS,
                             int maxPoints, int imgW, int imgH,
                             Config? config = null)
        {
            _config = config ?? Config.Default;
            _imgW = imgW;
            _imgH = imgH;

            _forward = new GSplatForwardPass(projectSH, tileSort, rasterize, maxPoints, imgW, imgH);

            _lossCS = lossCS;
            _kLossFwd = lossCS.FindKernel("LossForward");
            _kLossBwd = lossCS.FindKernel("LossBackward");

            _rasterBwdCS = rasterBwdCS;
            _kRastBwd = rasterBwdCS.FindKernel("RasterizeBackward");

            _projBwdCS = projBwdCS;
            _kProjBwd = projBwdCS.FindKernel("ProjectSHBackward");

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
        }

        /// <summary>
        /// Run one complete training iteration against a keyframe.
        /// </summary>
        public void TrainStep(GSplatBuffers gaussians, Texture2D keyframe,
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

            // 4. Zero gradient accumulators
            gaussians.ZeroGrads();

            // 5. Rasterize backward
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
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VXY", gaussians.GradMeans);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VConic", gaussians.GradScales);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VRGB", gaussians.GradColors);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VOpacity", gaussians.GradOpacities);
            _rasterBwdCS.SetBuffer(_kRastBwd, "_VDepth", gaussians.GradQuats);

            int rGroupsX = CeilDiv(_imgW, 8);
            int rGroupsY = CeilDiv(_imgH, 8);
            _rasterBwdCS.Dispatch(_kRastBwd, rGroupsX, rGroupsY, 1);

            // 6. Projection + SH backward (with fused SH Adam)
            _projBwdCS.SetInt(ID_NumPoints, N);
            _projBwdCS.SetFloat("_GlobScale", _config.GlobScale);
            _projBwdCS.SetInt("_Degree", _config.SHDegree);
            _projBwdCS.SetInt("_DegreesToUse", _config.SHDegreesToUse);
            _projBwdCS.SetInts(ID_ImgSize, _imgW, _imgH);
            _projBwdCS.SetVector("_Intrinsics", new Vector4(fx, fy, cx, cy));
            _projBwdCS.SetVector("_CamPos", camPos);
            _projBwdCS.SetMatrix("_ViewMat", viewMat);
            _projBwdCS.SetMatrix("_ProjMat", projMat);
            _projBwdCS.SetBuffer(_kProjBwd, "_Means3D", gaussians.Means);
            _projBwdCS.SetBuffer(_kProjBwd, "_Scales", gaussians.Scales);
            _projBwdCS.SetBuffer(_kProjBwd, "_Quats", gaussians.Quats);
            _projBwdCS.SetBuffer(_kProjBwd, "_Radii", _forward.Radii);
            _projBwdCS.SetBuffer(_kProjBwd, "_Conics", _forward.Conics);
            _projBwdCS.SetBuffer(_kProjBwd, "_VXY", gaussians.GradMeans);
            _projBwdCS.SetBuffer(_kProjBwd, "_VDepth", gaussians.GradQuats);
            _projBwdCS.SetBuffer(_kProjBwd, "_VConic", gaussians.GradScales);
            _projBwdCS.SetBuffer(_kProjBwd, "_VColors", gaussians.GradColors);
            _projBwdCS.SetBuffer(_kProjBwd, "_VMean3D", gaussians.GradMeans);
            _projBwdCS.SetBuffer(_kProjBwd, "_VScale", gaussians.GradScales);
            _projBwdCS.SetBuffer(_kProjBwd, "_VQuat", gaussians.GradQuats);
            _projBwdCS.SetBuffer(_kProjBwd, "_FeaturesDC", gaussians.FeaturesDC);
            _projBwdCS.SetBuffer(_kProjBwd, "_FeaturesRest", gaussians.FeaturesRest);
            _projBwdCS.SetBuffer(_kProjBwd, "_DCExpAvg", gaussians.AdamDCM);
            _projBwdCS.SetBuffer(_kProjBwd, "_DCExpAvgSq", gaussians.AdamDCV);
            _projBwdCS.SetBuffer(_kProjBwd, "_RestExpAvg", gaussians.AdamRestM);
            _projBwdCS.SetBuffer(_kProjBwd, "_RestExpAvgSq", gaussians.AdamRestV);
            float dcStep = _config.LRDC / (1f - Mathf.Pow(_config.AdamBeta1, _step));
            float restStep = _config.LRRest / (1f - Mathf.Pow(_config.AdamBeta1, _step));
            _projBwdCS.SetFloat("_DCStepSize", dcStep);
            _projBwdCS.SetFloat("_DCBc2Sqrt", bc2);
            _projBwdCS.SetFloat("_RestStepSize", restStep);
            _projBwdCS.SetFloat("_RestBc2Sqrt", bc2);
            _projBwdCS.SetFloat("_AdamBeta1", _config.AdamBeta1);
            _projBwdCS.SetFloat("_AdamBeta2", _config.AdamBeta2);
            _projBwdCS.SetFloat("_AdamEps", _config.AdamEps);
            _projBwdCS.Dispatch(_kProjBwd, CeilDiv(N, 256), 1, 1);

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

        public void Dispose()
        {
            _forward?.Dispose();
            _lossBuffer?.Release();
            _intermediates?.Release();
            if (_vRendered) UnityEngine.Object.Destroy(_vRendered);
        }

        static int CeilDiv(int a, int b) => (a + b - 1) / b;
    }
}
