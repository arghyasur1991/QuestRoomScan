using System;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// GPU buffer set for a single sector's Gaussian parameters + optimizer state.
    /// All buffers stay on GPU — no CPU readback during training.
    /// </summary>
    public class GSplatBuffers : IDisposable
    {
        public int MaxGaussians { get; }
        public int CurrentCount { get; set; }
        public int SHDegree { get; }
        public int SHRestSize { get; }

        public GraphicsBuffer Means;        // [max * 3]
        public GraphicsBuffer Scales;       // [max * 3] (log-space)
        public GraphicsBuffer Quats;        // [max * 4]
        public GraphicsBuffer Opacities;    // [max]
        public GraphicsBuffer FeaturesDC;   // [max * 3]
        public GraphicsBuffer FeaturesRest; // [max * restSize]

        // Gradient accumulators (zeroed each iter)
        public GraphicsBuffer GradMeans;
        public GraphicsBuffer GradScales;
        public GraphicsBuffer GradQuats;
        public GraphicsBuffer GradOpacities;
        public GraphicsBuffer GradColors;   // [max * 3] v_colors from rasterize backward

        // Adam optimizer state
        public GraphicsBuffer AdamMeanM, AdamMeanV;
        public GraphicsBuffer AdamScaleM, AdamScaleV;
        public GraphicsBuffer AdamQuatM, AdamQuatV;
        public GraphicsBuffer AdamOpacM, AdamOpacV;
        public GraphicsBuffer AdamDCM, AdamDCV;
        public GraphicsBuffer AdamRestM, AdamRestV;

        // Densification stats
        public GraphicsBuffer VisCounts;
        public GraphicsBuffer XYGradNorm;
        public GraphicsBuffer Max2DSize;

        // Counter (for atomic init)
        public GraphicsBuffer CountBuffer;

        public GSplatBuffers(int maxGaussians, int shDegree = 2)
        {
            MaxGaussians = maxGaussians;
            SHDegree = shDegree;
            int numBases = NumSHBases(shDegree);
            SHRestSize = (numBases - 1) * 3;

            const GraphicsBuffer.Target s = GraphicsBuffer.Target.Structured;

            Means        = new GraphicsBuffer(s, maxGaussians * 3, 4);
            Scales       = new GraphicsBuffer(s, maxGaussians * 3, 4);
            Quats        = new GraphicsBuffer(s, maxGaussians * 4, 4);
            Opacities    = new GraphicsBuffer(s, maxGaussians, 4);
            FeaturesDC   = new GraphicsBuffer(s, maxGaussians * 3, 4);
            FeaturesRest = new GraphicsBuffer(s, maxGaussians * SHRestSize, 4);

            GradMeans     = new GraphicsBuffer(s, maxGaussians * 3, 4);
            GradScales    = new GraphicsBuffer(s, maxGaussians * 3, 4);
            GradQuats     = new GraphicsBuffer(s, maxGaussians * 4, 4);
            GradOpacities = new GraphicsBuffer(s, maxGaussians, 4);
            GradColors    = new GraphicsBuffer(s, maxGaussians * 3, 4);

            AdamMeanM  = new GraphicsBuffer(s, maxGaussians * 3, 4);
            AdamMeanV  = new GraphicsBuffer(s, maxGaussians * 3, 4);
            AdamScaleM = new GraphicsBuffer(s, maxGaussians * 3, 4);
            AdamScaleV = new GraphicsBuffer(s, maxGaussians * 3, 4);
            AdamQuatM  = new GraphicsBuffer(s, maxGaussians * 4, 4);
            AdamQuatV  = new GraphicsBuffer(s, maxGaussians * 4, 4);
            AdamOpacM  = new GraphicsBuffer(s, maxGaussians, 4);
            AdamOpacV  = new GraphicsBuffer(s, maxGaussians, 4);
            AdamDCM    = new GraphicsBuffer(s, maxGaussians * 3, 4);
            AdamDCV    = new GraphicsBuffer(s, maxGaussians * 3, 4);
            AdamRestM  = new GraphicsBuffer(s, maxGaussians * SHRestSize, 4);
            AdamRestV  = new GraphicsBuffer(s, maxGaussians * SHRestSize, 4);

            VisCounts  = new GraphicsBuffer(s, maxGaussians, 4);
            XYGradNorm = new GraphicsBuffer(s, maxGaussians, 4);
            Max2DSize  = new GraphicsBuffer(s, maxGaussians, 4);

            CountBuffer = new GraphicsBuffer(s, 1, 4);

            ZeroAll();
        }

        public void ZeroAll()
        {
            ZeroBuf(Means);     ZeroBuf(Scales);     ZeroBuf(Quats);
            ZeroBuf(Opacities); ZeroBuf(FeaturesDC); ZeroBuf(FeaturesRest);
            ZeroGrads();
            ZeroBuf(AdamMeanM); ZeroBuf(AdamMeanV);
            ZeroBuf(AdamScaleM); ZeroBuf(AdamScaleV);
            ZeroBuf(AdamQuatM); ZeroBuf(AdamQuatV);
            ZeroBuf(AdamOpacM); ZeroBuf(AdamOpacV);
            ZeroBuf(AdamDCM); ZeroBuf(AdamDCV);
            ZeroBuf(AdamRestM); ZeroBuf(AdamRestV);
            ZeroBuf(VisCounts); ZeroBuf(XYGradNorm); ZeroBuf(Max2DSize);
            ZeroBuf(CountBuffer);
            CurrentCount = 0;
        }

        public void ZeroGrads()
        {
            ZeroBuf(GradMeans); ZeroBuf(GradScales); ZeroBuf(GradQuats);
            ZeroBuf(GradOpacities); ZeroBuf(GradColors);
        }

        public void Dispose()
        {
            Means?.Release();      Scales?.Release();      Quats?.Release();
            Opacities?.Release();  FeaturesDC?.Release();  FeaturesRest?.Release();
            GradMeans?.Release();  GradScales?.Release();  GradQuats?.Release();
            GradOpacities?.Release(); GradColors?.Release();
            AdamMeanM?.Release();  AdamMeanV?.Release();
            AdamScaleM?.Release(); AdamScaleV?.Release();
            AdamQuatM?.Release();  AdamQuatV?.Release();
            AdamOpacM?.Release();  AdamOpacV?.Release();
            AdamDCM?.Release();    AdamDCV?.Release();
            AdamRestM?.Release();  AdamRestV?.Release();
            VisCounts?.Release();  XYGradNorm?.Release();  Max2DSize?.Release();
            CountBuffer?.Release();
        }

        static void ZeroBuf(GraphicsBuffer buf)
        {
            if (buf == null) return;
            var zeros = new byte[buf.count * buf.stride];
            buf.SetData(zeros);
        }

        static int NumSHBases(int degree)
        {
            return degree switch { 0 => 1, 1 => 4, 2 => 9, 3 => 16, _ => 25 };
        }
    }
}
