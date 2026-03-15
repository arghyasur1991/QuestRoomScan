using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// URP Renderer Feature matching UGS architecture:
    /// 1. Create R16G16B16A16_SFloat intermediate RT (cleared to zero)
    /// 2. Prepass + sort + draw splats front-to-back into intermediate RT
    ///    (Blend OneMinusDstAlpha One)
    /// 3. Composite intermediate RT onto the scene per-eye
    ///    (Blend SrcAlpha OneMinusSrcAlpha + GammaToLinear)
    /// Per-eye VP matrices are captured at render time (after XR late-latching).
    /// </summary>
    public class GSplatRenderFeature : ScriptableRendererFeature
    {
        class GSplatPass : ScriptableRenderPass
        {
            const string GaussianSplatRTName = "_GaussianSplatRT";
            const string ProfilerTag = "GSplat Render";
            static readonly ProfilingSampler s_Sampler = new(ProfilerTag);
            static readonly int s_gaussianSplatRT = Shader.PropertyToID(GaussianSplatRTName);

            readonly Material m_CompositeMaterial;

            public GSplatPass(Material compositeMaterial)
            {
                m_CompositeMaterial = compositeMaterial;
            }

            class PassData
            {
                internal UniversalCameraData CameraData;
                internal TextureHandle SourceTexture;
                internal TextureHandle SourceDepth;
                internal TextureHandle GaussianSplatRT;
                internal bool IsStereo;
                internal Camera Camera;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph,
                ContextContainer frameData)
            {
                var inst = GSRenderer.ActiveInstance;
                if (inst == null || !inst.HasSplatsReady) return;
                if (m_CompositeMaterial == null) return;

                using var builder =
                    renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                bool isStereo = XRSettings.enabled && cameraData.camera.stereoEnabled &&
                    (XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassInstanced ||
                     XRSettings.stereoRenderingMode == XRSettings.StereoRenderingMode.SinglePassMultiview);

                RenderTextureDescriptor rtDesc = isStereo
                    ? XRSettings.eyeTextureDesc
                    : cameraData.cameraTargetDescriptor;
                rtDesc.depthBufferBits = 0;
                rtDesc.msaaSamples = 1;
                rtDesc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;

                var gaussianSplatRT = UniversalRenderer.CreateRenderGraphTexture(
                    renderGraph, rtDesc, GaussianSplatRTName, true);

                passData.CameraData = cameraData;
                passData.SourceTexture = resourceData.activeColorTexture;
                passData.SourceDepth = resourceData.activeDepthTexture;
                passData.GaussianSplatRT = gaussianSplatRT;
                passData.IsStereo = isStereo;
                passData.Camera = cameraData.camera;

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture);
                builder.UseTexture(gaussianSplatRT, AccessFlags.ReadWrite);
                builder.AllowPassCulling(false);

                var matComp = m_CompositeMaterial;

                builder.SetRenderFunc(
                    (PassData data, UnsafeGraphContext context) =>
                    {
                        var renderer = GSRenderer.ActiveInstance;
                        if (renderer == null || !renderer.HasSplatsReady) return;

                        var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                        using var _ = new ProfilingScope(cmd, s_Sampler);

                        var cam = data.Camera;
                        bool stereo = data.IsStereo;

                        if (stereo)
                        {
                            // Build per-eye VP matrices at render time (late-latched)
                            var viewL = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Left);
                            var projL = GL.GetGPUProjectionMatrix(
                                cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left), true);
                            Matrix4x4 vpL = projL * viewL;

                            var viewR = cam.GetStereoViewMatrix(Camera.StereoscopicEye.Right);
                            var projR = GL.GetGPUProjectionMatrix(
                                cam.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right), true);
                            Matrix4x4 vpR = projR * viewR;

                            // Prepass + Sort
                            renderer.PrepareAndSort(cmd, cam, true, vpL, vpR);

                            // Clear intermediate RT
                            CoreUtils.SetRenderTarget(cmd, data.GaussianSplatRT,
                                ClearFlag.Color, Color.clear);

                            // Render left eye to slice 0
                            CoreUtils.SetRenderTarget(cmd, data.GaussianSplatRT,
                                ClearFlag.Color, Color.clear, 0, CubemapFace.Unknown, 0);
                            renderer.DrawSplats(cmd, 0, true);

                            // Render right eye to slice 1
                            CoreUtils.SetRenderTarget(cmd, data.GaussianSplatRT,
                                ClearFlag.Color, Color.clear, 0, CubemapFace.Unknown, 1);
                            renderer.DrawSplats(cmd, 1, true);

                            // Composite per-eye to scene
                            matComp.SetTexture(s_gaussianSplatRT, data.GaussianSplatRT);

                            cmd.SetRenderTarget(data.SourceTexture, 0, CubemapFace.Unknown, 0);
                            cmd.SetGlobalInt("_CustomStereoEyeIndex", 0);
                            cmd.DrawProcedural(Matrix4x4.identity, matComp, 0,
                                MeshTopology.Triangles, 3, 1);

                            cmd.SetRenderTarget(data.SourceTexture, 0, CubemapFace.Unknown, 1);
                            cmd.SetGlobalInt("_CustomStereoEyeIndex", 1);
                            cmd.DrawProcedural(Matrix4x4.identity, matComp, 0,
                                MeshTopology.Triangles, 3, 1);
                        }
                        else
                        {
                            // Mono: build VP at render time
                            var view = cam.worldToCameraMatrix;
                            var proj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
                            Matrix4x4 vpL = proj * view;

                            // Prepass + Sort
                            renderer.PrepareAndSort(cmd, cam, false, vpL, Matrix4x4.identity);

                            // Render splats to intermediate RT
                            cmd.SetGlobalTexture(s_gaussianSplatRT, data.GaussianSplatRT);
                            CoreUtils.SetRenderTarget(cmd, data.GaussianSplatRT,
                                data.SourceDepth, ClearFlag.Color, Color.clear);
                            renderer.DrawSplats(cmd, 0, false);

                            // Composite to scene
                            Blitter.BlitCameraTexture(cmd, data.GaussianSplatRT,
                                data.SourceTexture, matComp, 0);
                        }
                    });
            }
        }

        [SerializeField] Shader compositeShader;

        GSplatPass m_Pass;
        Material m_CompositeMaterial;

        public override void Create()
        {
            if (m_CompositeMaterial == null)
            {
                var shader = compositeShader != null ? compositeShader : Shader.Find("Genesis/SplatComposite");
                if (shader != null)
                    m_CompositeMaterial = CoreUtils.CreateEngineMaterial(shader);
                else
                    Debug.LogWarning("[GSplatRenderFeature] Cannot find Genesis/SplatComposite shader");
            }

            m_Pass = new GSplatPass(m_CompositeMaterial)
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
            };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer,
            ref RenderingData renderingData)
        {
            var inst = GSRenderer.ActiveInstance;
            if (inst == null || !inst.HasSplatsReady) return;
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            if (m_CompositeMaterial != null)
                CoreUtils.Destroy(m_CompositeMaterial);
            m_CompositeMaterial = null;
            m_Pass = null;
        }
    }
}
