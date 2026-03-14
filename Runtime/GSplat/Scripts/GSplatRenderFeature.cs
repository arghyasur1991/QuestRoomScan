using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// URP Renderer Feature that draws Gaussian splats with proper stereo RT slice
    /// targeting. Per-eye VP matrices are captured at render time (after XR late-
    /// latching) so the vertex shader produces jitter-free clip-space positions.
    /// On Quest, Unity can't correctly set unity_stereoEyeIndex when drawing
    /// procedurally, so we render each eye separately — same workaround as UGS.
    /// </summary>
    public class GSplatRenderFeature : ScriptableRendererFeature
    {
        class GSplatPass : ScriptableRenderPass
        {
            const string ProfilerTag = "GSplat Render";
            static readonly ProfilingSampler s_Sampler = new(ProfilerTag);

            class PassData
            {
                internal TextureHandle ColorTarget;
                internal TextureHandle DepthTarget;
                internal bool IsStereo;
                internal Camera Camera;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph,
                ContextContainer frameData)
            {
                var inst = GSSectorRenderer.ActiveInstance;
                if (inst == null || !inst.HasPreparedSplats) return;

                using var builder =
                    renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                passData.ColorTarget = resourceData.activeColorTexture;
                passData.DepthTarget = resourceData.activeDepthTexture;
                passData.IsStereo = inst.IsStereoMode;
                passData.Camera = cameraData.camera;

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture, AccessFlags.Read);
                builder.AllowPassCulling(false);

                builder.SetRenderFunc(
                    static (PassData data, UnsafeGraphContext context) =>
                    {
                        var renderer = GSSectorRenderer.ActiveInstance;
                        if (renderer == null || !renderer.HasPreparedSplats) return;

                        var cmd =
                            CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                        using var _ = new ProfilingScope(cmd, s_Sampler);

                        var cam = data.Camera;

                        if (data.IsStereo)
                        {
                            var vpL = GL.GetGPUProjectionMatrix(
                                          cam.GetStereoProjectionMatrix(
                                              Camera.StereoscopicEye.Left), true)
                                      * cam.GetStereoViewMatrix(
                                          Camera.StereoscopicEye.Left);

                            var vpR = GL.GetGPUProjectionMatrix(
                                          cam.GetStereoProjectionMatrix(
                                              Camera.StereoscopicEye.Right), true)
                                      * cam.GetStereoViewMatrix(
                                          Camera.StereoscopicEye.Right);

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 0);
                            renderer.DrawSplats(cmd, vpL);

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 1);
                            renderer.DrawSplats(cmd, vpR);
                        }
                        else
                        {
                            var vp = GL.GetGPUProjectionMatrix(
                                         cam.projectionMatrix, true)
                                     * cam.worldToCameraMatrix;

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget);
                            renderer.DrawSplats(cmd, vp);
                        }
                    });
            }
        }

        GSplatPass m_Pass;

        public override void Create()
        {
            m_Pass = new GSplatPass
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents
            };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer,
            ref RenderingData renderingData)
        {
            var inst = GSSectorRenderer.ActiveInstance;
            if (inst == null || !inst.HasPreparedSplats) return;
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            m_Pass = null;
        }
    }
}
