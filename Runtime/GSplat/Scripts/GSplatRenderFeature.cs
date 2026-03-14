using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// URP Renderer Feature that draws Gaussian splats with proper stereo RT slice
    /// targeting. On Quest, Unity can't correctly set unity_stereoEyeIndex when
    /// drawing procedurally, so we render each eye separately by binding the
    /// correct texture-array slice before each DrawProcedural call — same
    /// workaround that UnityGaussianSplatting uses.
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
            }

            public override void RecordRenderGraph(RenderGraph renderGraph,
                ContextContainer frameData)
            {
                var inst = GSSectorRenderer.ActiveInstance;
                if (inst == null || !inst.HasPreparedSplats) return;

                using var builder =
                    renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var resourceData = frameData.Get<UniversalResourceData>();

                passData.ColorTarget = resourceData.activeColorTexture;
                passData.DepthTarget = resourceData.activeDepthTexture;
                passData.IsStereo = inst.IsStereoMode;

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

                        if (data.IsStereo)
                        {
                            // Left eye → texture array slice 0
                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 0);
                            renderer.DrawSplats(cmd, 0);

                            // Right eye → texture array slice 1
                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 1);
                            renderer.DrawSplats(cmd, 1);
                        }
                        else
                        {
                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget);
                            renderer.DrawSplats(cmd, -1);
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
