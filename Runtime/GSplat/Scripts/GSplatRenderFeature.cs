using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.XR;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// URP Renderer Feature: dispatches GSplat prepass + sort + draw all inside
    /// the render pass command buffer (matching UGS architecture). Per-eye VP
    /// matrices are captured at render time (after XR late-latching) and passed
    /// to the prepass. Renders back-to-front directly to the scene buffer.
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
                if (inst == null || !inst.HasSplatsReady) return;

                using var builder =
                    renderGraph.AddUnsafePass(ProfilerTag, out PassData passData);

                var cameraData = frameData.Get<UniversalCameraData>();
                var resourceData = frameData.Get<UniversalResourceData>();

                bool isStereo = XRSettings.enabled && cameraData.camera.stereoEnabled;

                passData.ColorTarget = resourceData.activeColorTexture;
                passData.DepthTarget = resourceData.activeDepthTexture;
                passData.IsStereo = isStereo;
                passData.Camera = cameraData.camera;

                builder.UseTexture(resourceData.activeColorTexture, AccessFlags.ReadWrite);
                builder.UseTexture(resourceData.activeDepthTexture, AccessFlags.Read);
                builder.AllowPassCulling(false);

                builder.SetRenderFunc(
                    static (PassData data, UnsafeGraphContext context) =>
                    {
                        var renderer = GSSectorRenderer.ActiveInstance;
                        if (renderer == null || !renderer.HasSplatsReady) return;

                        var cmd =
                            CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                        using var _ = new ProfilingScope(cmd, s_Sampler);

                        var cam = data.Camera;
                        bool stereo = data.IsStereo;

                        // Build per-eye VP matrices at render time (late-latched)
                        Matrix4x4 vpL, vpR = Matrix4x4.identity;
                        if (stereo)
                        {
                            var viewL = cam.GetStereoViewMatrix(
                                Camera.StereoscopicEye.Left);
                            var projL = GL.GetGPUProjectionMatrix(
                                cam.GetStereoProjectionMatrix(
                                    Camera.StereoscopicEye.Left), true);
                            vpL = projL * viewL;

                            var viewR = cam.GetStereoViewMatrix(
                                Camera.StereoscopicEye.Right);
                            var projR = GL.GetGPUProjectionMatrix(
                                cam.GetStereoProjectionMatrix(
                                    Camera.StereoscopicEye.Right), true);
                            vpR = projR * viewR;
                        }
                        else
                        {
                            var view = cam.worldToCameraMatrix;
                            var proj = GL.GetGPUProjectionMatrix(
                                cam.projectionMatrix, true);
                            vpL = proj * view;
                        }

                        // Prepass + Sort (all dispatches via command buffer)
                        renderer.PrepareAndSort(cmd, cam, stereo, vpL, vpR);

                        // Draw splats
                        if (stereo)
                        {
                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 0);
                            renderer.DrawSplats(cmd, 0, true);

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 1);
                            renderer.DrawSplats(cmd, 1, true);
                        }
                        else
                        {
                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget);
                            renderer.DrawSplats(cmd, 0, false);
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
            if (inst == null || !inst.HasSplatsReady) return;
            renderer.EnqueuePass(m_Pass);
        }

        protected override void Dispose(bool disposing)
        {
            m_Pass = null;
        }
    }
}
