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
        /// <summary>
        /// Unity's worldToCameraMatrix uses -Z forward (OpenGL convention).
        /// The EWA projection math expects +Z forward (objects in front have z > 0).
        /// Negate row 2 to convert.
        /// </summary>
        static Matrix4x4 ViewToPositiveZ(Matrix4x4 v)
        {
            v[2, 0] = -v[2, 0];
            v[2, 1] = -v[2, 1];
            v[2, 2] = -v[2, 2];
            v[2, 3] = -v[2, 3];
            return v;
        }

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
                        bool isStereo = data.IsStereo;
                        float screenW = isStereo
                            ? XRSettings.eyeTextureWidth
                            : cam.pixelWidth;
                        float screenH = isStereo
                            ? XRSettings.eyeTextureHeight
                            : cam.pixelHeight;

                        if (isStereo)
                        {
                            var viewL = cam.GetStereoViewMatrix(
                                Camera.StereoscopicEye.Left);
                            var projL = GL.GetGPUProjectionMatrix(
                                cam.GetStereoProjectionMatrix(
                                    Camera.StereoscopicEye.Left), true);
                            float fxL = Mathf.Abs(projL[0, 0]) * screenW / 2f;
                            float fyL = Mathf.Abs(projL[1, 1]) * screenH / 2f;

                            var viewR = cam.GetStereoViewMatrix(
                                Camera.StereoscopicEye.Right);
                            var projR = GL.GetGPUProjectionMatrix(
                                cam.GetStereoProjectionMatrix(
                                    Camera.StereoscopicEye.Right), true);
                            float fxR = Mathf.Abs(projR[0, 0]) * screenW / 2f;
                            float fyR = Mathf.Abs(projR[1, 1]) * screenH / 2f;

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 0);
                            renderer.DrawSplats(cmd, projL * viewL,
                                ViewToPositiveZ(viewL),
                                fxL, fyL, screenW, screenH);

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget,
                                0, CubemapFace.Unknown, 1);
                            renderer.DrawSplats(cmd, projR * viewR,
                                ViewToPositiveZ(viewR),
                                fxR, fyR, screenW, screenH);
                        }
                        else
                        {
                            var view = cam.worldToCameraMatrix;
                            var proj = GL.GetGPUProjectionMatrix(
                                cam.projectionMatrix, true);
                            float fx = Mathf.Abs(proj[0, 0]) * screenW / 2f;
                            float fy = Mathf.Abs(proj[1, 1]) * screenH / 2f;

                            cmd.SetRenderTarget(data.ColorTarget, data.DepthTarget);
                            renderer.DrawSplats(cmd, proj * view,
                                ViewToPositiveZ(view),
                                fx, fy, screenW, screenH);
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
