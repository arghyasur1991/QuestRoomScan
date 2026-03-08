using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Lightweight wrapper around a compute shader kernel to reduce boilerplate
    /// when setting properties and dispatching.
    /// </summary>
    public struct ComputeKernelHelper
    {
        public readonly ComputeShader Shader;
        public readonly int KernelIndex;
        private uint _threadGroupX, _threadGroupY, _threadGroupZ;

        public ComputeKernelHelper(ComputeShader shader, string kernelName)
        {
            Shader = shader;
            KernelIndex = shader.FindKernel(kernelName);
            shader.GetKernelThreadGroupSizes(KernelIndex, out _threadGroupX, out _threadGroupY, out _threadGroupZ);
        }

        public void Set(int nameID, Texture tex)
        {
            Shader.SetTexture(KernelIndex, nameID, tex);
        }

        public void Set(int nameID, ComputeBuffer buffer)
        {
            Shader.SetBuffer(KernelIndex, nameID, buffer);
        }

        public void Set(int nameID, GraphicsBuffer buffer)
        {
            Shader.SetBuffer(KernelIndex, nameID, buffer);
        }

        public void DispatchFit(int sizeX, int sizeY, int sizeZ = 1)
        {
            int gx = Mathf.CeilToInt((float)sizeX / _threadGroupX);
            int gy = Mathf.CeilToInt((float)sizeY / _threadGroupY);
            int gz = Mathf.CeilToInt((float)sizeZ / _threadGroupZ);
            Shader.Dispatch(KernelIndex, Mathf.Max(1, gx), Mathf.Max(1, gy), Mathf.Max(1, gz));
        }

        public void DispatchFit(Texture tex)
        {
            int w = tex.width;
            int h = tex.height;
            int d = 1;
            if (tex is RenderTexture rt && rt.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray)
                d = rt.volumeDepth;
            DispatchFit(w, h, d);
        }

        public void DispatchFit(RenderTexture volume)
        {
            DispatchFit(volume.width, volume.height, volume.volumeDepth);
        }
    }
}
