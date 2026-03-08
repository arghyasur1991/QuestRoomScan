using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Projects camera RGB onto mesh vertices via compute shader.
    /// Iterates over all populated chunks and dispatches the projection kernel
    /// for each mesh's vertex buffer.
    /// </summary>
    public class TextureProjector : MonoBehaviour
    {
        public static TextureProjector Instance { get; private set; }

        [SerializeField] private ComputeShader projectionCompute;
        [SerializeField] private float projectionWeight = 0.5f;

        private ComputeKernelHelper _projectKernel;

        private static readonly int VertexBufferID = Shader.PropertyToID("gsVertexBuffer");
        private static readonly int VertexCountID = Shader.PropertyToID("gsVertexCount");
        private static readonly int CameraRGBID = Shader.PropertyToID("gsCameraRGB");
        private static readonly int CameraVPID = Shader.PropertyToID("gsCameraVP");
        private static readonly int CameraPosID = Shader.PropertyToID("gsCameraPos");
        private static readonly int CameraWeightID = Shader.PropertyToID("gsCameraWeight");

        private ICameraProvider _cameraProvider;
        private bool _initialized;

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            if (projectionCompute != null)
                _projectKernel = new ComputeKernelHelper(projectionCompute, "ProjectColors");
            _initialized = projectionCompute != null;
        }

        public void SetCameraProvider(ICameraProvider provider)
        {
            _cameraProvider = provider;
        }

        /// <summary>
        /// Project current camera frame onto all populated chunk meshes.
        /// Called by RoomScanner at the configured frequency.
        /// </summary>
        public void ProjectFrame()
        {
            if (!_initialized || _cameraProvider == null || !_cameraProvider.IsReady)
                return;

            var chunkMgr = ChunkManager.Instance;
            if (chunkMgr == null) return;

            Texture frame = _cameraProvider.CurrentFrame;
            Matrix4x4 vp = _cameraProvider.ProjectionMatrix * _cameraProvider.CameraToWorld.inverse;
            Vector3 camPos = _cameraProvider.CameraToWorld.MultiplyPoint(Vector3.zero);

            projectionCompute.SetTexture(_projectKernel.KernelIndex, CameraRGBID, frame);
            projectionCompute.SetMatrix(CameraVPID, vp);
            projectionCompute.SetVector(CameraPosID, camPos);
            projectionCompute.SetFloat(CameraWeightID, projectionWeight);

            foreach (MeshChunkData chunk in chunkMgr.GetPopulatedChunks())
            {
                if (chunk.Mesh == null || chunk.Mesh.vertexCount == 0)
                    continue;

                GraphicsBuffer vertBuf = chunk.Mesh.GetVertexBuffer(0);
                if (vertBuf == null) continue;

                projectionCompute.SetInt(VertexCountID, chunk.Mesh.vertexCount);
                _projectKernel.Set(VertexBufferID, vertBuf);
                _projectKernel.DispatchFit(chunk.Mesh.vertexCount, 1);

                vertBuf.Dispose();
            }
        }
    }
}
