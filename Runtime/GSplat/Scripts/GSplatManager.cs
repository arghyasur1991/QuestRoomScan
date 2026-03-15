using GaussianSplatting.Runtime;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Loads server-trained Gaussian splats from PLY data and renders them
    /// via the Unity Gaussian Splatting (UGS) package's <see cref="GaussianSplatRenderer"/>.
    /// </summary>
    [RequireComponent(typeof(GaussianSplatRenderer))]
    public class GSplatManager : MonoBehaviour
    {
        GaussianSplatRenderer _ugsRenderer;

        public bool HasServerTrainedSplats =>
            _ugsRenderer != null && _ugsRenderer.isRuntimeLoaded && _ugsRenderer.splatCount > 0;

        /// <summary>
        /// Toggle splat visibility without releasing GPU resources.
        /// </summary>
        public bool RenderVisible
        {
            get => _ugsRenderer != null && _ugsRenderer.renderVisible;
            set { if (_ugsRenderer != null) _ugsRenderer.renderVisible = value; }
        }

        void Awake()
        {
            _ugsRenderer = GetComponent<GaussianSplatRenderer>();
        }

        /// <summary>
        /// Parses a 3DGS-format PLY, converts from COLMAP to Unity coordinates,
        /// and uploads to GPU buffers for UGS rendering.
        /// </summary>
        public void LoadTrainedPly(byte[] plyData)
        {
            if (_ugsRenderer == null)
                _ugsRenderer = GetComponent<GaussianSplatRenderer>();

            if (_ugsRenderer == null)
            {
                Debug.LogError("[GSplatManager] No GaussianSplatRenderer component found");
                return;
            }

            GaussianSplatPlyLoader.LoadFromPlyBytes(_ugsRenderer, plyData, colmapToUnity: true);
            Debug.Log($"[GSplatManager] Loaded trained splat via UGS ({_ugsRenderer.splatCount} Gaussians)");
        }

        /// <summary>Release all GPU resources for the loaded splat.</summary>
        public void ClearSplat()
        {
            if (_ugsRenderer != null)
                _ugsRenderer.ClearRuntimeSplatData();
        }

        void OnDestroy()
        {
            ClearSplat();
        }
    }
}
