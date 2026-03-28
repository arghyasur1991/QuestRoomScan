using GaussianSplatting.Runtime;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Loads server-trained Gaussian splats from PLY data and renders them
    /// via the Unity Gaussian Splatting (UGS) package's <see cref="GaussianSplatRenderer"/>
    /// on a dedicated child GameObject whose transform can be set for room-anchor relocation.
    /// </summary>
    public class GSplatManager : MonoBehaviour
    {
        GaussianSplatRenderer _ugsRenderer;
        Transform _splatHolder;

        public bool HasServerTrainedSplats =>
            _ugsRenderer != null && _ugsRenderer.isRuntimeLoaded && _ugsRenderer.splatCount > 0;

        /// <summary>
        /// The child transform hosting the GaussianSplatRenderer.
        /// Set its position/rotation to apply room-anchor relocation.
        /// </summary>
        public Transform SplatHolder => _splatHolder;

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
            EnsureRendererOnChild();
        }

        void EnsureRendererOnChild()
        {
            if (_ugsRenderer != null) return;

            // Migrate: if GaussianSplatRenderer is on this GO (legacy), move it to a child.
            var existingOnSelf = GetComponent<GaussianSplatRenderer>();

            var child = transform.Find("SplatRenderer");
            if (child == null)
            {
                var go = new GameObject("SplatRenderer");
                go.transform.SetParent(transform, false);
                child = go.transform;
            }
            _splatHolder = child;

            _ugsRenderer = child.GetComponent<GaussianSplatRenderer>();
            if (_ugsRenderer == null)
            {
                if (existingOnSelf != null)
                {
                    // Can't move components at runtime; create a new one and copy serialized
                    // shader/compute refs. The setup wizard will re-wire them on next run.
                    _ugsRenderer = child.gameObject.AddComponent<GaussianSplatRenderer>();
                    CopyRendererSettings(existingOnSelf, _ugsRenderer);
                    Destroy(existingOnSelf);
                }
                else
                {
                    _ugsRenderer = child.gameObject.AddComponent<GaussianSplatRenderer>();
                }
            }
        }

        static void CopyRendererSettings(GaussianSplatRenderer src, GaussianSplatRenderer dst)
        {
            // Copy serialized shader/compute references via reflection-free public fields.
            // GaussianSplatRenderer exposes these as serialized fields set by the wizard.
            // If the fields aren't accessible, the wizard will re-wire on next "Fix All".
            try
            {
                var srcJson = JsonUtility.ToJson(src);
                JsonUtility.FromJsonOverwrite(srcJson, dst);
            }
            catch
            {
                Logger.Warning("Could not copy renderer settings; run Setup Wizard to re-wire shaders");
            }
        }

        /// <summary>
        /// Parses a 3DGS-format PLY, converts from COLMAP to Unity coordinates,
        /// and uploads to GPU buffers for UGS rendering.
        /// </summary>
        public void LoadTrainedPly(byte[] plyData)
        {
            EnsureRendererOnChild();

            if (_ugsRenderer == null)
            {
                Logger.Error("No GaussianSplatRenderer component found");
                return;
            }

            GaussianSplatPlyLoader.LoadFromPlyBytes(_ugsRenderer, plyData, colmapToUnity: true);
            Logger.Info($"Loaded trained splat via UGS ({_ugsRenderer.splatCount} Gaussians)");
        }

        /// <summary>Release all GPU resources for the loaded splat.</summary>
        public void ClearSplat()
        {
            if (_ugsRenderer != null)
                _ugsRenderer.ClearRuntimeSplatData();
        }

        /// <summary>Reset the splat holder to local identity (no relocation offset).</summary>
        public void ResetSplatTransform()
        {
            if (_splatHolder != null)
            {
                _splatHolder.localPosition = Vector3.zero;
                _splatHolder.localRotation = Quaternion.identity;
                _splatHolder.localScale = Vector3.one;
            }
        }

        void OnDestroy()
        {
            ClearSplat();
        }
    }
}
