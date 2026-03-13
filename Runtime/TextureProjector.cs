using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Legacy texture projector stub. With the GPU Surface Nets pipeline,
    /// mesh coloring is handled by the fragment shader (keyframe projection,
    /// triplanar cache, vertex color fallback) — no per-vertex compute
    /// projection is needed. Retained as a component reference for scene
    /// compatibility; ProjectFrame is a no-op.
    /// </summary>
    public class TextureProjector : MonoBehaviour
    {
        public static TextureProjector Instance { get; private set; }

        private ICameraProvider _cameraProvider;

        private void Awake()
        {
            Instance = this;
        }

        public void SetCameraProvider(ICameraProvider provider)
        {
            _cameraProvider = provider;
        }

        public void ProjectFrame()
        {
            // No-op: GPU mesh has no per-chunk Mesh objects to project onto.
            // Fragment shader handles all coloring via keyframes + triplanar + vertex colors.
        }
    }
}
