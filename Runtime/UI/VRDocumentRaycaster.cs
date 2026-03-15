using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UIElements;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Extends <see cref="WorldDocumentRaycaster"/> so that VR controller rays
    /// (carried in <see cref="OVRPointerEventData.worldSpaceRay"/>) are used for
    /// raycasting against world-space UI Toolkit panels.
    ///
    /// When no VR pointer data is available (e.g. in-editor with a mouse), falls
    /// back to the default screen-to-camera-ray conversion.
    ///
    /// Add this component alongside (or instead of) the auto-created
    /// <c>WorldDocumentRaycaster</c> on the EventSystem GameObject.
    /// </summary>
    [AddComponentMenu("UI Toolkit/VR Document Raycaster (Quest)")]
    public class VRDocumentRaycaster : WorldDocumentRaycaster
    {
        [SerializeField, Tooltip("Max ray distance for UI interaction (meters)")]
        private float maxRayDistance = 5f;

        [SerializeField, Tooltip("Physics layers to raycast against")]
        private LayerMask interactionLayers = ~0;

        protected override bool GetWorldRay(
            PointerEventData eventData,
            out Ray worldRay,
            out float maxDistance,
            out int layerMask)
        {
            if (eventData is OVRPointerEventData ovrData
                && ovrData.worldSpaceRay.direction.sqrMagnitude > 0.001f)
            {
                worldRay = ovrData.worldSpaceRay;
                maxDistance = maxRayDistance;
                layerMask = interactionLayers.value;
                return true;
            }

            return base.GetWorldRay(eventData, out worldRay, out maxDistance, out layerMask);
        }
    }
}
