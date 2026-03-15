using UnityEngine;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Positions the debug menu as a world-space floating panel in front of the
    /// user's head. When toggled on, the panel snaps to the center of the user's
    /// view. It stays locked in place until the user looks far enough away, at
    /// which point it smoothly re-centers.
    /// </summary>
    public class DebugMenuFollower : MonoBehaviour
    {
        [Header("Placement")]
        [SerializeField, Tooltip("Distance from the camera in meters")]
        private float panelDistance = 0.75f;

        [SerializeField, Tooltip("Vertical offset from eye level (negative = below gaze)")]
        private float verticalOffset = -0.05f;

        [Header("Lazy Follow")]
        [SerializeField, Tooltip("Angle (degrees) the panel must drift before it re-centers")]
        private float followThreshold = 45f;

        [SerializeField, Tooltip("How fast the panel re-centers (higher = snappier)")]
        private float followSpeed = 6f;

        private Transform _cam;
        private bool _tracking;
        private Vector3 _anchorPosition;
        private bool _needsRecenter;

        public bool IsTracking => _tracking;

        private void OnEnable()
        {
            _cam = Camera.main != null ? Camera.main.transform : null;
        }

        private void LateUpdate()
        {
            if (!_tracking || _cam == null) return;

            // Check if the panel has drifted outside the user's comfortable view
            Vector3 toPanel = (_anchorPosition - _cam.position).normalized;
            float angle = Vector3.Angle(_cam.forward, toPanel);

            if (angle > followThreshold)
            {
                _anchorPosition = ComputeTargetPosition();
                _needsRecenter = true;
            }

            // Move toward anchor when re-centering
            if (_needsRecenter)
            {
                transform.position = Vector3.Lerp(transform.position, _anchorPosition,
                    followSpeed * Time.deltaTime);

                if (Vector3.Distance(transform.position, _anchorPosition) < 0.002f)
                {
                    transform.position = _anchorPosition;
                    _needsRecenter = false;
                }
            }

            // Billboard: always face the camera
            Vector3 awayFromCam = transform.position - _cam.position;
            if (awayFromCam.sqrMagnitude > 0.001f)
                transform.rotation = Quaternion.LookRotation(awayFromCam);
        }

        /// <summary>
        /// Instantly places the panel in the center of the user's view.
        /// </summary>
        public void SnapToView()
        {
            if (_cam == null)
                _cam = Camera.main != null ? Camera.main.transform : null;
            if (_cam == null) return;

            _anchorPosition = ComputeTargetPosition();
            transform.position = _anchorPosition;
            transform.rotation = Quaternion.LookRotation(transform.position - _cam.position);
            _needsRecenter = false;
            _tracking = true;
        }

        public void StopTracking()
        {
            _tracking = false;
        }

        private Vector3 ComputeTargetPosition()
        {
            // Place directly along the camera's forward direction so the panel
            // always appears dead-center in the user's view.
            return _cam.position
                + _cam.forward * panelDistance
                + Vector3.up * verticalOffset;
        }
    }
}
