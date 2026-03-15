using UnityEngine;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Positions the debug menu as a world-space floating panel that lazily
    /// follows the user's head. Attach to the same GameObject as
    /// <see cref="DebugMenuController"/>. Call <see cref="SnapToView"/> when
    /// the panel first becomes visible.
    /// </summary>
    public class DebugMenuFollower : MonoBehaviour
    {
        [Header("Placement")]
        [SerializeField, Tooltip("Distance from the camera in meters")]
        private float panelDistance = 0.75f;

        [SerializeField, Tooltip("Vertical offset from gaze point in meters (negative = below)")]
        private float verticalOffset = -0.08f;

        [Header("Lazy Follow")]
        [SerializeField, Tooltip("Angle (degrees) the panel must drift off-center before it re-centers")]
        private float followThreshold = 40f;

        [SerializeField, Tooltip("How fast the panel catches up (higher = snappier)")]
        private float followSpeed = 3f;

        [SerializeField, Tooltip("Rotation lerp speed for billboarding")]
        private float rotationSpeed = 6f;

        private Transform _cam;
        private bool _tracking;

        /// <summary>Current target position the panel is lerping toward.</summary>
        private Vector3 _targetPosition;

        private void OnEnable()
        {
            _cam = Camera.main != null ? Camera.main.transform : null;
        }

        private void LateUpdate()
        {
            if (!_tracking || _cam == null) return;

            Vector3 toPanel = (transform.position - _cam.position).normalized;
            float angle = Vector3.Angle(_cam.forward, toPanel);

            if (angle > followThreshold)
                _targetPosition = ComputeTargetPosition();

            // Only lerp position when the panel needs to re-center;
            // otherwise keep it locked so it doesn't drift when the user looks at it.
            float dist = Vector3.Distance(transform.position, _targetPosition);
            if (dist > 0.005f)
            {
                transform.position = Vector3.Lerp(transform.position, _targetPosition,
                    followSpeed * Time.deltaTime);
            }

            // Always billboard toward the camera
            Quaternion lookRot = Quaternion.LookRotation(transform.position - _cam.position);
            transform.rotation = Quaternion.Slerp(transform.rotation, lookRot,
                rotationSpeed * Time.deltaTime);
        }

        /// <summary>
        /// Instantly places the panel in front of the camera. Call when the
        /// menu first becomes visible.
        /// </summary>
        public void SnapToView()
        {
            if (_cam == null)
                _cam = Camera.main != null ? Camera.main.transform : null;
            if (_cam == null) return;

            _targetPosition = ComputeTargetPosition();
            transform.position = _targetPosition;
            transform.rotation = Quaternion.LookRotation(transform.position - _cam.position);
            _tracking = true;
        }

        /// <summary>
        /// Stops lazy-follow tracking (panel stays where it is).
        /// </summary>
        public void StopTracking()
        {
            _tracking = false;
        }

        public bool IsTracking => _tracking;

        private Vector3 ComputeTargetPosition()
        {
            // Use the horizontal component of the camera forward so the panel
            // stays at a consistent height regardless of head pitch.
            Vector3 flatForward = _cam.forward;
            flatForward.y = 0f;
            if (flatForward.sqrMagnitude < 0.001f)
                flatForward = Vector3.forward;
            flatForward.Normalize();

            return _cam.position
                + flatForward * panelDistance
                + Vector3.up * (_cam.forward.y * panelDistance * 0.3f + verticalOffset);
        }
    }
}
