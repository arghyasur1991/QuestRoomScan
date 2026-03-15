using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UIElements;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Picks the active VR controller, keeps <see cref="OVRInputModule.rayTransform"/>
    /// pointing along the controller ray, and draws a laser + cursor dot.
    /// Place on the same GameObject as the <c>EventSystem</c> / <c>OVRInputModule</c>.
    /// </summary>
    [RequireComponent(typeof(OVRInputModule))]
    public class ControllerRayDriver : MonoBehaviour
    {
        [Header("Ray")]
        [SerializeField, Tooltip("Forward offset from controller origin (meters)")]
        private float rayStartOffset = 0.05f;
        [SerializeField] private float maxLength = 5f;

        [Header("Laser Visual")]
        [SerializeField] private float beamWidth = 0.002f;
        [SerializeField] private Color idleColor = new(1f, 1f, 1f, 0.15f);
        [SerializeField] private Color hoverColor = new(0f, 0.8f, 1f, 0.7f);

        [Header("Cursor Dot")]
        [SerializeField] private float cursorRadius = 0.006f;
        [SerializeField] private Color cursorColor = new(1f, 1f, 1f, 0.9f);

        private OVRInputModule _inputModule;
        private Transform _rayHelper;
        private LineRenderer _line;
        private GameObject _cursor;
        private MeshRenderer _cursorRenderer;
        private OVRInput.Controller _activeController = OVRInput.Controller.RTouch;

        private static OVRPlugin.HandState _handState = new();

        // Layer mask matching the debug menu's panel collider layer
        private int _uiLayerMask;

        private void Awake()
        {
            _inputModule = GetComponent<OVRInputModule>();

            _rayHelper = new GameObject("ControllerRayHelper").transform;
            _rayHelper.SetParent(transform, false);
            _inputModule.rayTransform = _rayHelper;
            _inputModule.joyPadClickButton = OVRInput.Button.PrimaryIndexTrigger;

            _uiLayerMask = LayerMask.GetMask("Default", "UI");

            SetupLineRenderer();
            SetupCursor();
        }

        private void Update()
        {
            _activeController = ChooseBestController(_activeController);
            UpdateRayOrigin();
        }

        private void LateUpdate()
        {
            DrawLaser();
        }

        private void OnDestroy()
        {
            if (_rayHelper != null) Destroy(_rayHelper.gameObject);
            if (_cursor != null) Destroy(_cursor);
        }

        // ─── Controller Selection (adapted from Meta ImmersiveDebugger) ───

        private static OVRInput.Controller ChooseBestController(OVRInput.Controller previous)
        {
            var left = OVRInput.GetActiveControllerForHand(OVRInput.Handedness.LeftHanded);
            var right = OVRInput.GetActiveControllerForHand(OVRInput.Handedness.RightHanded);

            var ctrl = previous;
            if (ctrl == OVRInput.Controller.None || (ctrl != left && ctrl != right))
            {
                ctrl = right != OVRInput.Controller.None ? right
                     : left != OVRInput.Controller.None ? left
                     : OVRInput.GetDominantHand() == OVRInput.Handedness.LeftHanded ? left : right;
            }

            if (ctrl != left && OVRInput.Get(OVRInput.Button.Any, left)) ctrl = left;
            if (ctrl != right && OVRInput.Get(OVRInput.Button.Any, right)) ctrl = right;
            if (ctrl == OVRInput.Controller.None) ctrl = OVRInput.Controller.RTouch;

            return ctrl;
        }

        // ─── Ray Transform ───

        private void UpdateRayOrigin()
        {
            bool isHand = _activeController is OVRInput.Controller.LHand or OVRInput.Controller.RHand;

            Vector3 localPos;
            Quaternion localRot;

            if (isHand)
            {
                var hand = _activeController == OVRInput.Controller.LHand
                    ? OVRPlugin.Hand.HandLeft : OVRPlugin.Hand.HandRight;
                OVRPlugin.GetHandState(OVRPlugin.Step.Render, hand, ref _handState);
                localPos = _handState.PointerPose.Position.FromFlippedZVector3f();
                localRot = _handState.PointerPose.Orientation.FromFlippedZQuatf();
            }
            else
            {
                localPos = OVRInput.GetLocalControllerPosition(_activeController);
                localRot = OVRInput.GetLocalControllerRotation(_activeController);
            }

            var pose = new OVRPose { position = localPos, orientation = localRot };

            var cam = Camera.main;
            if (cam != null) pose = pose.ToWorldSpacePose(cam);

            _rayHelper.SetPositionAndRotation(pose.position, pose.orientation);
        }

        // ─── Laser Visual ───

        private void SetupLineRenderer()
        {
            _line = gameObject.AddComponent<LineRenderer>();
            _line.positionCount = 2;
            _line.startWidth = beamWidth;
            _line.endWidth = beamWidth * 0.5f;
            _line.material = new Material(Shader.Find("Sprites/Default"));
            _line.startColor = _line.endColor = idleColor;
            _line.useWorldSpace = true;
            _line.receiveShadows = false;
            _line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }

        private void SetupCursor()
        {
            _cursor = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _cursor.name = "RayCursor";
            _cursor.transform.localScale = Vector3.one * (cursorRadius * 2f);

            // Remove the collider so it doesn't interfere with raycasts
            var col = _cursor.GetComponent<Collider>();
            if (col != null) Destroy(col);

            _cursorRenderer = _cursor.GetComponent<MeshRenderer>();
            _cursorRenderer.material = new Material(Shader.Find("Sprites/Default"));
            _cursorRenderer.material.color = cursorColor;
            _cursorRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _cursorRenderer.receiveShadows = false;

            _cursor.SetActive(false);
        }

        private void DrawLaser()
        {
            if (_rayHelper == null || _line == null) return;

            var origin = _rayHelper.position;
            var dir = _rayHelper.forward;
            var start = origin + dir * rayStartOffset;
            var end = start + dir * maxLength;
            bool hoveringUI = false;

            // Only highlight when hitting a world-space UI Toolkit panel collider
            if (Physics.Raycast(origin, dir, out var hit, maxLength + rayStartOffset))
            {
                end = hit.point;

                // Check if we hit a UIDocument's auto-generated panel collider
                var uiDoc = hit.collider.GetComponentInParent<UIDocument>();
                hoveringUI = uiDoc != null;
            }

            _line.SetPosition(0, start);
            _line.SetPosition(1, end);

            var color = hoveringUI ? hoverColor : idleColor;
            _line.startColor = _line.endColor = color;

            // Position cursor dot at the end of the ray
            if (_cursor != null)
            {
                bool showCursor = hoveringUI;
                _cursor.SetActive(showCursor);
                if (showCursor)
                {
                    _cursor.transform.position = end;
                    _cursor.transform.LookAt(_rayHelper);
                    _cursorRenderer.material.color = hoverColor;
                }
            }
        }
    }
}
