using UnityEngine;
using UnityEngine.EventSystems;

namespace Genesis.RoomScan.UI
{
    /// <summary>
    /// Picks the active VR controller, keeps <see cref="OVRInputModule.rayTransform"/>
    /// pointing along the controller ray, and draws a laser visual.
    /// Place on the same GameObject as the <c>EventSystem</c> / <c>OVRInputModule</c>.
    /// </summary>
    [RequireComponent(typeof(OVRInputModule))]
    public class ControllerRayDriver : MonoBehaviour
    {
        [Header("Laser Visual")]
        [SerializeField] private float maxLength = 5f;
        [SerializeField] private float beamWidth = 0.003f;
        [SerializeField] private Color idleColor = new(1f, 1f, 1f, 0.25f);
        [SerializeField] private Color hoverColor = new(0f, 0.8f, 1f, 0.8f);

        private OVRInputModule _inputModule;
        private Transform _rayHelper;
        private LineRenderer _line;
        private OVRInput.Controller _activeController = OVRInput.Controller.RTouch;

        private static OVRPlugin.HandState _handState = new();

        private void Awake()
        {
            _inputModule = GetComponent<OVRInputModule>();

            _rayHelper = new GameObject("ControllerRayHelper").transform;
            _rayHelper.SetParent(transform, false);
            _inputModule.rayTransform = _rayHelper;
            _inputModule.joyPadClickButton = OVRInput.Button.PrimaryIndexTrigger;

            SetupLineRenderer();
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
            if (_rayHelper != null)
                Destroy(_rayHelper.gameObject);
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
            _line.endWidth = beamWidth * 0.3f;
            _line.material = new Material(Shader.Find("Sprites/Default"));
            _line.startColor = _line.endColor = idleColor;
            _line.useWorldSpace = true;
            _line.receiveShadows = false;
            _line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }

        private void DrawLaser()
        {
            if (_rayHelper == null || _line == null) return;

            var start = _rayHelper.position;
            var dir = _rayHelper.forward;
            var end = start + dir * maxLength;
            bool hovering = false;

            if (Physics.Raycast(start, dir, out var hit, maxLength))
            {
                end = hit.point;
                hovering = true;
            }

            _line.SetPosition(0, start);
            _line.SetPosition(1, end);
            _line.startColor = _line.endColor = hovering ? hoverColor : idleColor;
        }
    }
}
