using System;
using Unity.XR.CoreUtils.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace Genesis.RoomScan
{
    [DefaultExecutionOrder(-40)]
    public class DepthCapture : MonoBehaviour
    {
        public static DepthCapture Instance { get; private set; }

        [SerializeField] private ComputeShader depthNormalCompute;
        [SerializeField] private ComputeShader depthDilationCompute;

        [Header("Dilation")]
        [SerializeField] private int dilationSteps = 8;
        [SerializeField] private float voxelDistance = 0.2f;
        [SerializeField] private float voxelSize = 0.05f;

        private readonly Matrix4x4[] _proj = new Matrix4x4[2];
        private readonly Matrix4x4[] _projInv = new Matrix4x4[2];
        private readonly Matrix4x4[] _view = new Matrix4x4[2];
        private readonly Matrix4x4[] _viewInv = new Matrix4x4[2];
        private Vector2 _planes;

        public Matrix4x4[] Proj => _proj;
        public Matrix4x4[] ProjInv => _projInv;
        public Matrix4x4[] View => _view;
        public Matrix4x4[] ViewInv => _viewInv;
        public Vector2 Planes => _planes;

        // Shader property IDs
        public static readonly int DepthTexID = Shader.PropertyToID("gsDepthTex");
        public static readonly int DepthTexRWID = Shader.PropertyToID("gsDepthTexRW");
        public static readonly int TexSizeID = Shader.PropertyToID("gsDepthTexSize");
        public static readonly int NormTexID = Shader.PropertyToID("gsDepthNormalTex");
        public static readonly int NormTexRWID = Shader.PropertyToID("gsDepthNormalTexRW");
        public static readonly int ZParamsID = Shader.PropertyToID("gsDepthZParams");
        public static readonly int ProjID = Shader.PropertyToID("gsDepthProj");
        public static readonly int ProjInvID = Shader.PropertyToID("gsDepthProjInv");
        public static readonly int ViewID = Shader.PropertyToID("gsDepthView");
        public static readonly int ViewInvID = Shader.PropertyToID("gsDepthViewInv");
        public static readonly int InputRawMonoDepthID = Shader.PropertyToID("gsInputRawMonoDepth");
        public static readonly int DilateSrcID = Shader.PropertyToID("gsDilateSrc");
        public static readonly int DilateDestID = Shader.PropertyToID("gsDilateDest");
        public static readonly int DilateStepSizeID = Shader.PropertyToID("gsDilateStepSize");
        public static readonly int DilatedDepthTexID = Shader.PropertyToID("gsDilatedDepth");
        public static readonly int VoxDistID = Shader.PropertyToID("gsVoxDist");
        public static readonly int VoxSizeShaderID = Shader.PropertyToID("gsVoxSize");

        public static bool DepthAvailable { get; private set; }

        private ComputeKernelHelper _normKernel;
        private ComputeKernelHelper _monoConvertKernel;
        private ComputeKernelHelper _initDilateKernel;
        private ComputeKernelHelper _dilateStepKernel;

        private Texture _depthTex;
        public Texture DepthTex => _depthTex;

        private RenderTexture _normTex;
        public RenderTexture NormTex => _normTex;

        private RenderTexture _dilationA, _dilationB;
        private RenderTexture _dilatedDepth;
        public RenderTexture DilatedDepthTex => _dilatedDepth;

        private RenderTexture _simulatedDepthTex;
        private int _dilationMaxStep;

        private AROcclusionManager _arOcclusionManager;
        private Unity.XR.CoreUtils.XROrigin _xrOrigin;
        private Camera _mainCam;
        private bool _started;
        private bool _dilationDirty;
        private int _frameCount;
        private float _lastLogTime;

        private const string ScenePermission = "com.oculus.permission.USE_SCENE";

        public event Action Updated;

        private static readonly Vector3 ScaleFlipZ = new(1, 1, -1);

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            EnsureARSession();

            _arOcclusionManager = FindFirstObjectByType<AROcclusionManager>();
            if (!_arOcclusionManager)
                throw new Exception("[RoomScan] AROcclusionManager not found in scene");

            _xrOrigin = FindFirstObjectByType<Unity.XR.CoreUtils.XROrigin>();

            _normKernel = new ComputeKernelHelper(depthNormalCompute, "DepthNorm");
            _monoConvertKernel = new ComputeKernelHelper(depthNormalCompute, "MonoRawDepthToStereo");
            _initDilateKernel = new ComputeKernelHelper(depthDilationCompute, "InitDepthDilation");
            _dilateStepKernel = new ComputeKernelHelper(depthDilationCompute, "DilateDepthStep");

            _dilationMaxStep = 1;
            for (int i = 0; i < dilationSteps; i++)
                _dilationMaxStep *= 2;

            // Disable occlusion manager initially, enable after permission is confirmed
            _arOcclusionManager.enabled = false;
            CheckPermissionAndEnable();

            _started = true;
        }

        private void EnsureARSession()
        {
            if (FindFirstObjectByType<ARSession>() == null)
            {
                var go = new GameObject("[AR Session]");
                go.AddComponent<ARSession>();
                Debug.Log("[RoomScan] Created ARSession (was missing from scene)");
            }
        }

        private void CheckPermissionAndEnable()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (Permission.HasUserAuthorizedPermission(ScenePermission))
            {
                EnableOcclusion();
            }
            else
            {
                Debug.Log("[RoomScan] Requesting USE_SCENE permission...");
                var callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += _ => EnableOcclusion();
                callbacks.PermissionDenied += _ => Debug.LogError("[RoomScan] USE_SCENE permission denied — depth will not work");
                Permission.RequestUserPermission(ScenePermission, callbacks);
            }
#else
            EnableOcclusion();
#endif
        }

        private bool _subscribed;

        private async void EnableOcclusion()
        {
            if (_arOcclusionManager == null) return;

            Debug.Log("[RoomScan] Enabling AROcclusionManager...");

            // Unsubscribe first to avoid double subscription
            _arOcclusionManager.frameReceived -= OnDepthFrame;
            _arOcclusionManager.enabled = false;

            // Wait a frame before re-enabling (same pattern as lasertag reference)
            await Awaitable.NextFrameAsync();
            await Awaitable.NextFrameAsync();

            if (_arOcclusionManager == null) return;

            _arOcclusionManager.enabled = true;
            _arOcclusionManager.frameReceived += OnDepthFrame;
            _subscribed = true;

            await Awaitable.NextFrameAsync();
            await Awaitable.NextFrameAsync();

            if (_arOcclusionManager == null) return;
            var sub = _arOcclusionManager.subsystem;
            Debug.Log($"[RoomScan] Occlusion subsystem: {(sub != null ? sub.GetType().Name : "null")}, running={sub?.running}");
        }

        private void OnApplicationPause(bool paused)
        {
            if (!_started) return;

            if (paused)
            {
                if (_arOcclusionManager != null)
                {
                    _arOcclusionManager.frameReceived -= OnDepthFrame;
                    _arOcclusionManager.enabled = false;
                    _subscribed = false;
                }
            }
            else
            {
                CheckPermissionAndEnable();
            }
        }

        private void OnDisable()
        {
            if (_arOcclusionManager != null && _subscribed)
            {
                _arOcclusionManager.frameReceived -= OnDepthFrame;
                _subscribed = false;
            }
        }

        private void OnDestroy()
        {
            if (_normTex) Destroy(_normTex);
            if (_dilationA) Destroy(_dilationA);
            if (_dilationB) Destroy(_dilationB);
            if (_simulatedDepthTex) Destroy(_simulatedDepthTex);
        }

        private void Update()
        {
            float t = Time.unscaledTime;
            if (t - _lastLogTime >= 5f)
            {
                _lastLogTime = t;
                var sub = _arOcclusionManager != null ? _arOcclusionManager.subsystem : null;
                Debug.Log($"[RoomScan] DepthCapture: frames={_frameCount}, depthAvail={DepthAvailable}, " +
                          $"occMgr.enabled={_arOcclusionManager?.enabled}, sub={sub?.GetType().Name ?? "null"}, " +
                          $"running={sub?.running}");
            }
        }

        private void OnDepthFrame(AROcclusionFrameEventArgs args)
        {
            _frameCount++;
            if (_frameCount <= 3 || _frameCount % 100 == 0)
                Debug.Log($"[RoomScan] OnDepthFrame #{_frameCount}, textures={args.externalTextures.Count}");

            if (Application.isEditor)
                HandleEditorSimulation(args);
            else
                HandleDeviceDepth(args);

            if (!DepthAvailable) return;

            SetGlobalShaderProperties();
            ComputeNormals();
            _dilationDirty = true;

            Updated?.Invoke();
        }

        /// <summary>
        /// Run dilation if depth has been updated since last call.
        /// Called by VolumeIntegrator before integration (not every frame).
        /// </summary>
        public void UpdateDilationIfNeeded()
        {
            if (!_dilationDirty || !DepthAvailable) return;
            ComputeDilation();
            _dilationDirty = false;
        }

        private void HandleEditorSimulation(AROcclusionFrameEventArgs args)
        {
            Texture rawDepth = args.externalTextures[0].texture;
            DepthAvailable = rawDepth != null;
            if (!DepthAvailable) return;

            if (_simulatedDepthTex == null ||
                _simulatedDepthTex.width != rawDepth.width ||
                _simulatedDepthTex.height != rawDepth.height)
            {
                if (_simulatedDepthTex) Destroy(_simulatedDepthTex);
                _simulatedDepthTex = new RenderTexture(rawDepth.width, rawDepth.height, 0,
                    GraphicsFormat.R16_UNorm, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    enableRandomWrite = true
                };
            }

            _monoConvertKernel.Set(DepthTexRWID, _simulatedDepthTex);
            _monoConvertKernel.Set(InputRawMonoDepthID, rawDepth);
            _monoConvertKernel.DispatchFit(rawDepth.width, rawDepth.height);
            _depthTex = _simulatedDepthTex;

            if (!_mainCam) _mainCam = Camera.main;
            if (!_mainCam) return;

            Matrix4x4 p = _mainCam.projectionMatrix;
            Matrix4x4 pi = p.inverse;
            Transform ct = _mainCam.transform;
            Matrix4x4 vi = Matrix4x4.TRS(ct.position, ct.rotation, ScaleFlipZ);
            Matrix4x4 v = vi.inverse;

            for (int i = 0; i < 2; i++)
            {
                _proj[i] = p;
                _projInv[i] = pi;
                _view[i] = v;
                _viewInv[i] = vi;
            }

            _planes = new Vector2(_mainCam.nearClipPlane, _mainCam.farClipPlane);
        }

        private void HandleDeviceDepth(AROcclusionFrameEventArgs args)
        {
            _depthTex = args.externalTextures[0].texture;

            ReadOnlyList<XRFov> fovs = default;
            ReadOnlyList<Pose> poses = default;
            XRNearFarPlanes depthPlanes = default;

            DepthAvailable = _depthTex != null &&
                             args.TryGetFovs(out fovs) &&
                             args.TryGetPoses(out poses) &&
                             args.TryGetNearFarPlanes(out depthPlanes);

            if (!DepthAvailable) return;

            for (int i = 0; i < 2; i++)
            {
                _proj[i] = CalculateProjectionMatrix(fovs[i], depthPlanes);
                _projInv[i] = Matrix4x4.Inverse(_proj[i]);

                Pose pose = poses[i];
                Matrix4x4 depthFrameMat = Matrix4x4.TRS(pose.position, pose.rotation, ScaleFlipZ);

                Matrix4x4 worldToTracking = _xrOrigin != null
                    ? _xrOrigin.transform.worldToLocalMatrix
                    : Matrix4x4.identity;

                _view[i] = depthFrameMat.inverse * worldToTracking;
                _viewInv[i] = Matrix4x4.Inverse(_view[i]);
            }

            _planes = new Vector2(depthPlanes.nearZ, depthPlanes.farZ);
        }

        private void SetGlobalShaderProperties()
        {
            Shader.SetGlobalMatrixArray(ProjID, _proj);
            Shader.SetGlobalMatrixArray(ProjInvID, _projInv);
            Shader.SetGlobalMatrixArray(ViewID, _view);
            Shader.SetGlobalMatrixArray(ViewInvID, _viewInv);
            Shader.SetGlobalVector(ZParamsID, _planes);
            Shader.SetGlobalVector(TexSizeID, new Vector2(_depthTex.width, _depthTex.height));
            Shader.SetGlobalTexture(DepthTexID, _depthTex);
        }

        private void ComputeNormals()
        {
            if (_normTex == null || _normTex.width != _depthTex.width || _normTex.height != _depthTex.height)
            {
                if (_normTex) Destroy(_normTex);
                _normTex = new RenderTexture(_depthTex.width, _depthTex.height, 0,
                    GraphicsFormat.R8G8B8A8_SNorm, 1)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = 2,
                    useMipMap = false,
                    enableRandomWrite = true
                };
            }

            _normKernel.Set(DepthTexID, _depthTex);
            _normKernel.Set(NormTexRWID, _normTex);
            _normKernel.DispatchFit(_normTex);
            Shader.SetGlobalTexture(NormTexID, _normTex);
        }

        private void ComputeDilation()
        {
            if (_dilationA == null || _dilationA.width != _depthTex.width || _dilationA.height != _depthTex.height)
            {
                if (_dilationA) Destroy(_dilationA);
                if (_dilationB) Destroy(_dilationB);

                var desc = new RenderTextureDescriptor
                {
                    width = _depthTex.width,
                    height = _depthTex.height,
                    volumeDepth = 1,
                    dimension = TextureDimension.Tex2D,
                    autoGenerateMips = false,
                    enableRandomWrite = true,
                    graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat,
                    msaaSamples = 1
                };

                _dilationA = new RenderTexture(desc);
                _dilationB = new RenderTexture(desc);
            }

            depthDilationCompute.SetFloat(VoxDistID, voxelDistance);
            depthDilationCompute.SetFloat(VoxSizeShaderID, voxelSize);

            _initDilateKernel.Set(DepthTexID, _depthTex);
            _initDilateKernel.Set(DilateSrcID, _dilationA);
            _initDilateKernel.Set(DilateDestID, _dilationB);
            _initDilateKernel.DispatchFit(_dilationA.width, _dilationA.height);

            int stepSize = _dilationMaxStep;
            for (int i = 0; i < dilationSteps; i++)
            {
                _dilateStepKernel.Set(DilateSrcID, _dilationA);
                _dilateStepKernel.Set(DilateDestID, _dilationB);
                depthDilationCompute.SetInt(DilateStepSizeID, stepSize);
                _dilateStepKernel.DispatchFit(_dilationA.width, _dilationA.height);

                stepSize /= 2;
                (_dilationA, _dilationB) = (_dilationB, _dilationA);
            }

            _dilatedDepth = _dilationA;
            Shader.SetGlobalTexture(DilatedDepthTexID, _dilatedDepth);
        }

        private static Matrix4x4 CalculateProjectionMatrix(XRFov fov, XRNearFarPlanes planes)
        {
            float left = Mathf.Tan(fov.angleLeft);
            float right = Mathf.Tan(fov.angleRight);
            float bottom = Mathf.Tan(fov.angleDown);
            float top = Mathf.Tan(fov.angleUp);

            float near = planes.nearZ;
            float far = planes.farZ;

            float x = 2.0f / (right - left);
            float y = 2.0f / (top - bottom);
            float a = (right + left) / (right - left);
            float b = (top + bottom) / (top - bottom);

            float c, d;
            if (float.IsInfinity(far))
            {
                c = -1.0f;
                d = -2.0f * near;
            }
            else
            {
                c = -(far + near) / (far - near);
                d = -(2.0f * far * near) / (far - near);
            }

            return new Matrix4x4
            {
                m00 = x,  m01 = 0, m02 = a,  m03 = 0,
                m10 = 0,  m11 = y, m12 = b,  m13 = 0,
                m20 = 0,  m21 = 0, m22 = c,  m23 = d,
                m30 = 0,  m31 = 0, m32 = -1, m33 = 0
            };
        }

        /// <summary>
        /// Update voxel parameters used by dilation (called by VolumeIntegrator when its values change).
        /// </summary>
        public void SetVoxelParams(float voxDist, float voxSize)
        {
            voxelDistance = voxDist;
            voxelSize = voxSize;
        }
    }
}
