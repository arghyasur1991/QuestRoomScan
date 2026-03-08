using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class VolumeIntegrator : MonoBehaviour
    {
        public static VolumeIntegrator Instance { get; private set; }

        [SerializeField] private ComputeShader compute;

        [Header("Volume")]
        [SerializeField] private int3 voxelCount = new(128, 64, 128);
        [SerializeField] private float voxelSize = 0.05f;
        [SerializeField] private float voxelDistance = 0.2f;
        [SerializeField] private float voxelMin = 0.1f;

        [Header("Integration")]
        [SerializeField] private float depthDisparityThreshold = 1f;
        [SerializeField] private float maxUpdateDist = 3.5f;
        [SerializeField] private float minUpdateDist = 0.3f;
        [SerializeField] private int maxFrustumPositions = 500000;

        private RenderTexture _volume;
        public RenderTexture Volume => _volume;
        public int3 VoxelCount => voxelCount;
        public float VoxelSize => voxelSize;
        public float VoxelDistance => voxelDistance;

        // Shader IDs
        private static readonly int VolumeRWID = Shader.PropertyToID("gsVolumeRW");
        private static readonly int VolumeID = Shader.PropertyToID("gsVolume");
        private static readonly int VoxCountID = Shader.PropertyToID("gsVoxCount");
        private static readonly int VoxSizeID = Shader.PropertyToID("gsVoxSize");
        private static readonly int VoxMinID = Shader.PropertyToID("gsVoxMin");
        private static readonly int VoxDistID = Shader.PropertyToID("gsVoxDist");
        private static readonly int FrustumVolumeID = Shader.PropertyToID("gsFrustumVolume");
        private static readonly int DepthDispThreshID = Shader.PropertyToID("gsDepthDispThresh");
        private static readonly int NumExclusionsID = Shader.PropertyToID("gsNumExclusions");
        private static readonly int ExclusionHeadsID = Shader.PropertyToID("gsExclusionHeads");

        private ComputeKernelHelper _clearKernel;
        private ComputeKernelHelper _integrateKernel;

        private ComputeBuffer _frustumVolume;
        private bool _frustumReady;

        public readonly List<Transform> ExclusionZones = new();
        private readonly Vector4[] _exclusionPositions = new Vector4[64];

        public event Action Integrated;
        public event Action Cleared;

        private void Awake()
        {
            Instance = this;
        }

        private void Start()
        {
            CreateVolume();

            _clearKernel = new ComputeKernelHelper(compute, "Clear");
            _clearKernel.Set(VolumeRWID, _volume);

            _integrateKernel = new ComputeKernelHelper(compute, "Integrate");
            _integrateKernel.Set(VolumeRWID, _volume);

            SetShaderConstants();
            Clear();

            if (DepthCapture.Instance != null)
                DepthCapture.Instance.SetVoxelParams(voxelDistance, voxelSize);
        }

        private void OnDestroy()
        {
            _frustumVolume?.Release();
            if (_volume) Destroy(_volume);
        }

        private void CreateVolume()
        {
            long texBytes = (long)voxelCount.x * voxelCount.y * voxelCount.z;
            Debug.Log($"[RoomScan] TSDF volume: {voxelCount} = {texBytes / (1024 * 1024)}MB");

            _volume = new RenderTexture(voxelCount.x, voxelCount.y, 0, GraphicsFormat.R8_SNorm, 0)
            {
                dimension = TextureDimension.Tex3D,
                volumeDepth = voxelCount.z,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _volume.Create();
        }

        private void SetShaderConstants()
        {
            int3 s = voxelCount;
            compute.SetInts(VoxCountID, s.x, s.y, s.z);
            Shader.SetGlobalVector(VoxCountID, new Vector4(s.x, s.y, s.z, 0));

            compute.SetFloat(VoxSizeID, voxelSize);
            Shader.SetGlobalFloat(VoxSizeID, voxelSize);

            compute.SetFloat(VoxMinID, voxelMin);
            compute.SetFloat(VoxDistID, voxelDistance);
            Shader.SetGlobalFloat(VoxDistID, voxelDistance);

            compute.SetFloat(DepthDispThreshID, depthDisparityThreshold);

            Shader.SetGlobalTexture(VolumeID, _volume);
        }

        public void Clear()
        {
            _clearKernel.Set(VolumeRWID, _volume);
            _clearKernel.DispatchFit(_volume);
            Cleared?.Invoke();
        }

        /// <summary>
        /// Build the frustum volume (froxel buffer) once we have depth data.
        /// Must be called from the main thread after DepthCapture has produced at least one frame.
        /// </summary>
        public void SetupFrustumVolume()
        {
            if (!DepthCapture.DepthAvailable) return;

            Matrix4x4 depthProj = Shader.GetGlobalMatrixArray(DepthCapture.ProjID)[0];
            FrustumPlanes frustum = depthProj.decomposeProjection;
            frustum.zFar = maxUpdateDist;

            var positions = new List<Vector3>(Mathf.Min(maxFrustumPositions, 200000));

            float ls = frustum.left / frustum.zNear;
            float rs = frustum.right / frustum.zNear;
            float ts = frustum.top / frustum.zNear;
            float bs = frustum.bottom / frustum.zNear;

            float step = voxelSize;
            bool capped = false;

            for (float z = frustum.zNear; z < frustum.zFar && !capped; z += step)
            {
                float xMin = ls * z + step;
                float xMax = rs * z - step;
                float yMin = bs * z + step;
                float yMax = ts * z - step;

                for (float x = xMin; x < xMax && !capped; x += step)
                for (float y = yMin; y < yMax; y += step)
                {
                    var v = new Vector3(x, y, -z);
                    float mag = v.magnitude;
                    if (mag > minUpdateDist && mag < maxUpdateDist)
                    {
                        positions.Add(v);
                        if (positions.Count >= maxFrustumPositions)
                        {
                            capped = true;
                            break;
                        }
                    }
                }
            }

            if (positions.Count == 0) return;

            Debug.Log($"[RoomScan] Frustum volume: {positions.Count} positions ({positions.Count * 12 / 1024}KB)");

            _frustumVolume?.Release();
            _frustumVolume = new ComputeBuffer(positions.Count, sizeof(float) * 3);
            _frustumVolume.SetData(positions);
            _integrateKernel.Set(FrustumVolumeID, _frustumVolume);
            _frustumReady = true;
        }

        /// <summary>
        /// Run one integration pass: dilate + integrate depth into the TSDF volume.
        /// Called by RoomScanner at the configured frequency.
        /// </summary>
        public void Integrate()
        {
            var dc = DepthCapture.Instance;
            if (dc == null || !DepthCapture.DepthAvailable) return;
            if (!_frustumReady) SetupFrustumVolume();
            if (!_frustumReady) return;

            dc.UpdateDilationIfNeeded();

            compute.SetMatrixArray(DepthCapture.ViewID, dc.View);
            compute.SetMatrixArray(DepthCapture.ProjID, dc.Proj);
            compute.SetMatrixArray(DepthCapture.ViewInvID, dc.ViewInv);
            compute.SetMatrixArray(DepthCapture.ProjInvID, dc.ProjInv);

            int numExclusions = Mathf.Min(ExclusionZones.Count, 64);
            for (int i = 0; i < numExclusions; i++)
            {
                if (ExclusionZones[i] != null)
                    _exclusionPositions[i] = ExclusionZones[i].position;
            }
            compute.SetInt(NumExclusionsID, numExclusions);
            compute.SetVectorArray(ExclusionHeadsID, _exclusionPositions);

            _integrateKernel.Set(DepthCapture.DepthTexID, dc.DepthTex);
            _integrateKernel.Set(DepthCapture.NormTexID, dc.NormTex);
            _integrateKernel.Set(DepthCapture.DilatedDepthTexID, dc.DilatedDepthTex);

            _integrateKernel.DispatchFit(_frustumVolume.count, 1);

            Integrated?.Invoke();
        }

        public float3 VoxelToWorld(uint3 indices)
        {
            float3 pos = indices;
            pos += 0.5f;
            pos -= (float3)VoxelCount / 2.0f;
            pos *= voxelSize;
            return pos;
        }

        public int3 WorldToVoxel(float3 pos)
        {
            pos /= voxelSize;
            pos += (float3)VoxelCount / 2.0f;
            int3 id = (int3)math.floor(pos);
            id = math.clamp(id, int3.zero, VoxelCount);
            return id;
        }
    }
}
