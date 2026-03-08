using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    public class KeyframeStore : MonoBehaviour
    {
        public static KeyframeStore Instance { get; private set; }

        [SerializeField, Range(2, 16)] private int maxKeyframes = 8;
        [SerializeField] private float moveThreshold = 0.3f;
        [SerializeField] private float rotateThresholdDeg = 20f;
        [SerializeField, Range(1f, 10f)] private float exposure = 3f;

        private RenderTexture _texArray;
        private RenderTexture _blitTemp;

        private int _count;
        private int _nextHistoricalSlot = 1;
        private Vector3[] _keyframePositions;
        private Quaternion[] _keyframeRotations;
        private Vector4[] _projVectors;

        static readonly int KeyframeTexID = Shader.PropertyToID("_RSKeyframeTex");
        static readonly int KeyframeDataID = Shader.PropertyToID("_RSKeyframeData");
        static readonly int KeyframeCountID = Shader.PropertyToID("_RSKeyframeCount");
        static readonly int KeyframeExposureID = Shader.PropertyToID("_RSCamExposure");

        private const int Vec4sPerKeyframe = 7;

        private void Awake() => Instance = this;

        private void Start()
        {
            _keyframePositions = new Vector3[maxKeyframes];
            _keyframeRotations = new Quaternion[maxKeyframes];
            _projVectors = new Vector4[maxKeyframes * Vec4sPerKeyframe];

            long bytes = 1280L * 960 * 4 * maxKeyframes;
            Debug.Log($"[RoomScan] Keyframe store: {maxKeyframes} slots, 1280x960 RGBA8 = {bytes / (1024 * 1024)}MB");
        }

        private void OnDestroy()
        {
            if (_texArray) Destroy(_texArray);
            if (_blitTemp) Destroy(_blitTemp);
        }

        private void EnsureTexArray(int w, int h)
        {
            if (_texArray != null && _texArray.width == w && _texArray.height == h) return;
            if (_texArray) Destroy(_texArray);

            _texArray = new RenderTexture(w, h, 0, GraphicsFormat.R8G8B8A8_SRGB)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = maxKeyframes,
                enableRandomWrite = false,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                useMipMap = false,
                name = "KeyframeArray"
            };
            _texArray.Create();
            _count = 0;
            _nextHistoricalSlot = 1;
        }

        private void EnsureBlitTemp(int w, int h)
        {
            if (_blitTemp != null && _blitTemp.width == w && _blitTemp.height == h) return;
            if (_blitTemp) Destroy(_blitTemp);
            _blitTemp = new RenderTexture(w, h, 0, GraphicsFormat.R8G8B8A8_SRGB)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _blitTemp.Create();
        }

        public void SetLiveFrame(Texture frame, Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (frame == null) return;
            EnsureTexArray(frame.width, frame.height);
            EnsureBlitTemp(frame.width, frame.height);

            Graphics.Blit(frame, _blitTemp);
            Graphics.CopyTexture(_blitTemp, 0, 0, _texArray, 0, 0);

            _keyframePositions[0] = camPos;
            _keyframeRotations[0] = camRot;
            PackProjectionData(0, camPos, camRot, focalLen, principalPt, sensorRes, currentRes);

            if (_count == 0) _count = 1;
            UpdateShaderGlobals();
        }

        public void TryInsertKeyframe(Texture frame, Vector3 camPos, Quaternion camRot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            if (frame == null || _texArray == null) return;

            for (int i = 1; i < _count; i++)
            {
                float dist = Vector3.Distance(camPos, _keyframePositions[i]);
                float angle = Quaternion.Angle(camRot, _keyframeRotations[i]);
                if (dist < moveThreshold && angle < rotateThresholdDeg)
                    return;
            }

            int slot = _nextHistoricalSlot;
            if (slot >= maxKeyframes) slot = 1;

            EnsureBlitTemp(frame.width, frame.height);
            Graphics.Blit(frame, _blitTemp);
            Graphics.CopyTexture(_blitTemp, 0, 0, _texArray, slot, 0);

            _keyframePositions[slot] = camPos;
            _keyframeRotations[slot] = camRot;
            PackProjectionData(slot, camPos, camRot, focalLen, principalPt, sensorRes, currentRes);

            _nextHistoricalSlot = slot + 1;
            if (_nextHistoricalSlot >= maxKeyframes)
                _nextHistoricalSlot = 1;

            if (slot >= _count) _count = slot + 1;
            UpdateShaderGlobals();
        }

        private void PackProjectionData(int slot, Vector3 pos, Quaternion rot,
            Vector2 focalLen, Vector2 principalPt, Vector2 sensorRes, Vector2 currentRes)
        {
            int o = slot * Vec4sPerKeyframe;
            var invRot = Matrix4x4.Rotate(Quaternion.Inverse(rot));

            _projVectors[o + 0] = new Vector4(pos.x, pos.y, pos.z, 0);

            // invRot rows as float4s
            _projVectors[o + 1] = invRot.GetRow(0);
            _projVectors[o + 2] = invRot.GetRow(1);
            _projVectors[o + 3] = invRot.GetRow(2);
            _projVectors[o + 4] = invRot.GetRow(3);

            _projVectors[o + 5] = new Vector4(focalLen.x, focalLen.y, principalPt.x, principalPt.y);
            _projVectors[o + 6] = new Vector4(sensorRes.x, sensorRes.y, currentRes.x, currentRes.y);
        }

        private void UpdateShaderGlobals()
        {
            if (_texArray == null) return;

            Shader.SetGlobalTexture(KeyframeTexID, _texArray);
            Shader.SetGlobalVectorArray(KeyframeDataID, _projVectors);
            Shader.SetGlobalInt(KeyframeCountID, _count);
            Shader.SetGlobalFloat(KeyframeExposureID, exposure);
        }
    }
}
