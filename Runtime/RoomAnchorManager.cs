using System;
using System.Collections;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// MRUK floor anchor manager. Provides the anchor's current world pose for persistence
    /// (save/load relocation) and unconditionally keeps <c>volumeToWorld = I</c> for live fusion.
    /// The relocation formula <c>R = A_now * Inv(A_save) * V_save</c> is computed transiently
    /// inside <see cref="RoomScanPersistence.LoadAsync"/> to feed the bake pass — it never runs
    /// per-frame, so anchor drift cannot poison the volume transform.
    /// </summary>
    [DisallowMultipleComponent]
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        /// <summary>
        /// World origin expressed in floor-anchor local space (metadata for <c>scan.bin</c> / v2 field).
        /// </summary>
        public Vector3 OriginInRoomSpace { get; private set; }

        private bool _volumeOriginLocked;

        private MRUK _mruk;
        private Transform _anchorTransform;
        private VolumeIntegrator _volumeIntegrator;

        private void Awake()
        {
            Instance = this;
        }

        private IEnumerator Start()
        {
            if (!enabled)
                yield break;

            _volumeIntegrator = VolumeIntegrator.Instance
                ?? FindFirstObjectByType<VolumeIntegrator>();

            _mruk = FindFirstObjectByType<MRUK>();
            if (_mruk == null)
            {
                var go = new GameObject("[MRUK]");
                go.transform.SetParent(transform, false);
                _mruk = go.AddComponent<MRUK>();
            }

            _mruk.SceneSettings ??= new MRUK.MRUKSettings();
            _mruk.SceneSettings.DataSource = MRUK.SceneDataSource.Device;
            _mruk.SceneSettings.LoadSceneOnStartup = false;

            if (_mruk.SceneLoadedEvent != null)
                _mruk.SceneLoadedEvent.AddListener(OnSceneLoaded);

            yield return null;
            _ = _mruk.LoadSceneFromDevice();
            Debug.Log("[RoomAnchor] MRUK LoadSceneFromDevice started (awaiting SceneLoadedEvent)...");
        }

        private void OnDestroy()
        {
            if (_mruk != null && _mruk.SceneLoadedEvent != null)
                _mruk.SceneLoadedEvent.RemoveListener(OnSceneLoaded);
            if (Instance == this)
                Instance = null;
        }

        private void OnSceneLoaded()
        {
            if (!enabled)
                return;

            if (_mruk.Rooms == null || _mruk.Rooms.Count == 0)
            {
                Debug.LogWarning("[RoomAnchor] MRUK loaded but no rooms found");
                IsRoomLoaded = true;
                RoomReady?.Invoke();
                return;
            }

            MRUKRoom room = _mruk.GetCurrentRoom() ?? _mruk.Rooms[0];

            MRUKAnchor floorAnchor = null;
            if (room.FloorAnchors != null && room.FloorAnchors.Count > 0)
                floorAnchor = room.FloorAnchors[0];

            _anchorTransform = floorAnchor != null ? floorAnchor.transform : room.transform;
            if (_anchorTransform == null)
            {
                Debug.LogWarning("[RoomAnchor] No anchor transform");
                IsRoomLoaded = true;
                RoomReady?.Invoke();
                return;
            }

            if (!_volumeOriginLocked)
            {
                OriginInRoomSpace = _anchorTransform.InverseTransformPoint(Vector3.zero);
                Debug.Log($"[RoomAnchor] Default volume origin (world origin in floor-anchor space): {OriginInRoomSpace}");

                if (floorAnchor != null)
                    Debug.Log($"[RoomAnchor] Using floor MRUKAnchor '{floorAnchor.name}' " +
                              $"(label={floorAnchor.Label}) pos={_anchorTransform.position}, rot={_anchorTransform.rotation.eulerAngles}");
                else
                    Debug.LogWarning($"[RoomAnchor] No FloorAnchors — falling back to MRUKRoom.transform (pos={_anchorTransform.position})");
            }

            _volumeOriginLocked = true;
            RefreshVolumeTransform();

            IsRoomLoaded = true;
            Debug.Log($"[RoomAnchor] Room ready — volumeToWorld=I always. " +
                      $"anchor pos={_anchorTransform.position}, rot={_anchorTransform.rotation.eulerAngles}");
            RoomReady?.Invoke();
        }

        // ─────────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────────

        public void SetOriginInRoomSpace(Vector3 origin)
        {
            OriginInRoomSpace = origin;
            _volumeOriginLocked = true;
        }

        /// <summary>
        /// Floor MRUK anchor → world matrix for persistence. Main thread only.
        /// </summary>
        public Matrix4x4 GetRoomLocalToWorldForPersistence()
        {
            return _anchorTransform != null ? _anchorTransform.localToWorldMatrix : Matrix4x4.identity;
        }

        /// <summary>
        /// One-shot relocation: <c>R = A_now * Inv(A_save) * V_save</c>.
        /// Called once during <see cref="RoomScanPersistence.LoadAsync"/> to compute
        /// the matrix for <see cref="VolumeIntegrator.BakeRelocation"/>.
        /// </summary>
        public Matrix4x4 ComputeRelocationMatrix(Matrix4x4 anchorAtSave, Matrix4x4 volumeToWorldAtSave)
        {
            Matrix4x4 aNow = _anchorTransform != null ? _anchorTransform.localToWorldMatrix : Matrix4x4.identity;
            Matrix4x4 reloc = aNow * anchorAtSave.inverse * volumeToWorldAtSave;
            Debug.Log($"[RoomAnchor] ComputeRelocation: R = A_now * Inv(A_save) * V_save\n" +
                      $"  A_save row3: {anchorAtSave.GetRow(3)}\n" +
                      $"  A_now  row3: {aNow.GetRow(3)}\n" +
                      $"  R      row3: {reloc.GetRow(3)}");
            return reloc;
        }

        /// <summary>
        /// After clearing the TSDF volume for a fresh/re-scan: refresh anchor metadata for saves,
        /// and ensure volume transform is identity.
        /// </summary>
        public void NotifyClearedVolumeForRescan()
        {
            if (_anchorTransform != null)
            {
                OriginInRoomSpace = _anchorTransform.InverseTransformPoint(Vector3.zero);
                _volumeOriginLocked = true;
            }
            RefreshVolumeTransform();
        }

        /// <summary>
        /// Unconditionally sets <c>volumeToWorld = I</c>. Anchor drift cannot affect fusion.
        /// </summary>
        public void RefreshVolumeTransform()
        {
            if (!enabled)
                return;
            if (_volumeIntegrator == null)
                _volumeIntegrator = VolumeIntegrator.Instance;
            if (_volumeIntegrator == null)
                return;
            _volumeIntegrator.SetVolumeTransform(Matrix4x4.identity, Matrix4x4.identity);
        }
    }
}
