using System;
using System.Collections;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// MRUK floor anchor manager. Provides the anchor's current world pose for persistence
    /// and computes the one-shot relocation matrix <c>R = A_now * Inv(A_save)</c> during load.
    /// </summary>
    [DisallowMultipleComponent]
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        private MRUK _mruk;
        private Transform _anchorTransform;

        private void Awake()
        {
            Instance = this;
        }

        private IEnumerator Start()
        {
            if (!enabled)
                yield break;

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

            if (floorAnchor != null)
                Debug.Log($"[RoomAnchor] Using floor MRUKAnchor '{floorAnchor.name}' " +
                          $"(label={floorAnchor.Label}) pos={_anchorTransform.position}, rot={_anchorTransform.rotation.eulerAngles}");
            else
                Debug.LogWarning($"[RoomAnchor] No FloorAnchors — falling back to MRUKRoom.transform (pos={_anchorTransform.position})");

            IsRoomLoaded = true;
            Debug.Log($"[RoomAnchor] Room ready — anchor pos={_anchorTransform.position}, rot={_anchorTransform.rotation.eulerAngles}");
            RoomReady?.Invoke();
        }

        // ─────────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Floor MRUK anchor → world matrix for persistence. Main thread only.
        /// </summary>
        public Matrix4x4 GetRoomLocalToWorldForPersistence()
        {
            return _anchorTransform != null ? _anchorTransform.localToWorldMatrix : Matrix4x4.identity;
        }

        /// <summary>
        /// One-shot relocation: <c>R = A_now * Inv(A_save)</c>.
        /// Called once during <see cref="RoomScanPersistence.LoadAsync"/> to compute
        /// the matrix for <see cref="VolumeIntegrator.BakeRelocation"/>.
        /// </summary>
        public Matrix4x4 ComputeRelocationMatrix(Matrix4x4 anchorAtSave)
        {
            Matrix4x4 aNow = _anchorTransform != null ? _anchorTransform.localToWorldMatrix : Matrix4x4.identity;
            Matrix4x4 reloc = aNow * anchorAtSave.inverse;
            Debug.Log($"[RoomAnchor] ComputeRelocation: R = A_now * Inv(A_save)\n" +
                      $"  A_save col3(pos): {anchorAtSave.GetColumn(3)}\n" +
                      $"  A_save row0: {anchorAtSave.GetRow(0)}\n" +
                      $"  A_now  col3(pos): {aNow.GetColumn(3)}\n" +
                      $"  A_now  row0: {aNow.GetRow(0)}\n" +
                      $"  R      col3(pos): {reloc.GetColumn(3)}\n" +
                      $"  R      row0: {reloc.GetRow(0)}");
            return reloc;
        }
    }
}
