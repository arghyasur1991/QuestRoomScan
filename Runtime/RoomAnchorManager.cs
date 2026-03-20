using System;
using System.Collections;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// MRUK floor anchor for <b>session relocation only</b> (saved scans). Depth fusion uses tracking/world
    /// with <see cref="VolumeIntegrator"/>’s default <c>volumeToWorld = I</c> — we keep that during live scan.
    /// After load, <c>V = A_now * Inverse(A_save) * V_save</c> maps the stored volume into the current session
    /// using the delta between anchor poses (T₁ vs T at save). Uses <see cref="MRUKRoom.FloorAnchors"/>[0]
    /// like <c>SceneMeshManager</c>; falls back to room root if missing.
    /// </summary>
    [DisallowMultipleComponent]
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        /// <summary>
        /// World origin expressed in floor-anchor local space (metadata for <c>scan.bin</c> / v2 field).
        /// Live scanning does <b>not</b> use this for <see cref="VolumeIntegrator"/> — fusion stays identity.
        /// </summary>
        public Vector3 OriginInRoomSpace { get; private set; }

        /// <summary>
        /// When false, first <see cref="OnSceneLoaded"/> assigns default origin (world origin in floor-anchor space).
        /// Set true after any explicit origin (persistence or <see cref="SetOriginInRoomSpace"/>) so we never
        /// treat Vector3.zero as "unset" — a saved origin of (0,0,0) is valid.
        /// </summary>
        private bool _volumeOriginLocked;

        private Matrix4x4 _sessionSavedAnchorLocalToWorld = Matrix4x4.identity;
        private Matrix4x4 _sessionSavedVolumeToWorld = Matrix4x4.identity;

        private MRUK _mruk;
        /// <summary>Floor <see cref="MRUKAnchor"/> transform when available; else room root.</summary>
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
            // MRUK API is async Task; fire-and-forget — SceneLoadedEvent runs when complete.
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
                {
                    Debug.Log($"[RoomAnchor] Using floor MRUKAnchor '{floorAnchor.name}' " +
                              $"(label={floorAnchor.Label}) pos={_anchorTransform.position}, rot={_anchorTransform.rotation.eulerAngles}");
                }
                else
                {
                    Debug.LogWarning("[RoomAnchor] No FloorAnchors — falling back to MRUKRoom.transform " +
                                     $"(pos={_anchorTransform.position})");
                }
            }
            _sessionSavedAnchorLocalToWorld = _anchorTransform.localToWorldMatrix;
            _sessionSavedVolumeToWorld = Matrix4x4.identity;

            _volumeOriginLocked = true;

            RefreshVolumeTransform();

            IsRoomLoaded = true;
            var ap = _anchorTransform.position;
            var ar = _anchorTransform.rotation.eulerAngles;
            Debug.Log($"[RoomAnchor] Room ready — live fusion uses volumeToWorld=I; anchor for relocation only. " +
                      $"originInAnchorSpace(meta)={OriginInRoomSpace}, anchor pos={ap}, rot={ar}");
            RoomReady?.Invoke();
        }

        /// <summary>
        /// Restore persisted volume origin (floor-anchor–local). Call before applying loaded volume data.
        /// </summary>
        public void SetOriginInRoomSpace(Vector3 origin)
        {
            OriginInRoomSpace = origin;
            _volumeOriginLocked = true;
            Debug.Log($"[RoomAnchor] Volume origin set (persisted or manual): {origin}");
        }

        /// <summary>
        /// Floor MRUK anchor → world matrix for persistence (identity if anchor not ready). Main thread only.
        /// </summary>
        public Matrix4x4 GetRoomLocalToWorldForPersistence()
        {
            if (_anchorTransform == null)
                return Matrix4x4.identity;
            return _anchorTransform.localToWorldMatrix;
        }

        /// <summary>
        /// Apply snapshots from scan.bin v3. <paramref name="roomLocalToWorldAtSave"/> is the saved
        /// floor-anchor <c>localToWorld</c> (field name kept for format); <paramref name="volumeToWorldAtSave"/>
        /// is <see cref="VolumeIntegrator.VolumeToWorld"/> at save.
        /// </summary>
        public void ApplySessionRelocationSnapshots(Matrix4x4 roomLocalToWorldAtSave,
            Matrix4x4 volumeToWorldAtSave)
        {
            _sessionSavedAnchorLocalToWorld = roomLocalToWorldAtSave;
            _sessionSavedVolumeToWorld = volumeToWorldAtSave;
            _volumeOriginLocked = true;
            Debug.Log("[RoomAnchor] Session relocation active: V = A_now * Inverse(A_save) * V_save");
        }

        /// <summary>
        /// Updates saved A/V matrices after re-saving a loaded scan — keeps relocation correct without
        /// toggling mode. No-op when <see cref="SessionRelocationActive"/> is false.
        /// </summary>
        public void ReplaceSessionRelocationSnapshots(Matrix4x4 anchorLocalToWorldAtSave,
            Matrix4x4 volumeToWorldAtSave)
        {
            _sessionSavedAnchorLocalToWorld = anchorLocalToWorldAtSave;
            _sessionSavedVolumeToWorld = volumeToWorldAtSave;
        }

        /// <summary>
        /// Disables v3 relocation (e.g. after clearing scan). Live placement uses identity volume/world.
        /// </summary>
        public void ClearSessionRelocation()
        {
            _sessionSavedAnchorLocalToWorld = Matrix4x4.identity;
            _sessionSavedVolumeToWorld = Matrix4x4.identity;
        }

        /// <summary>
        /// After clearing the TSDF volume for a fresh scan: drop relocation, refresh anchor metadata for saves,
        /// and set volume transform to identity (same as live scanning).
        /// </summary>
        public void NotifyClearedVolumeForRescan()
        {
            ClearSessionRelocation();
            if (_anchorTransform != null)
            {
                OriginInRoomSpace = _anchorTransform.InverseTransformPoint(Vector3.zero);
                _volumeOriginLocked = true;
            }

            RefreshVolumeTransform();
        }

        /// <summary>
        /// Live: <c>volumeToWorld = I</c> (fusion matches headset/tracking world). Loaded scan:
        /// <c>V = A_now * Inverse(A_save) * V_save</c>. Safe to call every frame.
        /// </summary>
        public void RefreshVolumeTransform()
        {
            if (!enabled)
                return;

            if (_volumeIntegrator == null)
                _volumeIntegrator = VolumeIntegrator.Instance;
            if (_volumeIntegrator == null)
                return;

            if (_anchorTransform == null)
                return;
            Matrix4x4 aNow = _anchorTransform.localToWorldMatrix;
            Debug.Log($"[RoomAnchor] RefreshVolumeTransform: aNow: {aNow}");
            Debug.Log($"[RoomAnchor] RefreshVolumeTransform: _sessionSavedAnchorLocalToWorld: {_sessionSavedAnchorLocalToWorld}");
            Debug.Log($"[RoomAnchor] RefreshVolumeTransform: _sessionSavedVolumeToWorld: {_sessionSavedVolumeToWorld}");
            Matrix4x4 volumeToWorld = aNow * _sessionSavedAnchorLocalToWorld.inverse * _sessionSavedVolumeToWorld;
            
            Debug.Log($"[RoomAnchor] RefreshVolumeTransform: volumeToWorld: {volumeToWorld}");

            Matrix4x4 worldToVolume = volumeToWorld.inverse;
            _volumeIntegrator.SetVolumeTransform(volumeToWorld, worldToVolume);
        }
    }
}
