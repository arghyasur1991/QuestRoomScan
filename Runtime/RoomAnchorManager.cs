using System;
using System.Collections;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Anchors the TSDF volume to the physical room via MRUK (required package).
    /// Recomputes volume/world matrices from the room anchor via <see cref="RefreshVolumeTransform"/>
    /// (called from <see cref="RoomScanner"/> every frame while enabled) so the mesh stays aligned when
    /// tracking recenters. Disable this component in the inspector to use identity volume placement.
    /// </summary>
    [DisallowMultipleComponent]
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        /// <summary>
        /// Volume center offset in room-anchor-local space. Set when the room is first ready
        /// (world origin in room space) or restored from a saved scan.
        /// </summary>
        public Vector3 OriginInRoomSpace { get; private set; }

        /// <summary>
        /// When false, first <see cref="OnSceneLoaded"/> assigns default origin (world origin in room space).
        /// Set true after any explicit origin (persistence or <see cref="SetOriginInRoomSpace"/>) so we never
        /// treat Vector3.zero as "unset" — a saved origin of (0,0,0) is valid.
        /// </summary>
        private bool _volumeOriginLocked;

        /// <summary>
        /// When true (loaded/saved with scan.bin v3), volume placement uses saved room + volume snapshots:
        /// <c>volumeToWorld = R_now * Inverse(R_save) * V_save</c> so mesh stays aligned when the MRUK room
        /// anchor gets a different world pose across sessions.
        /// </summary>
        private bool _sessionRelocationActive;

        private Matrix4x4 _sessionSavedRoomLocalToWorld = Matrix4x4.identity;
        private Matrix4x4 _sessionSavedVolumeToWorld = Matrix4x4.identity;

        private MRUK _mruk;
        private Transform _roomTransform;
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

            MRUKRoom room = _mruk.Rooms[0];
            _roomTransform = room != null ? room.transform : null;
            if (_roomTransform == null)
            {
                Debug.LogWarning("[RoomAnchor] No room transform");
                IsRoomLoaded = true;
                RoomReady?.Invoke();
                return;
            }

            if (!_volumeOriginLocked)
            {
                OriginInRoomSpace = _roomTransform.InverseTransformPoint(Vector3.zero);
                Debug.Log($"[RoomAnchor] Default volume origin (world origin in room space): {OriginInRoomSpace}");
            }

            _volumeOriginLocked = true;

            RefreshVolumeTransform();

            IsRoomLoaded = true;
            var roomPos = _roomTransform.position;
            var roomRot = _roomTransform.rotation.eulerAngles;
            Debug.Log($"[RoomAnchor] Room loaded — origin={OriginInRoomSpace}, " +
                      $"roomWorldPos={roomPos}, roomWorldRot={roomRot}");
            RoomReady?.Invoke();
        }

        /// <summary>
        /// Restore persisted volume origin (room-local). Call before applying loaded volume data.
        /// </summary>
        public void SetOriginInRoomSpace(Vector3 origin)
        {
            OriginInRoomSpace = origin;
            _volumeOriginLocked = true;
            Debug.Log($"[RoomAnchor] Volume origin set (persisted or manual): {origin}");
        }

        /// <summary>
        /// Room → world matrix for persistence (identity if room not ready). Call on main thread.
        /// </summary>
        public Matrix4x4 GetRoomLocalToWorldForPersistence()
        {
            if (_roomTransform == null)
                return Matrix4x4.identity;
            return _roomTransform.localToWorldMatrix;
        }

        /// <summary>
        /// Apply snapshots from scan.bin v3. <paramref name="roomLocalToWorldAtSave"/> and
        /// <paramref name="volumeToWorldAtSave"/> must be captured on the main thread at save time.
        /// </summary>
        public void ApplySessionRelocationSnapshots(Matrix4x4 roomLocalToWorldAtSave,
            Matrix4x4 volumeToWorldAtSave)
        {
            _sessionSavedRoomLocalToWorld = roomLocalToWorldAtSave;
            _sessionSavedVolumeToWorld = volumeToWorldAtSave;
            _sessionRelocationActive = true;
            _volumeOriginLocked = true;
            Debug.Log("[RoomAnchor] Session relocation active: V = R_now * Inverse(R_save) * V_save");
        }

        /// <summary>
        /// Disables v3 relocation (e.g. after clearing scan). Live placement uses room + <see cref="OriginInRoomSpace"/>.
        /// </summary>
        public void ClearSessionRelocation()
        {
            _sessionRelocationActive = false;
            _sessionSavedRoomLocalToWorld = Matrix4x4.identity;
            _sessionSavedVolumeToWorld = Matrix4x4.identity;
        }

        /// <summary>
        /// After clearing the TSDF volume for a fresh scan, rebind origin to current world zero in room space
        /// and drop relocation so integration uses the same rule as a first-time room load.
        /// </summary>
        public void NotifyClearedVolumeForRescan()
        {
            ClearSessionRelocation();
            if (_roomTransform != null)
            {
                OriginInRoomSpace = _roomTransform.InverseTransformPoint(Vector3.zero);
                _volumeOriginLocked = true;
            }

            RefreshVolumeTransform();
        }

        /// <summary>
        /// Recompute volume ↔ world from the current room anchor pose. Safe to call every frame.
        /// </summary>
        public void RefreshVolumeTransform()
        {
            if (!enabled || _roomTransform == null)
                return;

            if (_volumeIntegrator == null)
                _volumeIntegrator = VolumeIntegrator.Instance;
            if (_volumeIntegrator == null)
                return;

            Matrix4x4 volumeToWorld;
            if (_sessionRelocationActive)
            {
                Matrix4x4 rNow = _roomTransform.localToWorldMatrix;
                volumeToWorld = rNow * _sessionSavedRoomLocalToWorld.inverse * _sessionSavedVolumeToWorld;
            }
            else
            {
                volumeToWorld = _roomTransform.localToWorldMatrix *
                                Matrix4x4.Translate(OriginInRoomSpace);
            }

            Matrix4x4 worldToVolume = volumeToWorld.inverse;
            _volumeIntegrator.SetVolumeTransform(volumeToWorld, worldToVolume);
        }
    }
}
