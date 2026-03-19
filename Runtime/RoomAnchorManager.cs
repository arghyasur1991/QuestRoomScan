using System;
using System.Collections;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Anchors the TSDF volume to the physical room via MRUK (required package).
    /// Uses the first <see cref="MRUKAnchor"/> on the floor (<see cref="MRUKRoom.FloorAnchors"/>), same as
    /// <c>SceneMeshManager</c> — not <see cref="MRUKRoom.transform"/>, which is often identity / not the
    /// stable scene anchor frame. Falls back to room root if no floor anchors exist.
    /// Recomputes volume/world via <see cref="RefreshVolumeTransform"/> every frame while enabled.
    /// </summary>
    [DisallowMultipleComponent]
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        /// <summary>
        /// Volume center offset in <see cref="MRUKAnchor"/> (floor) local space — world origin expressed
        /// in that frame. Set when the room is first ready or restored from a saved scan.
        /// </summary>
        public Vector3 OriginInRoomSpace { get; private set; }

        /// <summary>
        /// When false, first <see cref="OnSceneLoaded"/> assigns default origin (world origin in floor-anchor space).
        /// Set true after any explicit origin (persistence or <see cref="SetOriginInRoomSpace"/>) so we never
        /// treat Vector3.zero as "unset" — a saved origin of (0,0,0) is valid.
        /// </summary>
        private bool _volumeOriginLocked;

        /// <summary>
        /// When true (loaded/saved with scan.bin v3), volume placement uses saved anchor + volume snapshots:
        /// <c>volumeToWorld = A_now * Inverse(A_save) * V_save</c> (A = floor <see cref="MRUKAnchor"/> L2W).
        /// </summary>
        private bool _sessionRelocationActive;

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

            _volumeOriginLocked = true;

            RefreshVolumeTransform();

            IsRoomLoaded = true;
            var ap = _anchorTransform.position;
            var ar = _anchorTransform.rotation.eulerAngles;
            Debug.Log($"[RoomAnchor] Room ready — originInFloorAnchorSpace={OriginInRoomSpace}, " +
                      $"anchorWorldPos={ap}, anchorWorldRot={ar}");
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
            _sessionRelocationActive = true;
            _volumeOriginLocked = true;
            Debug.Log("[RoomAnchor] Session relocation active: V = A_now * Inverse(A_save) * V_save");
        }

        /// <summary>
        /// Disables v3 relocation (e.g. after clearing scan). Live placement uses anchor + <see cref="OriginInRoomSpace"/>.
        /// </summary>
        public void ClearSessionRelocation()
        {
            _sessionRelocationActive = false;
            _sessionSavedAnchorLocalToWorld = Matrix4x4.identity;
            _sessionSavedVolumeToWorld = Matrix4x4.identity;
        }

        /// <summary>
        /// After clearing the TSDF volume for a fresh scan, rebind origin to current world zero in anchor space
        /// and drop relocation so integration uses the same rule as a first-time room load.
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
        /// Recompute volume ↔ world from the current floor MRUK anchor pose. Safe to call every frame.
        /// </summary>
        public void RefreshVolumeTransform()
        {
            if (!enabled || _anchorTransform == null)
                return;

            if (_volumeIntegrator == null)
                _volumeIntegrator = VolumeIntegrator.Instance;
            if (_volumeIntegrator == null)
                return;

            Matrix4x4 volumeToWorld;
            if (_sessionRelocationActive)
            {
                Matrix4x4 aNow = _anchorTransform.localToWorldMatrix;
                volumeToWorld = aNow * _sessionSavedAnchorLocalToWorld.inverse * _sessionSavedVolumeToWorld;
            }
            else
            {
                volumeToWorld = _anchorTransform.localToWorldMatrix *
                                Matrix4x4.Translate(OriginInRoomSpace);
            }

            Matrix4x4 worldToVolume = volumeToWorld.inverse;
            _volumeIntegrator.SetVolumeTransform(volumeToWorld, worldToVolume);
        }
    }
}
