using System;
using System.Collections;
using UnityEngine;
#if HAS_MRUK
using Meta.XR.MRUtilityKit;
#endif
#if HAS_META_OPENXR
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features.Meta;
#endif

namespace Genesis.RoomScan
{
    /// <summary>
    /// Anchors the TSDF volume to the physical room via MRUK.
    /// Recomputes volumeToWorld/worldToVolume every frame from the room anchor
    /// so the mesh stays stable across tracking recenters and boundary exits.
    /// Also handles boundaryless mode suppression.
    /// </summary>
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        /// <summary>
        /// Volume center offset in room-anchor-local space. Fixed at scan start,
        /// persisted across sessions. Room anchor's world transform changes on
        /// recenter but this stays constant, keeping the volume locked to the room.
        /// </summary>
        public Vector3 OriginInRoomSpace { get; private set; }

#if HAS_MRUK
        private MRUK _mruk;
        private MRUKRoom _room;
#endif

        private VolumeIntegrator _volumeIntegrator;

        private void Awake()
        {
            Instance = this;
        }

        private IEnumerator Start()
        {
            _volumeIntegrator = VolumeIntegrator.Instance;
            if (_volumeIntegrator == null)
                _volumeIntegrator = FindFirstObjectByType<VolumeIntegrator>();

#if HAS_MRUK
            _mruk = FindFirstObjectByType<MRUK>();
            if (_mruk == null)
            {
                var go = new GameObject("[MRUK]");
                go.transform.SetParent(transform, false);
                _mruk = go.AddComponent<MRUK>();
            }

            _mruk.SceneSettings ??= new MRUK.MRUKSettings();
            _mruk.SceneSettings.DataSource = MRUK.SceneDataSource.DeviceOnly;
            _mruk.SceneSettings.LoadSceneOnStartup = false;

            _mruk.RegisterSceneLoadedCallback(OnSceneLoaded);

            yield return null;
            _mruk.LoadSceneFromDevice();
            Debug.Log("[RoomAnchor] Loading MRUK scene from device...");
#else
            Debug.LogWarning("[RoomAnchor] MRUK not available, using identity volume transform");
            IsRoomLoaded = true;
            RoomReady?.Invoke();
            yield break;
#endif
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
#if HAS_MRUK
            if (_mruk != null && _mruk.SceneLoadedEvent != null)
                _mruk.SceneLoadedEvent.RemoveListener(OnSceneLoaded);
#endif
        }

        private void OnSceneLoaded()
        {
#if HAS_MRUK
            if (_mruk.Rooms == null || _mruk.Rooms.Count == 0)
            {
                Debug.LogWarning("[RoomAnchor] MRUK loaded but no rooms found");
                IsRoomLoaded = true;
                RoomReady?.Invoke();
                return;
            }

            _room = _mruk.GetCurrentRoom() ?? _mruk.Rooms[0];

            if (OriginInRoomSpace == Vector3.zero)
            {
                var bounds = _room.GetRoomBounds();
                OriginInRoomSpace = bounds.center;
                Debug.Log($"[RoomAnchor] Volume origin set to room center: {OriginInRoomSpace}");
            }

            UpdateVolumeTransform();
            TrySuppressBoundary();

            IsRoomLoaded = true;
            Debug.Log($"[RoomAnchor] Room loaded, origin={OriginInRoomSpace}");
            RoomReady?.Invoke();
#endif
        }

        private void LateUpdate()
        {
#if HAS_MRUK
            if (_room != null && _volumeIntegrator != null)
                UpdateVolumeTransform();
#endif
        }

        /// <summary>
        /// Restore a previously persisted volume origin.
        /// Call before scanning starts (i.e. before RoomReady fires or in its handler).
        /// </summary>
        public void SetOriginInRoomSpace(Vector3 origin)
        {
            OriginInRoomSpace = origin;
            Debug.Log($"[RoomAnchor] Restored volume origin from persistence: {origin}");
        }

#if HAS_MRUK
        private void UpdateVolumeTransform()
        {
            if (_room == null || _volumeIntegrator == null) return;

            Matrix4x4 volumeToWorld = _room.transform.localToWorldMatrix *
                                      Matrix4x4.Translate(OriginInRoomSpace);
            Matrix4x4 worldToVolume = volumeToWorld.inverse;

            _volumeIntegrator.SetVolumeTransform(volumeToWorld, worldToVolume);
        }
#endif

        private void TrySuppressBoundary()
        {
#if HAS_META_OPENXR
            var feature = OpenXRSettings.Instance?.GetFeature<BoundaryVisibilityFeature>();
            if (feature != null && feature.enabled)
            {
                var result = feature.TryRequestBoundaryVisibility(
                    XrBoundaryVisibility.VisibilitySuppressed);
                Debug.Log($"[RoomAnchor] Boundary suppression request: {result}");
            }
            else
            {
                Debug.LogWarning("[RoomAnchor] BoundaryVisibilityFeature not available or not enabled");
            }
#endif
        }
    }
}
