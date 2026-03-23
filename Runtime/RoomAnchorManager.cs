using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Room anchor manager. Uses MRUK for runtime world-locking and provides
    /// <see cref="OVRSpatialAnchor"/>-based persistence for reliable cross-session relocation.
    /// Computes per-artifact relocation matrices via <c>R = A_now * Inv(A_create)</c>.
    /// </summary>
    [DisallowMultipleComponent]
    public class RoomAnchorManager : MonoBehaviour
    {
        public static RoomAnchorManager Instance { get; private set; }

        public event Action RoomReady;

        public bool IsRoomLoaded { get; private set; }

        private MRUK _mruk;
        private Transform _anchorTransform;

        private OVRSpatialAnchor _activeSpatialAnchor;
        private readonly List<OVRSpatialAnchor.UnboundAnchor> _unboundAnchors = new();

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
        //  MRUK fallback API (unchanged)
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Floor MRUK anchor → world matrix. Used as fallback when spatial anchor
        /// localization fails. Main thread only.
        /// </summary>
        public Matrix4x4 GetRoomLocalToWorldForPersistence()
        {
            return _anchorTransform != null ? _anchorTransform.localToWorldMatrix : Matrix4x4.identity;
        }

        /// <summary>
        /// One-shot relocation: <c>R = A_now * Inv(A_save)</c>.
        /// </summary>
        public static Matrix4x4 ComputeRelocationMatrix(Matrix4x4 anchorNow, Matrix4x4 anchorAtSave)
        {
            Matrix4x4 reloc = anchorNow * anchorAtSave.inverse;
            Debug.Log($"[RoomAnchor] ComputeRelocation: R = A_now * Inv(A_save)\n" +
                      $"  A_save col3(pos): {anchorAtSave.GetColumn(3)}\n" +
                      $"  A_now  col3(pos): {anchorNow.GetColumn(3)}\n" +
                      $"  R      col3(pos): {reloc.GetColumn(3)}");
            return reloc;
        }

        /// <summary>
        /// Overload for backward compat — uses the current MRUK anchor as A_now.
        /// </summary>
        public Matrix4x4 ComputeRelocationMatrix(Matrix4x4 anchorAtSave)
        {
            Matrix4x4 aNow = _anchorTransform != null ? _anchorTransform.localToWorldMatrix : Matrix4x4.identity;
            return ComputeRelocationMatrix(aNow, anchorAtSave);
        }

        // ─────────────────────────────────────────────────────────────
        //  OVRSpatialAnchor API
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Current spatial anchor localization matrix. Valid after
        /// <see cref="CreateAndSaveSpatialAnchorAsync"/> or <see cref="LoadSpatialAnchorAsync"/>.
        /// Returns identity if no spatial anchor is active.
        /// </summary>
        public Matrix4x4 SpatialAnchorMatrix =>
            _activeSpatialAnchor != null
                ? _activeSpatialAnchor.transform.localToWorldMatrix
                : Matrix4x4.identity;

        /// <summary>
        /// Whether a spatial anchor is currently loaded and localized.
        /// </summary>
        public bool HasSpatialAnchor => _activeSpatialAnchor != null;

        /// <summary>
        /// Creates an <see cref="OVRSpatialAnchor"/> at the given world pose, waits for
        /// creation, persists it, and returns the UUID + localToWorld matrix.
        /// Falls back to MRUK anchor position if <paramref name="position"/> is default.
        /// </summary>
        public async Task<(Guid uuid, Matrix4x4 matrix)?> CreateAndSaveSpatialAnchorAsync(
            Vector3 position, Quaternion rotation)
        {
            if (position == Vector3.zero && rotation == Quaternion.identity && _anchorTransform != null)
            {
                position = _anchorTransform.position;
                rotation = _anchorTransform.rotation;
            }

            var go = new GameObject("[SpatialAnchor]");
            go.transform.SetPositionAndRotation(position, rotation);
            var anchor = go.AddComponent<OVRSpatialAnchor>();

            // Wait for async creation (up to 5s)
            float timeout = 5f;
            float elapsed = 0f;
            while (!anchor.Created && elapsed < timeout)
            {
                await Task.Yield();
                elapsed += Time.unscaledDeltaTime;
            }

            if (!anchor.Created)
            {
                Debug.LogError("[RoomAnchor] Spatial anchor creation timed out");
                Destroy(go);
                return null;
            }

            Debug.Log($"[RoomAnchor] Spatial anchor created: {anchor.Uuid}, pos={position}");

            var saveResult = await anchor.SaveAnchorAsync();
            if (!saveResult.Success)
            {
                Debug.LogError($"[RoomAnchor] Spatial anchor save failed: {saveResult.Status}");
                Destroy(go);
                return null;
            }

            Debug.Log($"[RoomAnchor] Spatial anchor persisted: {anchor.Uuid}");

            // Wait a few frames for transform to stabilize
            await StabilizeAnchorTransform(anchor.transform);

            if (_activeSpatialAnchor != null && _activeSpatialAnchor.gameObject != go)
                Destroy(_activeSpatialAnchor.gameObject);
            _activeSpatialAnchor = anchor;

            Matrix4x4 matrix = anchor.transform.localToWorldMatrix;
            return (anchor.Uuid, matrix);
        }

        /// <summary>
        /// Loads a previously persisted spatial anchor by UUID, localizes it, and returns
        /// the anchor's current localToWorld matrix. Returns null on failure.
        /// Falls back to MRUK anchor if localization fails.
        /// </summary>
        public async Task<Matrix4x4?> LoadSpatialAnchorAsync(Guid uuid)
        {
            Debug.Log($"[RoomAnchor] Loading spatial anchor {uuid}...");

            var loadResult = await OVRSpatialAnchor.LoadUnboundAnchorsAsync(
                new[] { uuid }, _unboundAnchors);

            if (!loadResult.Success || _unboundAnchors.Count == 0)
            {
                Debug.LogWarning($"[RoomAnchor] Spatial anchor load failed: {loadResult.Status}, " +
                                 $"count={_unboundAnchors.Count}. Falling back to MRUK.");
                return null;
            }

            var unbound = _unboundAnchors[0];

            bool localized = await unbound.LocalizeAsync();
            if (!localized && !unbound.Localized)
            {
                // Poll for localization (up to 10s)
                float timeout = 10f;
                float elapsed = 0f;
                while (!unbound.Localized && elapsed < timeout)
                {
                    await Task.Yield();
                    elapsed += Time.unscaledDeltaTime;
                }
                if (!unbound.Localized)
                {
                    Debug.LogWarning("[RoomAnchor] Spatial anchor localization timed out. Falling back to MRUK.");
                    return null;
                }
            }

            // Bind to a new OVRSpatialAnchor GO
            var go = new GameObject($"[SpatialAnchor-{uuid:N}]");
            var anchor = go.AddComponent<OVRSpatialAnchor>();
            unbound.BindTo(anchor);

            Debug.Log($"[RoomAnchor] Spatial anchor localized: {uuid}, pos={anchor.transform.position}");

            await StabilizeAnchorTransform(anchor.transform);

            if (_activeSpatialAnchor != null && _activeSpatialAnchor.gameObject != go)
                Destroy(_activeSpatialAnchor.gameObject);
            _activeSpatialAnchor = anchor;

            return anchor.transform.localToWorldMatrix;
        }

        /// <summary>
        /// Erases a spatial anchor from persistent storage by UUID.
        /// Does not require the anchor to be loaded.
        /// </summary>
        public async Task<bool> EraseSpatialAnchorAsync(Guid uuid)
        {
            Debug.Log($"[RoomAnchor] Erasing spatial anchor {uuid}...");
            var result = await OVRSpatialAnchor.EraseAnchorsAsync(
                null, new[] { uuid });

            if (result.Success)
                Debug.Log($"[RoomAnchor] Spatial anchor erased: {uuid}");
            else
                Debug.LogWarning($"[RoomAnchor] Spatial anchor erase failed: {result.Status}");

            return result.Success;
        }

        /// <summary>
        /// Waits for an anchor transform to stabilize (5 consecutive frames with &lt; 1mm movement).
        /// </summary>
        private static async Task StabilizeAnchorTransform(Transform t)
        {
            int stableFrames = 0;
            const int required = 5;
            const int maxPolls = 60;
            Vector3 prevPos = t.position;

            for (int i = 0; i < maxPolls && stableFrames < required; i++)
            {
                await Task.Yield();
                float delta = Vector3.Distance(prevPos, t.position);
                if (delta < 0.001f)
                    stableFrames++;
                else
                    stableFrames = 0;
                prevPos = t.position;
            }
        }
    }
}
