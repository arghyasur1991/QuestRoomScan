using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// A scene root whose local space <b>is</b> the spatial anchor's space, so
    /// that anything parented under it can be saved as plain local coordinates
    /// and reloaded into the same physical spot on a later run.
    ///
    /// <para>
    /// <b>The problem this solves.</b> Unity's world origin is wherever the
    /// headset happened to boot, so world coordinates mean a different place in
    /// the room on every run. The spatial anchor is the only transform that
    /// re-localizes to the same physical spot, so it is the only frame worth
    /// storing against. Parenting content under the anchor is <i>not</i>
    /// sufficient on its own: <c>SetParent(anchor, worldPositionStays: true)</c>
    /// keeps the child's world pose and expresses it as a local offset, which
    /// tracks the anchor's drift correction — the usual reason for parenting —
    /// but leaves local space equal to world space plus a constant. That
    /// distinction is invisible for as long as you only ever author and look
    /// within one session, and it is the single most common way room-scale
    /// persistence goes wrong.
    /// </para>
    ///
    /// <para>
    /// <b>The invariant.</b> This root's local transform under the anchor is
    /// always identity. Its local space therefore is the anchor's space, and
    /// descendants' local coordinates are meaningful across restarts. When the
    /// anchor binds or changes, direct children keep their world poses, so
    /// nothing visibly moves; their local coordinates are simply re-expressed
    /// in the new frame, which is what you want.
    /// </para>
    ///
    /// <para>
    /// <b>Which recipe to use.</b> Content you can parent belongs here: store
    /// local coordinates, reload them, done. Data with world coordinates baked
    /// into an array — vertex buffers, precomputed fields, anything a transform
    /// cannot move — cannot use this, and must instead store
    /// <see cref="RoomAnchorManager.SpatialAnchorMatrix"/> alongside itself and
    /// be relocated on load by
    /// <see cref="RoomAnchorManager.ComputeRelocationMatrix(Matrix4x4, Matrix4x4)"/>.
    /// That is how the refined scan mesh itself survives a restart. See the
    /// README's persistence section.
    /// </para>
    ///
    /// <para>
    /// <b>Before the anchor binds</b> this root sits at the world origin and
    /// room space is world space. Content authored during that window is placed
    /// correctly for the session but its local coordinates are not yet room
    /// coordinates, so anything that intends to persist should await
    /// <see cref="WaitForBindAsync"/> first rather than assume.
    /// </para>
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("Room Scan/Room Space Root")]
    public class RoomSpaceRoot : MonoBehaviour
    {
        [Tooltip("Log every bind and rebind.")]
        [SerializeField] protected bool verbose = true;

        [Tooltip("Editor/testing only: bind to this Transform instead of polling " +
                 "RoomAnchorManager for a real OVRSpatialAnchor, neither of which " +
                 "work in playmode without a headset. Leave null on device.")]
        [SerializeField] protected Transform anchorOverride;

        /// <summary>The active root, or null before one is in the scene.</summary>
        public static RoomSpaceRoot Instance { get; private set; }

        /// <summary>Fired on every bind and rebind, after children have been
        /// re-framed. Systems holding cached room-space coordinates from an
        /// earlier frame should recompute them here.</summary>
        public event System.Action<Transform> Bound;

        /// <summary>The anchor this root is bound to, or null.</summary>
        public Transform CurrentAnchor => _currentAnchor;

        /// <summary>Whether local space currently means room space. False before
        /// an anchor binds, when this root sits at the world origin.</summary>
        public bool IsBound => _currentAnchor != null;

        /// <summary>Whether a root exists and room space is meaningful.</summary>
        public static bool RoomSpaceReady => Instance != null && Instance.IsBound;

        /// <summary>Transform world coordinates into room space. Identity with no
        /// root in the scene, which makes editor and test paths that never bind
        /// behave as though world space were room space.</summary>
        public static Matrix4x4 WorldToRoom =>
            Instance != null ? Instance.transform.worldToLocalMatrix : Matrix4x4.identity;

        /// <summary>Transform room coordinates into world space.</summary>
        public static Matrix4x4 RoomToWorld =>
            Instance != null ? Instance.transform.localToWorldMatrix : Matrix4x4.identity;

        Transform _currentAnchor;

        protected virtual void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Logger.Warning(
                    $"Duplicate RoomSpaceRoot ('{name}' and '{Instance.name}'). Keeping the " +
                    "first and destroying this one; two roots would mean two definitions of " +
                    "room space.");
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        protected virtual void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        protected virtual void Update()
        {
            Transform anchor = anchorOverride;
            if (anchor == null)
            {
                var manager = RoomAnchorManager.Instance;
                if (manager == null) return;
                anchor = manager.SpatialAnchorTransform;
            }

            if (anchor == null || anchor == _currentAnchor) return;
            Reframe(anchor);
        }

        /// <summary>
        /// Bind to <paramref name="anchor"/>, or to the world origin when it is
        /// null, holding this root's local transform at identity and preserving
        /// every direct child's world pose.
        ///
        /// <para>Children are moved back deliberately rather than reparented
        /// with <c>worldPositionStays</c>: the whole point is that this root's
        /// own local transform must not absorb the difference, because that is
        /// exactly what stops local space from being room space.</para>
        /// </summary>
        protected void Reframe(Transform anchor)
        {
            int count = transform.childCount;
            var children = new Transform[count];
            var positions = new Vector3[count];
            var rotations = new Quaternion[count];
            for (int i = 0; i < count; i++)
            {
                children[i] = transform.GetChild(i);
                positions[i] = children[i].position;
                rotations[i] = children[i].rotation;
            }

            // worldPositionStays:false keeps the local values as they are; the
            // explicit reset covers a root that was moved in the inspector.
            transform.SetParent(anchor, worldPositionStays: false);
            transform.localPosition = Vector3.zero;
            transform.localRotation = Quaternion.identity;
            transform.localScale = Vector3.one;

            // Local scale is left alone: both frames are rigid, so a child's
            // local scale is still correct and reading lossyScale back would
            // only introduce drift through the matrix chain.
            for (int i = 0; i < count; i++)
                children[i].SetPositionAndRotation(positions[i], rotations[i]);

            _currentAnchor = anchor;

            if (verbose)
            {
                Logger.Info(anchor != null
                    ? $"RoomSpaceRoot bound to '{anchor.name}' — local space is now room space. " +
                      $"{count} child(ren) held in place."
                    : $"RoomSpaceRoot unbound — local space is world space again. " +
                      $"{count} child(ren) held in place.");
            }

            Bound?.Invoke(anchor);
        }

        // ─────────────────────── Parenting ───────────────────────

        /// <summary>Reparent an existing GameObject into room space, keeping its
        /// world pose. Its local coordinates afterwards are room coordinates and
        /// are what you persist.</summary>
        public static void Adopt(GameObject go)
        {
            if (go == null) return;
            AttachToRoot(go.transform, keepWorldPose: true);
        }

        /// <summary>
        /// Reparent into room space and sit at the room origin.
        ///
        /// <para>For content whose geometry is already expressed in room
        /// coordinates — meshes built from <see cref="WorldToRoom"/>-transformed
        /// vertices, for instance. Adopting such an object with its world pose
        /// preserved would apply the room transform a second time.</para>
        /// </summary>
        public static void AdoptAtRoomOrigin(GameObject go)
        {
            if (go == null) return;
            AttachToRoot(go.transform, keepWorldPose: false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;
        }

        /// <summary>Instantiate at a world pose and adopt into room space. Use
        /// instead of raw <c>Instantiate</c> for anything that belongs to the
        /// room rather than to the headset.</summary>
        public static T Spawn<T>(T prefab, Vector3 worldPosition, Quaternion worldRotation)
            where T : Object
        {
            if (prefab == null)
            {
                Logger.Warning("RoomSpaceRoot.Spawn called with a null prefab.");
                return null;
            }

            T instance = Instantiate(prefab, worldPosition, worldRotation);
            var go = instance as GameObject ?? (instance as Component)?.gameObject;
            if (go != null) AttachToRoot(go.transform, keepWorldPose: true);
            return instance;
        }

        static void AttachToRoot(Transform t, bool keepWorldPose)
        {
            if (Instance == null)
            {
                Logger.Warning(
                    $"No RoomSpaceRoot in the scene when attaching '{t.name}'. It stays at " +
                    "scene root, will not follow the spatial anchor, and its coordinates will " +
                    "not survive a restart.");
                return;
            }
            t.SetParent(Instance.transform, keepWorldPose);
        }

        // ─────────────────────── Readiness ───────────────────────

        /// <summary>
        /// Wait until room space is meaningful. Returns false on timeout or
        /// cancellation, in which case the caller is authoring in world space
        /// and should not persist local coordinates.
        ///
        /// <para>Resolves within a frame in the common case. Polls rather than
        /// waiting on <see cref="Bound"/> so that a caller starting after the
        /// bind has already happened does not wait for an event that will not
        /// fire again.</para>
        /// </summary>
        public static async Task<bool> WaitForBindAsync(
            float timeoutSeconds = 10f, CancellationToken cancellationToken = default)
        {
            float deadline = Time.realtimeSinceStartup + timeoutSeconds;
            while (!RoomSpaceReady)
            {
                if (cancellationToken.IsCancellationRequested) return false;
                if (Time.realtimeSinceStartup >= deadline)
                {
                    Logger.Warning(
                        $"Room space did not bind within {timeoutSeconds:F0}s. Either no scan " +
                        "anchor exists yet or localization failed; content authored now cannot " +
                        "be persisted to a fixed spot in the room.");
                    return false;
                }
                await Task.Yield();
            }
            return true;
        }

        /// <summary>
        /// Bind to a stand-in anchor without going through MRUK or
        /// <c>OVRSpatialAnchor</c>, for editor playmode and tests. Applies
        /// synchronously so a caller can bake or author against the result in
        /// the same frame. Pass null to return to the world origin.
        /// </summary>
        public void SetAnchorOverride(Transform anchor)
        {
            anchorOverride = anchor;
            if (anchor != _currentAnchor) Reframe(anchor);
        }
    }
}
