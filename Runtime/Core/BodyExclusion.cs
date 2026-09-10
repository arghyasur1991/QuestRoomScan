using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Packs head / hand / forearm capsules into the GPU exclusion arrays.
    /// Tests run on voxel world position so a wall behind a hand still
    /// integrates when the hand moves. p1.w ≥ 0.5 marks erasable
    /// (hand / forearm) capsules; torso stays 0.
    /// </summary>
    internal static class BodyExclusion
    {
        public const int Max = 64;
        public const float Erasable = 1f;
        public const float Torso = 0f;

        public static int Pack(
            Vector4[] p0, Vector4[] p1,
            Transform head, Transform leftHand, Transform rightHand,
            IList<Transform> extra,
            float torsoRadius, float torsoAbove, float torsoBelow,
            float handRadius, float handHalfLength,
            float forearmRadius, float forearmLength,
            float shoulderDrop, float shoulderLateral)
        {
            int n = 0;
            if (head != null)
                n = AppendCapsule(p0, p1, n,
                    head.position + Vector3.up * torsoAbove,
                    head.position - Vector3.up * torsoBelow,
                    torsoRadius, Torso);

            n = AppendHandArm(p0, p1, n, head, leftHand, left: true,
                handRadius, handHalfLength, forearmRadius, forearmLength,
                shoulderDrop, shoulderLateral);
            n = AppendHandArm(p0, p1, n, head, rightHand, left: false,
                handRadius, handHalfLength, forearmRadius, forearmLength,
                shoulderDrop, shoulderLateral);

            if (extra == null) return n;
            for (int i = 0; i < extra.Count && n < Max; i++)
            {
                var t = extra[i];
                if (t == null || t == head) continue;
                n = AppendCapsule(p0, p1, n,
                    t.position + Vector3.up * torsoAbove,
                    t.position - Vector3.up * torsoBelow,
                    torsoRadius, Torso);
            }
            return n;
        }

        static int AppendHandArm(
            Vector4[] p0, Vector4[] p1, int n,
            Transform head, Transform wrist, bool left,
            float handRadius, float handHalfLength,
            float forearmRadius, float forearmLength,
            float shoulderDrop, float shoulderLateral)
        {
            if (wrist == null) return n;
            Vector3 w = wrist.position;
            Vector3 fwd = wrist.forward;
            n = AppendCapsule(p0, p1, n,
                w - fwd * handHalfLength,
                w + fwd * handHalfLength,
                handRadius, Erasable);

            Vector3 shoulder;
            if (head != null)
            {
                Vector3 lateral = head.right;
                lateral.y = 0f;
                if (lateral.sqrMagnitude < 1e-6f) lateral = Vector3.right;
                else lateral.Normalize();
                if (left) lateral = -lateral;
                shoulder = head.position - Vector3.up * shoulderDrop + lateral * shoulderLateral;
            }
            else
            {
                shoulder = w - Vector3.up * shoulderDrop;
            }

            Vector3 toShoulder = shoulder - w;
            float mag = toShoulder.magnitude;
            if (mag < 1e-4f) return n;
            Vector3 end = w + toShoulder * (forearmLength / mag);
            return AppendCapsule(p0, p1, n, w, end, forearmRadius, Erasable);
        }

        static int AppendCapsule(
            Vector4[] p0, Vector4[] p1, int n,
            Vector3 a, Vector3 b, float radius, float erasable)
        {
            if (n >= Max) return n;
            p0[n] = new Vector4(a.x, a.y, a.z, radius);
            p1[n] = new Vector4(b.x, b.y, b.z, erasable);
            return n + 1;
        }
    }
}
