using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Detects dominant planes (walls, floor, ceiling) from mesh vertices using
    /// sequential RANSAC with axis-aligned bias. RANSAC runs on a background thread
    /// to avoid blocking the main thread. Results are applied on LateUpdate.
    /// </summary>
    public class PlaneDetector : MonoBehaviour
    {
        [Header("Detection")]
        [SerializeField, Tooltip("Mesh cycles between plane re-detection runs.")]
        private int detectionInterval = 3;
        [SerializeField, Tooltip("Maximum number of planes to detect.")]
        private int maxPlanes = 6;
        [SerializeField, Tooltip("RANSAC iterations per plane candidate.")]
        private int ransacIterations = 80;
        [SerializeField, Tooltip("Inlier distance threshold (meters).")]
        private float inlierDistance = 0.02f;
        [SerializeField, Tooltip("Minimum inlier count to accept a plane.")]
        private int minInliers = 30;
        [SerializeField, Tooltip("Minimum normal alignment (dot product) for inlier test.")]
        [Range(0.8f, 1f)] private float normalAlignThreshold = 0.95f;

        [Header("Sampling")]
        [SerializeField, Tooltip("Max vertices to sample from all chunks. Caps RANSAC input size.")]
        private int maxSampleVertices = 2048;

        [Header("Axis Bias")]
        [SerializeField, Tooltip("Snap plane normals to axis if within this angle (degrees).")]
        [Range(0f, 20f)] private float axisSnapAngle = 10f;

        [Header("Persistence")]
        [SerializeField, Tooltip("Plane merging distance threshold. Planes closer than this are merged.")]
        private float mergePlaneDistance = 0.05f;
        [SerializeField, Tooltip("Confidence growth per detection cycle.")]
        private float confidenceGrowth = 0.15f;
        [SerializeField, Tooltip("Confidence decay per missed detection cycle.")]
        private float confidenceDecay = 0.05f;

        private NativeArray<PlaneData> _planes;
        private int _planeCount;
        private int _meshCyclesSinceDetection;

        private readonly List<PlaneCandidate> _persistentPlanes = new();

        // Background detection state
        private volatile bool _detectionRunning;
        private volatile bool _detectionReady;
        private List<PlaneCandidate> _pendingResults;

        public NativeArray<PlaneData> Planes => _planes;
        public int PlaneCount => _planeCount;

        private void LateUpdate()
        {
            if (!_detectionReady) return;
            _detectionReady = false;

            if (_pendingResults != null)
            {
                MergeWithPersistent(_pendingResults);
                PublishPlanes();
                _pendingResults = null;
            }

            _detectionRunning = false;
        }

        private void OnDestroy()
        {
            if (_planes.IsCreated) _planes.Dispose();
        }

        public void OnMeshCycleComplete(ChunkManager chunkManager)
        {
            _meshCyclesSinceDetection++;
            if (_meshCyclesSinceDetection < detectionInterval) return;
            if (_detectionRunning) return;
            _meshCyclesSinceDetection = 0;

            CollectAndDetect(chunkManager);
        }

        /// <summary>
        /// Submit pre-collected vertex samples (e.g. from GPU readback).
        /// Skips the Mesh sampling step and goes directly to background RANSAC.
        /// </summary>
        public void SubmitSamples(float3[] positions, float3[] normals, int count)
        {
            if (_detectionRunning) return;
            if (count < minInliers * 2) return;

            _detectionRunning = true;
            int stride = math.max(1, count / maxSampleVertices);
            int sampleCount = (count + stride - 1) / stride;
            var samplePos = new float3[sampleCount];
            var sampleNrm = new float3[sampleCount];
            int idx = 0;
            for (int i = 0; i < count && idx < sampleCount; i += stride)
            {
                samplePos[idx] = positions[i];
                sampleNrm[idx] = normals[i];
                idx++;
            }
            sampleCount = idx;

            LaunchBackgroundDetection(samplePos, sampleNrm, sampleCount);
        }

        private void CollectAndDetect(ChunkManager chunkManager)
        {
            int totalVertCount = 0;
            foreach (MeshChunkData chunk in chunkManager.GetPopulatedChunks())
            {
                if (chunk.Mesh != null) totalVertCount += chunk.Mesh.vertexCount;
            }

            if (totalVertCount < minInliers * 2) return;

            int stride = math.max(1, totalVertCount / maxSampleVertices);
            int estimatedSamples = (totalVertCount / stride) + 256;
            var samplePos = new float3[estimatedSamples];
            var sampleNrm = new float3[estimatedSamples];
            int sampleCount = 0;

            int globalIdx = 0;
            foreach (MeshChunkData chunk in chunkManager.GetPopulatedChunks())
            {
                Mesh mesh = chunk.Mesh;
                if (mesh == null || mesh.vertexCount == 0) continue;

                using var meshData = Mesh.AcquireReadOnlyMeshData(mesh);
                var md = meshData[0];
                if (md.vertexCount == 0) continue;

                using var positions = new NativeArray<UnityEngine.Vector3>(md.vertexCount, Allocator.TempJob);
                using var normals = new NativeArray<UnityEngine.Vector3>(md.vertexCount, Allocator.TempJob);
                md.GetVertices(positions);
                md.GetNormals(normals);

                UnityEngine.Vector3 chunkOrigin = chunk.GameObject.transform.position;
                for (int i = 0; i < positions.Length; i++)
                {
                    if (globalIdx % stride == 0 && sampleCount < estimatedSamples)
                    {
                        samplePos[sampleCount] = (float3)(positions[i] + chunkOrigin);
                        sampleNrm[sampleCount] = (float3)normals[i];
                        sampleCount++;
                    }
                    globalIdx++;
                }
            }

            if (sampleCount < minInliers * 2) return;

            _detectionRunning = true;
            LaunchBackgroundDetection(samplePos, sampleNrm, sampleCount);
        }

        private void LaunchBackgroundDetection(float3[] positions, float3[] normals, int count)
        {
            int mPlanes = maxPlanes;
            int mIter = ransacIterations;
            float mInlierDist = inlierDistance;
            int mMinInliers = minInliers;
            float mNormalAlign = normalAlignThreshold;
            float cosSnap = math.cos(math.radians(axisSnapAngle));
            uint seed = (uint)(Time.frameCount + 1);

            Task.Run(() =>
            {
                try
                {
                    var detected = RunRansac(positions, normals, count,
                        mPlanes, mIter, mInlierDist, mMinInliers, mNormalAlign, cosSnap, seed);
                    _pendingResults = detected;
                    _detectionReady = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"[PlaneDetector] Background RANSAC failed: {ex.Message}");
                    _detectionRunning = false;
                }
            });
        }

        private static List<PlaneCandidate> RunRansac(
            float3[] positions, float3[] normals, int count,
            int maxPlanes, int ransacIterations, float inlierDistance,
            int minInliers, float normalAlignThreshold, float cosAxisSnap, uint seed)
        {
            var detected = new List<PlaneCandidate>();
            var inlierFlags = new bool[count];
            var rng = new Unity.Mathematics.Random(seed);

            for (int planeIdx = 0; planeIdx < maxPlanes; planeIdx++)
            {
                int availableCount = 0;
                for (int i = 0; i < count; i++)
                    if (!inlierFlags[i]) availableCount++;

                if (availableCount < minInliers) break;

                float3 bestNormal = default;
                float bestDistance = 0f;
                int bestInlierCount = 0;

                for (int iter = 0; iter < ransacIterations; iter++)
                {
                    int idxA = SampleNonInlier(ref rng, inlierFlags, count);
                    int idxB = SampleNonInlier(ref rng, inlierFlags, count);
                    int idxC = SampleNonInlier(ref rng, inlierFlags, count);

                    if (idxA == idxB || idxB == idxC || idxA == idxC) continue;

                    float3 pA = positions[idxA];
                    float3 pB = positions[idxB];
                    float3 pC = positions[idxC];

                    float3 normal = math.normalize(math.cross(pB - pA, pC - pA));
                    if (math.any(math.isnan(normal))) continue;

                    if (normal.y < 0) normal = -normal;
                    else if (normal.y == 0 && normal.x < 0) normal = -normal;

                    normal = SnapToAxis(normal, cosAxisSnap);
                    float distance = math.dot(normal, pA);

                    int inlierCount = 0;
                    for (int i = 0; i < count; i++)
                    {
                        if (inlierFlags[i]) continue;
                        float d = math.abs(math.dot(positions[i], normal) - distance);
                        if (d > inlierDistance) continue;
                        float nDot = math.abs(math.dot(normals[i], normal));
                        if (nDot >= normalAlignThreshold)
                            inlierCount++;
                    }
                    if (inlierCount > bestInlierCount)
                    {
                        bestInlierCount = inlierCount;
                        bestNormal = normal;
                        bestDistance = distance;
                    }
                }

                if (bestInlierCount < minInliers) break;

                RefinePlaneStatic(positions, normals, inlierFlags, count,
                    ref bestNormal, ref bestDistance, inlierDistance, normalAlignThreshold, cosAxisSnap);

                for (int i = 0; i < count; i++)
                {
                    if (inlierFlags[i]) continue;
                    float d = math.abs(math.dot(positions[i], bestNormal) - bestDistance);
                    float nDot = math.abs(math.dot(normals[i], bestNormal));
                    if (d < inlierDistance && nDot >= normalAlignThreshold)
                        inlierFlags[i] = true;
                }

                detected.Add(new PlaneCandidate
                {
                    Normal = bestNormal,
                    Distance = bestDistance,
                    InlierCount = bestInlierCount
                });
            }

            return detected;
        }

        private static float3 SnapToAxis(float3 normal, float cosThreshold)
        {
            float3[] axes = { new(1, 0, 0), new(0, 1, 0), new(0, 0, 1) };
            for (int a = 0; a < 3; a++)
            {
                float dot = math.abs(math.dot(normal, axes[a]));
                if (dot >= cosThreshold)
                    return math.dot(normal, axes[a]) > 0 ? axes[a] : -axes[a];
            }
            return normal;
        }

        private static void RefinePlaneStatic(float3[] positions, float3[] normals,
            bool[] inlierFlags, int count,
            ref float3 normal, ref float distance,
            float inlierDistance, float normalAlignThreshold, float cosAxisSnap)
        {
            float3 centroid = float3.zero;
            int cnt = 0;
            for (int i = 0; i < count; i++)
            {
                if (inlierFlags[i]) continue;
                float d = math.abs(math.dot(positions[i], normal) - distance);
                float nDot = math.abs(math.dot(normals[i], normal));
                if (d < inlierDistance * 1.5f && nDot >= normalAlignThreshold * 0.95f)
                {
                    centroid += positions[i];
                    cnt++;
                }
            }
            if (cnt < 3) return;
            centroid /= cnt;

            float3x3 cov = float3x3.zero;
            for (int i = 0; i < count; i++)
            {
                if (inlierFlags[i]) continue;
                float d = math.abs(math.dot(positions[i], normal) - distance);
                float nDot = math.abs(math.dot(normals[i], normal));
                if (d < inlierDistance * 1.5f && nDot >= normalAlignThreshold * 0.95f)
                {
                    float3 diff = positions[i] - centroid;
                    cov.c0 += diff * diff.x;
                    cov.c1 += diff * diff.y;
                    cov.c2 += diff * diff.z;
                }
            }
            cov.c0 /= cnt;
            cov.c1 /= cnt;
            cov.c2 /= cnt;

            float3 refinedNormal = SmallestEigenvector(cov);
            if (math.dot(refinedNormal, normal) < 0)
                refinedNormal = -refinedNormal;

            refinedNormal = SnapToAxis(refinedNormal, cosAxisSnap);
            normal = refinedNormal;
            distance = math.dot(refinedNormal, centroid);
        }

        private static float3 SmallestEigenvector(float3x3 m)
        {
            float shift = -0.0001f;
            float3x3 shifted = m;
            shifted.c0.x += shift;
            shifted.c1.y += shift;
            shifted.c2.z += shift;

            float det = math.determinant(shifted);
            if (math.abs(det) < 1e-12f) return math.normalize(new float3(0, 1, 0));

            float3x3 inv = math.inverse(shifted);

            float3 v = math.normalize(new float3(1, 1, 1));
            for (int i = 0; i < 20; i++)
            {
                v = math.mul(inv, v);
                float len = math.length(v);
                if (len < 1e-10f) break;
                v /= len;
            }

            return math.normalize(v);
        }

        private void MergeWithPersistent(List<PlaneCandidate> detected)
        {
            bool[] matched = new bool[_persistentPlanes.Count];

            foreach (PlaneCandidate dp in detected)
            {
                int bestIdx = -1;
                float bestScore = float.MaxValue;

                for (int i = 0; i < _persistentPlanes.Count; i++)
                {
                    if (matched[i]) continue;
                    PlaneCandidate pp = _persistentPlanes[i];
                    float normalDot = math.abs(math.dot(dp.Normal, pp.Normal));
                    if (normalDot < 0.95f) continue;
                    float distDiff = math.abs(dp.Distance - pp.Distance);
                    if (distDiff < mergePlaneDistance && distDiff < bestScore)
                    {
                        bestScore = distDiff;
                        bestIdx = i;
                    }
                }

                if (bestIdx >= 0)
                {
                    matched[bestIdx] = true;
                    PlaneCandidate pp = _persistentPlanes[bestIdx];
                    float w = dp.InlierCount / (float)(dp.InlierCount + pp.InlierCount);
                    pp.Normal = math.normalize(math.lerp(pp.Normal, dp.Normal, w));
                    pp.Distance = math.lerp(pp.Distance, dp.Distance, w);
                    pp.InlierCount = math.max(pp.InlierCount, dp.InlierCount);
                    pp.Confidence = math.min(pp.Confidence + confidenceGrowth, 1f);
                    _persistentPlanes[bestIdx] = pp;
                }
                else
                {
                    _persistentPlanes.Add(new PlaneCandidate
                    {
                        Normal = dp.Normal,
                        Distance = dp.Distance,
                        InlierCount = dp.InlierCount,
                        Confidence = 0.3f
                    });
                }
            }

            for (int i = _persistentPlanes.Count - 1; i >= 0; i--)
            {
                if (matched[i]) continue;
                PlaneCandidate pp = _persistentPlanes[i];
                pp.Confidence -= confidenceDecay;
                if (pp.Confidence <= 0f)
                    _persistentPlanes.RemoveAt(i);
                else
                    _persistentPlanes[i] = pp;
            }
        }

        private void PublishPlanes()
        {
            if (_planes.IsCreated && _planes.Length < _persistentPlanes.Count)
            {
                _planes.Dispose();
                _planes = default;
            }

            if (!_planes.IsCreated && _persistentPlanes.Count > 0)
                _planes = new NativeArray<PlaneData>(
                    math.max(_persistentPlanes.Count, 16), Allocator.Persistent);

            _planeCount = _persistentPlanes.Count;
            for (int i = 0; i < _planeCount; i++)
            {
                PlaneCandidate pc = _persistentPlanes[i];
                _planes[i] = new PlaneData
                {
                    Normal = pc.Normal,
                    Distance = pc.Distance,
                    Confidence = pc.Confidence
                };
            }
        }

        private static int SampleNonInlier(ref Unity.Mathematics.Random rng, bool[] inlierFlags, int count)
        {
            for (int attempt = 0; attempt < 100; attempt++)
            {
                int idx = rng.NextInt(0, count);
                if (!inlierFlags[idx]) return idx;
            }
            return rng.NextInt(0, count);
        }

        private struct PlaneCandidate
        {
            public float3 Normal;
            public float Distance;
            public int InlierCount;
            public float Confidence;
        }
    }
}
