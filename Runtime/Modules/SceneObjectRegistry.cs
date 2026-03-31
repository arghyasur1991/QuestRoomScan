using System;
using System.Collections.Generic;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Unified inventory of all detected objects in the scanned room.
    /// Fed by MRUK anchors (structural) and AI detection (granular).
    /// Persisted as JSON in the scan package.
    /// </summary>
    public class SceneObjectRegistry
    {
        private readonly List<SceneObject> _objects = new();
        private readonly Dictionary<string, SceneObject> _byId = new();

        public IReadOnlyList<SceneObject> AllObjects => _objects;
        public int Count => _objects.Count;

        public int MrukCount { get; private set; }
        public int AiCount { get; private set; }

        public event Action<SceneObject> ObjectAdded;

        public void Add(SceneObject obj)
        {
            if (obj == null || string.IsNullOrEmpty(obj.id)) return;
            if (_byId.ContainsKey(obj.id))
            {
                Update(obj);
                return;
            }

            _objects.Add(obj);
            _byId[obj.id] = obj;
            UpdateCounts(obj, 1);
            ObjectAdded?.Invoke(obj);
        }

        public void Update(SceneObject obj)
        {
            if (obj == null || !_byId.TryGetValue(obj.id, out var existing)) return;

            int idx = _objects.IndexOf(existing);
            if (idx >= 0)
            {
                UpdateCounts(existing, -1);
                _objects[idx] = obj;
                _byId[obj.id] = obj;
                UpdateCounts(obj, 1);
            }
        }

        public bool TryGet(string id, out SceneObject obj) => _byId.TryGetValue(id, out obj);

        /// <summary>Find all objects matching a label (case-insensitive contains).</summary>
        public List<SceneObject> FindByLabel(string label)
        {
            var results = new List<SceneObject>();
            if (string.IsNullOrEmpty(label)) return results;
            var lower = label.ToLowerInvariant();
            foreach (var obj in _objects)
            {
                if (obj.label != null && obj.label.ToLowerInvariant().Contains(lower))
                    results.Add(obj);
            }
            return results;
        }

        /// <summary>Find all objects from a specific source.</summary>
        public List<SceneObject> FindBySource(SceneObjectSource source)
        {
            var results = new List<SceneObject>();
            foreach (var obj in _objects)
            {
                if (obj.source == source)
                    results.Add(obj);
            }
            return results;
        }

        /// <summary>Find all objects of a specific surface type.</summary>
        public List<SceneObject> FindBySurface(SurfaceType surfaceType)
        {
            var results = new List<SceneObject>();
            foreach (var obj in _objects)
            {
                if (obj.surfaceType == surfaceType)
                    results.Add(obj);
            }
            return results;
        }

        /// <summary>Find the closest object to a world position.</summary>
        public SceneObject FindClosest(Vector3 worldPos, float maxDistance = float.MaxValue)
        {
            SceneObject best = null;
            float bestDist = maxDistance * maxDistance;
            foreach (var obj in _objects)
            {
                float d = (obj.position - worldPos).sqrMagnitude;
                if (d < bestDist)
                {
                    bestDist = d;
                    best = obj;
                }
            }
            return best;
        }

        /// <summary>Find all objects whose bounds intersect a sphere.</summary>
        public List<SceneObject> FindInRadius(Vector3 center, float radius)
        {
            var results = new List<SceneObject>();
            float r2 = radius * radius;
            foreach (var obj in _objects)
            {
                if ((obj.position - center).sqrMagnitude <= r2)
                    results.Add(obj);
            }
            return results;
        }

        /// <summary>
        /// Associate detected objects with mesh triangles. For each vertex,
        /// determine which SceneObject's bounding volume it falls within.
        /// Returns per-vertex object index (-1 = no match).
        /// </summary>
        public int[] AssociateMeshVertices(Mesh mesh)
        {
            if (mesh == null) return null;
            var verts = mesh.vertices;
            var result = new int[verts.Length];
            for (int v = 0; v < verts.Length; v++)
            {
                result[v] = -1;
                float bestDist = float.MaxValue;
                for (int o = 0; o < _objects.Count; o++)
                {
                    var bounds = _objects[o].WorldBounds;
                    if (!bounds.Contains(verts[v])) continue;
                    float d = (verts[v] - _objects[o].position).sqrMagnitude;
                    if (d < bestDist)
                    {
                        bestDist = d;
                        result[v] = o;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Applies a relocation matrix to all objects from a given source.
        /// Used to transform saved AI detections from the original session's
        /// coordinate frame into the current tracking space.
        /// </summary>
        public void Relocate(Matrix4x4 relocation, SceneObjectSource source)
        {
            if (relocation == Matrix4x4.identity) return;
            foreach (var obj in _objects)
            {
                if (obj.source != source) continue;
                obj.position = relocation.MultiplyPoint3x4(obj.position);
                obj.rotation = relocation.rotation * obj.rotation;
            }
        }

        /// <summary>Remove all objects from a specific source (e.g. stale MRUK data on reload).</summary>
        public void RemoveBySource(SceneObjectSource source)
        {
            for (int i = _objects.Count - 1; i >= 0; i--)
            {
                if (_objects[i].source == source)
                {
                    UpdateCounts(_objects[i], -1);
                    _byId.Remove(_objects[i].id);
                    _objects.RemoveAt(i);
                }
            }
        }

        public void Clear()
        {
            _objects.Clear();
            _byId.Clear();
            MrukCount = 0;
            AiCount = 0;
        }

        private void UpdateCounts(SceneObject obj, int delta)
        {
            switch (obj.source)
            {
                case SceneObjectSource.MRUK: MrukCount += delta; break;
                case SceneObjectSource.AIDetection: AiCount += delta; break;
            }
        }

        // ── Serialization ──────────────────────────────────────────

        [Serializable]
        private class SerializedRegistry
        {
            public List<SceneObject> objects = new();
        }

        public string ToJson()
        {
            var data = new SerializedRegistry { objects = new List<SceneObject>(_objects) };
            return JsonUtility.ToJson(data, true);
        }

        public static SceneObjectRegistry FromJson(string json)
        {
            var registry = new SceneObjectRegistry();
            if (string.IsNullOrEmpty(json)) return registry;
            try
            {
                var data = JsonUtility.FromJson<SerializedRegistry>(json);
                if (data?.objects != null)
                {
                    foreach (var obj in data.objects)
                        registry.Add(obj);
                }
            }
            catch (Exception e)
            {
                Logger.Warning($"SceneObjectRegistry: failed to deserialize: {e.Message}");
            }
            return registry;
        }
    }
}
