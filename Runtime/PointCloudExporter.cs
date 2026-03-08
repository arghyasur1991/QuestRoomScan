using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Exports the current TSDF mesh as a dense PLY point cloud for Gaussian Splat initialization.
    /// Collects all world-space vertices with colors and normals from populated mesh chunks.
    /// </summary>
    public class PointCloudExporter : MonoBehaviour
    {
        [SerializeField, Tooltip("Seconds between automatic PLY exports (0 = manual only)")]
        private float autoExportIntervalSeconds = 30f;

        private string _exportDir;
        private string _plyPath;
        private float _lastExportTime;
        private bool _exporting;

        public string ExportPath => _plyPath;

        private void Start()
        {
            _exportDir = Path.Combine(Application.persistentDataPath, "GSExport");
            _plyPath = Path.Combine(_exportDir, "points3d.ply");
            Directory.CreateDirectory(_exportDir);
            _lastExportTime = Time.time;
        }

        private void Update()
        {
            if (autoExportIntervalSeconds <= 0 || _exporting) return;
            if (ChunkManager.Instance == null) return;

            if (Time.time - _lastExportTime >= autoExportIntervalSeconds)
            {
                _lastExportTime = Time.time;
                _ = ExportAsync();
            }
        }

        public async Task ExportAsync()
        {
            if (_exporting) return;
            _exporting = true;

            try
            {
                var chunks = ChunkManager.Instance?.GetPopulatedChunks();
                if (chunks == null) { _exporting = false; return; }

                int totalVerts = 0;
                using var vertexStream = new MemoryStream();
                using var writer = new BinaryWriter(vertexStream);

                foreach (var chunk in chunks)
                {
                    if (chunk.Mesh == null || chunk.GameObject == null) continue;
                    var mesh = chunk.Mesh;
                    var transform = chunk.GameObject.transform;

                    Vector3[] verts = mesh.vertices;
                    Vector3[] normals = mesh.normals;
                    Color32[] colors = mesh.colors32;

                    bool hasNormals = normals != null && normals.Length == verts.Length;
                    bool hasColors = colors != null && colors.Length == verts.Length;

                    for (int i = 0; i < verts.Length; i++)
                    {
                        Vector3 worldPos = transform.TransformPoint(verts[i]);
                        writer.Write(worldPos.x);
                        writer.Write(worldPos.y);
                        writer.Write(worldPos.z);

                        if (hasNormals)
                        {
                            Vector3 worldNormal = transform.TransformDirection(normals[i]).normalized;
                            writer.Write(worldNormal.x);
                            writer.Write(worldNormal.y);
                            writer.Write(worldNormal.z);
                        }
                        else
                        {
                            writer.Write(0f); writer.Write(1f); writer.Write(0f);
                        }

                        if (hasColors)
                        {
                            writer.Write(colors[i].r);
                            writer.Write(colors[i].g);
                            writer.Write(colors[i].b);
                        }
                        else
                        {
                            writer.Write((byte)128);
                            writer.Write((byte)128);
                            writer.Write((byte)128);
                        }

                        totalVerts++;
                    }
                }

                if (totalVerts == 0)
                {
                    Debug.Log("[RoomScan] PointCloudExporter: no vertices to export");
                    _exporting = false;
                    return;
                }

                byte[] vertexData = vertexStream.ToArray();
                int vertCount = totalVerts;
                string path = _plyPath;

                await Task.Run(() =>
                {
                    using var fs = new FileStream(path, FileMode.Create);
                    using var bw = new BinaryWriter(fs);

                    string header =
                        "ply\n" +
                        "format binary_little_endian 1.0\n" +
                        $"element vertex {vertCount}\n" +
                        "property float x\n" +
                        "property float y\n" +
                        "property float z\n" +
                        "property float nx\n" +
                        "property float ny\n" +
                        "property float nz\n" +
                        "property uchar red\n" +
                        "property uchar green\n" +
                        "property uchar blue\n" +
                        "end_header\n";

                    byte[] headerBytes = System.Text.Encoding.ASCII.GetBytes(header);
                    bw.Write(headerBytes);
                    bw.Write(vertexData);
                });

                Debug.Log($"[RoomScan] PointCloudExporter: saved {vertCount} vertices to {path} ({new FileInfo(path).Length / 1024}KB)");
            }
            catch (Exception e)
            {
                Debug.LogError($"[RoomScan] PointCloudExporter: export error: {e.Message}");
            }
            finally
            {
                _exporting = false;
            }
        }
    }
}
