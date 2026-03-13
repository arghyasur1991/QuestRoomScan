using System;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Exports the current GPU mesh as a dense PLY point cloud for Gaussian Splat initialization.
    /// Reads vertices from the GPU Surface Nets vertex buffer via async readback.
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
            if (MeshExtractor.Instance == null) return;

            if (Time.time - _lastExportTime >= autoExportIntervalSeconds)
            {
                _lastExportTime = Time.time;
                _ = ExportAsync();
            }
        }

        // GPUVertex layout must match the compute shader: float3 pos, float3 norm, uint packedColor, uint voxelFlatIdx
        private const int GpuVertexStride = 32;

        public async Task ExportAsync()
        {
            if (_exporting) return;
            _exporting = true;

            try
            {
                var gpuSN = MeshExtractor.Instance?.GpuSurfaceNets;
                if (gpuSN == null || gpuSN.VertexBuffer == null)
                {
                    _exporting = false;
                    return;
                }

                var vertBuf = gpuSN.VertexBuffer;
                var req = await AsyncGPUReadback.RequestAsync(vertBuf);
                if (req.hasError)
                {
                    Debug.LogWarning("[RoomScan] PointCloudExporter: GPU readback error");
                    _exporting = false;
                    return;
                }

                var raw = req.GetData<byte>();
                int vertCount = raw.Length / GpuVertexStride;
                if (vertCount == 0)
                {
                    Debug.Log("[RoomScan] PointCloudExporter: no vertices to export");
                    _exporting = false;
                    return;
                }

                // Copy to managed array for background thread
                byte[] data = new byte[raw.Length];
                NativeArray<byte>.Copy(raw, data, raw.Length);
                string path = _plyPath;

                await Task.Run(() => WritePly(path, data, vertCount));

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

        private static void WritePly(string path, byte[] vertexData, int vertCount)
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

            bw.Write(System.Text.Encoding.ASCII.GetBytes(header));

            // Parse GPUVertex structs: float3 pos (12B), float3 norm (12B), uint packedColor (4B), uint voxelFlatIdx (4B)
            for (int i = 0; i < vertCount; i++)
            {
                int off = i * GpuVertexStride;

                // pos xyz
                bw.Write(BitConverter.ToSingle(vertexData, off));
                bw.Write(BitConverter.ToSingle(vertexData, off + 4));
                bw.Write(BitConverter.ToSingle(vertexData, off + 8));

                // norm xyz
                bw.Write(BitConverter.ToSingle(vertexData, off + 12));
                bw.Write(BitConverter.ToSingle(vertexData, off + 16));
                bw.Write(BitConverter.ToSingle(vertexData, off + 20));

                // packedColor → RGB
                uint packed = BitConverter.ToUInt32(vertexData, off + 24);
                bw.Write((byte)(packed & 0xFF));
                bw.Write((byte)((packed >> 8) & 0xFF));
                bw.Write((byte)((packed >> 16) & 0xFF));
            }
        }
    }
}
