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
    internal static class PointCloudExporter
    {
        // GPUVertex layout must match the compute shader: float3 pos, float3 norm, uint packedColor, uint voxelFlatIdx
        private const int GpuVertexStride = 32;

        private static bool _exporting;

        /// <summary>Absolute path of the exported PLY file on device.</summary>
        public static string ExportPath =>
            Path.Combine(Application.persistentDataPath, "GSExport", "points3d.ply");

        /// <summary>
        /// Reads the GPU vertex buffer via async readback and writes a binary PLY point cloud to disk.
        /// </summary>
        public static async Task ExportAsync()
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

                int activeVertCount = 0;
                if (gpuSN.CountersBuffer != null)
                {
                    var counterReq = await AsyncGPUReadback.RequestAsync(gpuSN.CountersBuffer);
                    if (!counterReq.hasError)
                    {
                        var counterData = counterReq.GetData<int>();
                        if (counterData.Length > 0)
                            activeVertCount = counterData[0];
                    }
                }

                var vertBuf = gpuSN.VertexBuffer;
                var req = await AsyncGPUReadback.RequestAsync(vertBuf);
                if (req.hasError)
                {
                    Logger.Warning("PointCloudExporter: GPU readback error");
                    _exporting = false;
                    return;
                }

                var raw = req.GetData<byte>();
                int bufferCapacity = raw.Length / GpuVertexStride;

                int vertCount = (activeVertCount > 0 && activeVertCount <= bufferCapacity)
                    ? activeVertCount
                    : bufferCapacity;

                if (vertCount == 0)
                {
                    Logger.Info("PointCloudExporter: no vertices to export");
                    _exporting = false;
                    return;
                }

                byte[] data = new byte[raw.Length];
                NativeArray<byte>.Copy(raw, data, raw.Length);
                string path = ExportPath;
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);

                await Task.Run(() => WritePly(path, data, vertCount));

                Logger.Info($"PointCloudExporter: saved {vertCount} vertices " +
                          $"(buffer capacity: {bufferCapacity}) to {path} ({new FileInfo(path).Length / 1024}KB)");
            }
            catch (Exception e)
            {
                Logger.Error($"PointCloudExporter: export error: {e.Message}");
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

            for (int i = 0; i < vertCount; i++)
            {
                int off = i * GpuVertexStride;

                bw.Write(BitConverter.ToSingle(vertexData, off));
                bw.Write(BitConverter.ToSingle(vertexData, off + 4));
                bw.Write(BitConverter.ToSingle(vertexData, off + 8));

                bw.Write(BitConverter.ToSingle(vertexData, off + 12));
                bw.Write(BitConverter.ToSingle(vertexData, off + 16));
                bw.Write(BitConverter.ToSingle(vertexData, off + 20));

                uint packed = BitConverter.ToUInt32(vertexData, off + 24);
                bw.Write((byte)(packed & 0xFF));
                bw.Write((byte)((packed >> 8) & 0xFF));
                bw.Write((byte)((packed >> 16) & 0xFF));
            }
        }
    }
}
