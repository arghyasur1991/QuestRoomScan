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
        private const int GpuVertexStride = GPUSurfaceNets.VertexStride;
        private const int VertPos = GPUSurfaceNets.VertexPosOffset;
        private const int VertNorm = GPUSurfaceNets.VertexNormalOffset;
        private const int VertPacked = GPUSurfaceNets.VertexPackedColorOffset;
        private const string PlyFileName = "points3d.ply";

        private static bool _exporting;

        /// <summary>
        /// Reads the GPU vertex buffer via async readback and writes a binary PLY to disk.
        /// <paramref name="outputDir"/> must be a valid directory path.
        /// </summary>
        public static async Task ExportAsync(string outputDir)
        {
            if (_exporting) return;
            if (string.IsNullOrEmpty(outputDir))
            {
                Logger.Warning("PointCloudExporter: no output directory specified");
                return;
            }
            _exporting = true;

            try
            {
                MeshExtractor.Instance?.ExtractForAuthoring();
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

                Directory.CreateDirectory(outputDir);
                string path = Path.Combine(outputDir, PlyFileName);

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

        /// <summary>Returns true if a PLY file exists in the given directory.</summary>
        public static bool ExistsIn(string dir) =>
            dir != null && File.Exists(Path.Combine(dir, PlyFileName));

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

                bw.Write(BitConverter.ToSingle(vertexData, off + VertPos));
                bw.Write(BitConverter.ToSingle(vertexData, off + VertPos + 4));
                bw.Write(BitConverter.ToSingle(vertexData, off + VertPos + 8));

                bw.Write(BitConverter.ToSingle(vertexData, off + VertNorm));
                bw.Write(BitConverter.ToSingle(vertexData, off + VertNorm + 4));
                bw.Write(BitConverter.ToSingle(vertexData, off + VertNorm + 8));

                uint packed = BitConverter.ToUInt32(vertexData, off + VertPacked);
                bw.Write((byte)(packed & 0xFF));
                bw.Write((byte)((packed >> 8) & 0xFF));
                bw.Write((byte)((packed >> 16) & 0xFF));
            }
        }
    }
}
