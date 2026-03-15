using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// Loads server-trained Gaussian splats from PLY data and uploads them
    /// to GPU buffers for rendering via <see cref="GSSectorRenderer"/>.
    /// </summary>
    public class GSplatManager : MonoBehaviour
    {
        [SerializeField] GSSectorRenderer splatRenderer;

        GSplatBuffers _serverTrainedBuffers;

        public bool HasServerTrainedSplats => _serverTrainedBuffers != null && _serverTrainedBuffers.CurrentCount > 0;

        /// <summary>
        /// Parses a 3DGS-format PLY file, converts from COLMAP to Unity coordinates,
        /// and uploads to GPU buffers for rendering.
        /// </summary>
        public void LoadTrainedPly(byte[] plyData)
        {
            ParsePlyGaussians(plyData,
                out float[] means, out float[] scales, out float[] quats,
                out float[] opacities, out float[] featuresDC, out float[] featuresRest,
                out int count, out int shDegree);

            if (count == 0)
            {
                Debug.LogWarning("[GSplatManager] PLY contained 0 Gaussians");
                return;
            }

            ConvertColmapToUnity(means, quats, count);

            _serverTrainedBuffers?.Dispose();
            _serverTrainedBuffers = new GSplatBuffers(count, shDegree);
            _serverTrainedBuffers.CurrentCount = count;

            _serverTrainedBuffers.Means.SetData(means);
            _serverTrainedBuffers.Scales.SetData(scales);
            _serverTrainedBuffers.Quats.SetData(quats);
            _serverTrainedBuffers.Opacities.SetData(opacities);
            _serverTrainedBuffers.FeaturesDC.SetData(featuresDC);
            if (featuresRest.Length > 0)
                _serverTrainedBuffers.FeaturesRest.SetData(featuresRest);

            Debug.Log($"[GSplatManager] Loaded {count} trained Gaussians (SH degree {shDegree})");

            if (splatRenderer != null)
            {
                splatRenderer.Initialize();
                splatRenderer.SetServerTrainedBuffers(_serverTrainedBuffers);
            }
        }

        static void ParsePlyGaussians(byte[] data,
            out float[] means, out float[] scales, out float[] quats,
            out float[] opacities, out float[] featuresDC, out float[] featuresRest,
            out int count, out int shDegree)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            int vertexCount = 0;
            var properties = new List<string>();
            while (true)
            {
                string line = ReadAsciiLine(reader);
                if (line == null) break;
                if (line.StartsWith("element vertex"))
                    vertexCount = int.Parse(line.Split(' ')[2]);
                else if (line.StartsWith("property float") || line.StartsWith("property double"))
                    properties.Add(line.Split(' ')[2]);
                else if (line == "end_header")
                    break;
            }

            count = vertexCount;
            int propCount = properties.Count;

            // 3(xyz) + 3(nxnynz) + 3(f_dc) + N(f_rest) + 1(opacity) + 3(scale) + 4(rot) = 17 + N
            int shRestCount = propCount - 17;
            if (shRestCount < 0) shRestCount = 0;
            shDegree = shRestCount switch
            {
                0 => 0,
                >= 45 => 3,
                >= 24 => 2,
                >= 9 => 1,
                _ => 0
            };
            int actualRestPerGaussian = shDegree switch
            {
                0 => 0, 1 => 9, 2 => 24, 3 => 45, _ => 0
            };

            means = new float[count * 3];
            scales = new float[count * 3];
            quats = new float[count * 4];
            opacities = new float[count];
            featuresDC = new float[count * 3];
            featuresRest = new float[count * actualRestPerGaussian];

            int idxX = properties.IndexOf("x");
            int idxY = properties.IndexOf("y");
            int idxZ = properties.IndexOf("z");
            int idxDC0 = properties.IndexOf("f_dc_0");
            int idxOpac = properties.IndexOf("opacity");
            int idxScale0 = properties.IndexOf("scale_0");
            int idxRot0 = properties.IndexOf("rot_0");
            int idxRest0 = properties.IndexOf("f_rest_0");

            var propValues = new float[propCount];

            for (int i = 0; i < count; i++)
            {
                for (int p = 0; p < propCount; p++)
                    propValues[p] = reader.ReadSingle();

                means[i * 3 + 0] = propValues[idxX];
                means[i * 3 + 1] = propValues[idxY];
                means[i * 3 + 2] = propValues[idxZ];

                if (idxDC0 >= 0)
                {
                    featuresDC[i * 3 + 0] = propValues[idxDC0];
                    featuresDC[i * 3 + 1] = propValues[idxDC0 + 1];
                    featuresDC[i * 3 + 2] = propValues[idxDC0 + 2];
                }

                if (idxRest0 >= 0 && actualRestPerGaussian > 0)
                {
                    for (int r = 0; r < actualRestPerGaussian; r++)
                        featuresRest[i * actualRestPerGaussian + r] = propValues[idxRest0 + r];
                }

                if (idxOpac >= 0)
                    opacities[i] = propValues[idxOpac];

                if (idxScale0 >= 0)
                {
                    scales[i * 3 + 0] = propValues[idxScale0];
                    scales[i * 3 + 1] = propValues[idxScale0 + 1];
                    scales[i * 3 + 2] = propValues[idxScale0 + 2];
                }

                if (idxRot0 >= 0)
                {
                    quats[i * 4 + 0] = propValues[idxRot0];     // w
                    quats[i * 4 + 1] = propValues[idxRot0 + 1]; // x
                    quats[i * 4 + 2] = propValues[idxRot0 + 2]; // y
                    quats[i * 4 + 3] = propValues[idxRot0 + 3]; // z
                }
            }
        }

        /// <summary>
        /// Converts positions and quaternions from COLMAP (right-handed Y-down)
        /// to Unity (left-handed Y-up). Reverses unity_to_colmap_pose:
        /// negate Y for positions, negate Y and flip handedness for quaternions.
        /// </summary>
        static void ConvertColmapToUnity(float[] means, float[] quats, int count)
        {
            for (int i = 0; i < count; i++)
            {
                means[i * 3 + 1] = -means[i * 3 + 1];

                // COLMAP q = (w, x, y, z) -> Unity buffer layout (x, -y, z, w)
                float qw = quats[i * 4 + 0];
                float qx = quats[i * 4 + 1];
                float qy = quats[i * 4 + 2];
                float qz = quats[i * 4 + 3];
                quats[i * 4 + 0] = qx;
                quats[i * 4 + 1] = -qy;
                quats[i * 4 + 2] = qz;
                quats[i * 4 + 3] = qw;
            }
        }

        static string ReadAsciiLine(BinaryReader reader)
        {
            var sb = new StringBuilder();
            try
            {
                while (true)
                {
                    byte b = reader.ReadByte();
                    if (b == '\n') break;
                    if (b != '\r') sb.Append((char)b);
                }
            }
            catch (EndOfStreamException) { return null; }
            return sb.ToString();
        }

        void OnDestroy()
        {
            _serverTrainedBuffers?.Dispose();
        }
    }
}
