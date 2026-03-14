using UnityEngine;

namespace Genesis.RoomScan.GSplat
{
    /// <summary>
    /// GPU 8-bit LSD radix sort for (uint key, uint payload) pairs.
    /// Adapted from GPUSorting by Thomas Smith (MIT) via UGS.
    /// Uses direct Dispatch calls (no CommandBuffer).
    /// 13 dispatches total: 1 init + 4 passes × (Upsweep + Scan + Downsweep).
    /// </summary>
    public class GpuRadixSort
    {
        const uint PartitionSize = 3840;
        const uint Radix = 256;
        const uint RadixPasses = 4;
        const uint RadixBits = 8;

        readonly ComputeShader _cs;
        readonly int _kernelInit;
        readonly int _kernelUpsweep;
        readonly int _kernelScan;
        readonly int _kernelDownsweep;
        readonly bool _valid;

        LocalKeyword _kwKeyUint, _kwPayloadUint, _kwAscend, _kwSortPairs, _kwVulkan;

        public bool Valid => _valid;

        public struct Resources
        {
            public GraphicsBuffer altBuffer;
            public GraphicsBuffer altPayloadBuffer;
            public GraphicsBuffer passHistBuffer;
            public GraphicsBuffer globalHistBuffer;

            public static Resources Create(int count)
            {
                uint threadBlocks = DivRoundUp((uint)count, PartitionSize);
                uint passHistSize = threadBlocks * Radix;
                uint globalHistSize = Radix * RadixPasses;

                return new Resources
                {
                    altBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "RadixAlt" },
                    altPayloadBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "RadixAltPayload" },
                    passHistBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)passHistSize, 4) { name = "RadixPassHist" },
                    globalHistBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)globalHistSize, 4) { name = "RadixGlobalHist" },
                };
            }

            public void Dispose()
            {
                altBuffer?.Dispose();
                altPayloadBuffer?.Dispose();
                passHistBuffer?.Dispose();
                globalHistBuffer?.Dispose();
                altBuffer = null;
                altPayloadBuffer = null;
                passHistBuffer = null;
                globalHistBuffer = null;
            }
        }

        static uint DivRoundUp(uint x, uint y) => (x + y - 1) / y;

        public GpuRadixSort(ComputeShader cs)
        {
            _cs = cs;
            if (cs == null) return;

            _kernelInit = cs.FindKernel("InitDeviceRadixSort");
            _kernelUpsweep = cs.FindKernel("Upsweep");
            _kernelScan = cs.FindKernel("Scan");
            _kernelDownsweep = cs.FindKernel("Downsweep");

            _valid = _kernelInit >= 0 && _kernelUpsweep >= 0 &&
                     _kernelScan >= 0 && _kernelDownsweep >= 0;

            if (_valid)
            {
                if (!cs.IsSupported(_kernelInit) || !cs.IsSupported(_kernelUpsweep) ||
                    !cs.IsSupported(_kernelScan) || !cs.IsSupported(_kernelDownsweep))
                {
                    _valid = false;
                    Debug.LogWarning("[GpuRadixSort] Wave intrinsics not supported — falling back to bitonic sort");
                    return;
                }
            }

            _kwKeyUint = new LocalKeyword(cs, "KEY_UINT");
            _kwPayloadUint = new LocalKeyword(cs, "PAYLOAD_UINT");
            _kwAscend = new LocalKeyword(cs, "SHOULD_ASCEND");
            _kwSortPairs = new LocalKeyword(cs, "SORT_PAIRS");
            _kwVulkan = new LocalKeyword(cs, "VULKAN");

            cs.EnableKeyword(_kwKeyUint);
            cs.EnableKeyword(_kwPayloadUint);
            cs.EnableKeyword(_kwAscend);
            cs.EnableKeyword(_kwSortPairs);

            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Vulkan)
                cs.EnableKeyword(_kwVulkan);
            else
                cs.DisableKeyword(_kwVulkan);
        }

        static readonly int ID_numKeys = Shader.PropertyToID("e_numKeys");
        static readonly int ID_radixShift = Shader.PropertyToID("e_radixShift");
        static readonly int ID_threadBlocks = Shader.PropertyToID("e_threadBlocks");
        static readonly int ID_sort = Shader.PropertyToID("b_sort");
        static readonly int ID_alt = Shader.PropertyToID("b_alt");
        static readonly int ID_sortPayload = Shader.PropertyToID("b_sortPayload");
        static readonly int ID_altPayload = Shader.PropertyToID("b_altPayload");
        static readonly int ID_passHist = Shader.PropertyToID("b_passHist");
        static readonly int ID_globalHist = Shader.PropertyToID("b_globalHist");

        /// <summary>
        /// Sort key-payload pairs in ascending order.
        /// keys and payloads buffers are sorted in-place (result ends up in them after 4 even passes).
        /// </summary>
        public void Dispatch(int count, GraphicsBuffer keys, GraphicsBuffer payloads, Resources res)
        {
            if (!_valid || count <= 0) return;

            uint threadBlocks = DivRoundUp((uint)count, PartitionSize);

            _cs.SetInt(ID_numKeys, count);
            _cs.SetInt(ID_threadBlocks, (int)threadBlocks);

            // Static buffer bindings
            _cs.SetBuffer(_kernelUpsweep, ID_passHist, res.passHistBuffer);
            _cs.SetBuffer(_kernelUpsweep, ID_globalHist, res.globalHistBuffer);
            _cs.SetBuffer(_kernelScan, ID_passHist, res.passHistBuffer);
            _cs.SetBuffer(_kernelDownsweep, ID_passHist, res.passHistBuffer);
            _cs.SetBuffer(_kernelDownsweep, ID_globalHist, res.globalHistBuffer);

            // Clear global histogram
            _cs.SetBuffer(_kernelInit, ID_globalHist, res.globalHistBuffer);
            _cs.Dispatch(_kernelInit, 1, 1, 1);

            var srcKey = keys;
            var srcPayload = payloads;
            var dstKey = res.altBuffer;
            var dstPayload = res.altPayloadBuffer;

            for (uint shift = 0; shift < 32; shift += RadixBits)
            {
                _cs.SetInt(ID_radixShift, (int)shift);

                // Upsweep
                _cs.SetBuffer(_kernelUpsweep, ID_sort, srcKey);
                _cs.Dispatch(_kernelUpsweep, (int)threadBlocks, 1, 1);

                // Scan
                _cs.Dispatch(_kernelScan, (int)Radix, 1, 1);

                // Downsweep
                _cs.SetBuffer(_kernelDownsweep, ID_sort, srcKey);
                _cs.SetBuffer(_kernelDownsweep, ID_sortPayload, srcPayload);
                _cs.SetBuffer(_kernelDownsweep, ID_alt, dstKey);
                _cs.SetBuffer(_kernelDownsweep, ID_altPayload, dstPayload);
                _cs.Dispatch(_kernelDownsweep, (int)threadBlocks, 1, 1);

                // Swap src/dst
                (srcKey, dstKey) = (dstKey, srcKey);
                (srcPayload, dstPayload) = (dstPayload, srcPayload);
            }
        }
    }
}
