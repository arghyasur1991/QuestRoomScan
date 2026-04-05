#if HAS_ONNXRUNTIME
namespace Genesis.RoomScan.ObjectReconstruction
{
    internal enum ExecutionProvider
    {
        CPU,
        NNAPI,
        XNNPACK,
        CoreML,
        QNN_HTP
    }
}
#endif
