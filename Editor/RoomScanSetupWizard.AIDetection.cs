#if HAS_AI_INFERENCE
using Genesis.RoomScan.AIDetection;
using UnityEditor;
using UnityEngine;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        const string AI_PKG_DATA = "Packages/com.genesis.roomscan/Runtime.AIDetection/Data/";
        const string AI_PKG_SHADERS = "Packages/com.genesis.roomscan/Runtime.AIDetection/Shaders/";

        ObjectDetectionModule _objectDetection;
        bool _aiModelAssigned;
        bool _aiLabelsAssigned;
        bool _aiNmsShaderAssigned;
        bool _aiDepthProjectionAssigned;

        // -----------------------------------------------------------------
        //  Partial method implementations
        // -----------------------------------------------------------------

        partial void RefreshAIDetection()
        {
            _objectDetection = FindAny<ObjectDetectionModule>();
            if (_objectDetection != null)
            {
                _aiModelAssigned = AreFieldsAssigned(_objectDetection, "modelAsset");
                _aiLabelsAssigned = AreFieldsAssigned(_objectDetection, "classLabels");
                _aiNmsShaderAssigned = AreFieldsAssigned(_objectDetection, "nmsComputeShader");
                _aiDepthProjectionAssigned = AreFieldsAssigned(_objectDetection, "depthProjectionShader");
            }
            else
            {
                _aiModelAssigned = false;
                _aiLabelsAssigned = false;
                _aiNmsShaderAssigned = false;
                _aiDepthProjectionAssigned = false;
            }
        }

        partial void DrawAIDetectionOptionalStatus()
        {
            StatusRowOptional("ObjectDetectionModule (AI YOLO)", _objectDetection != null);
            if (_objectDetection != null)
            {
                StatusRow("  YOLO model asset (.onnx)", _aiModelAssigned);
                StatusRow("  Class labels (.txt)", _aiLabelsAssigned);
                StatusRow("  NMS compute shader", _aiNmsShaderAssigned);
                StatusRow("  Depth projection shader", _aiDepthProjectionAssigned);
            }
        }

        partial void CheckAIDetectionAnyMissing(ref bool anyMissing)
        {
            // Don't flag AI detection as required — it's fully optional
        }

        partial void DrawAIDetectionShaderStatus(ref bool needsFix)
        {
            if (_objectDetection != null && !_aiLabelsAssigned)
            {
                StatusRow("AI Detection class labels", false);
                needsFix = true;
            }
            if (_objectDetection != null && !_aiNmsShaderAssigned)
            {
                StatusRow("AI Detection NMS compute shader", false);
                needsFix = true;
            }
            if (_objectDetection != null && !_aiDepthProjectionAssigned)
            {
                StatusRow("AI Detection depth projection shader", false);
                needsFix = true;
            }
        }

        partial void WireAIDetectionComponents()
        {
            if (_objectDetection != null)
                WireComponent(_objectDetection);
        }

        partial void SetupAIDetectionIfAvailable(GameObject root)
        {
            SetupAIDetectionModule(root);
        }

        // -----------------------------------------------------------------
        //  AI Detection-specific methods
        // -----------------------------------------------------------------

        static void WireAIDetectionComponent(Component component)
        {
            if (component is ObjectDetectionModule odm)
            {
                var so = new SerializedObject(odm);

                var labelsProp = so.FindProperty("classLabels");
                if (labelsProp != null && labelsProp.objectReferenceValue == null)
                {
                    var labels = AssetDatabase.LoadAssetAtPath<TextAsset>(
                        AI_PKG_DATA + "coco_classes.txt");
                    if (labels != null)
                        labelsProp.objectReferenceValue = labels;
                }

                AssignNmsShader(so);
                AssignDepthProjectionShader(so);

                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(odm);
            }
        }

        static void AssignNmsShader(SerializedObject so)
        {
            var nmsProp = so.FindProperty("nmsComputeShader");
            if (nmsProp != null && nmsProp.objectReferenceValue == null)
            {
                var nms = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                    AI_PKG_SHADERS + "NMSCompute.compute");
                if (nms != null)
                    nmsProp.objectReferenceValue = nms;
            }
        }

        static void AssignDepthProjectionShader(SerializedObject so)
        {
            var prop = so.FindProperty("depthProjectionShader");
            if (prop != null && prop.objectReferenceValue == null)
            {
                var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(
                    AI_PKG_SHADERS + "DepthProjection.compute");
                if (shader != null)
                    prop.objectReferenceValue = shader;
            }
        }

        internal static void SetupAIDetectionModule(GameObject root)
        {
            if (root.GetComponent<ObjectDetectionModule>() == null)
                Undo.AddComponent<ObjectDetectionModule>(root);

            var module = root.GetComponent<ObjectDetectionModule>();
            if (module == null) return;

            var so = new SerializedObject(module);

            // Auto-assign class labels from package if not set
            var labelsProp = so.FindProperty("classLabels");
            if (labelsProp != null && labelsProp.objectReferenceValue == null)
            {
                var labels = AssetDatabase.LoadAssetAtPath<TextAsset>(
                    AI_PKG_DATA + "coco_classes.txt");
                if (labels != null)
                    labelsProp.objectReferenceValue = labels;
            }

            AssignNmsShader(so);
            AssignDepthProjectionShader(so);

            // Try to find a model asset in the project (user must have imported one)
            var modelProp = so.FindProperty("modelAsset");
            if (modelProp != null && modelProp.objectReferenceValue == null)
            {
                // Search for any .onnx model in common locations
                string[] guids = AssetDatabase.FindAssets("t:ModelAsset yolov9t");
                if (guids.Length == 0)
                    guids = AssetDatabase.FindAssets("t:ModelAsset yolo");
                if (guids.Length > 0)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                    var model = AssetDatabase.LoadAssetAtPath<Object>(path);
                    if (model != null)
                    {
                        modelProp.objectReferenceValue = model;
                        Debug.Log($"[RoomScan Setup] Auto-assigned AI model: {path}");
                    }
                }
                else
                {
                    Debug.LogWarning("[RoomScan Setup] No YOLO .onnx model found. " +
                        "Download yolov9t.onnx from https://huggingface.co/unity/inference-engine-yolo " +
                        "and place it in Assets/Game/AIModels/");
                }
            }

            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(module);
        }
    }
}
#endif
