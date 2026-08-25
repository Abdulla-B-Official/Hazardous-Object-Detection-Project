from ultralytics import YOLO

# --------------------------------------------------
# 1. Load trained YOLO model
# --------------------------------------------------
MODEL_PATH = "runs/hazardous_detection/weights/best.pt"

model = YOLO(MODEL_PATH)

print("=" * 60)
print("HAZARDOUS DETECTION - MODEL EVALUATION")
print("=" * 60)

# --------------------------------------------------
# 2. Evaluate on TEST dataset
# --------------------------------------------------
metrics = model.val(
    data="data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device="cpu",
    project="runs/evaluation",
    name="test_evaluation"
)

# --------------------------------------------------
# 3. Extract YOLO metrics
# --------------------------------------------------
precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map50_95 = metrics.box.map

# F1 Score
if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0.0

# --------------------------------------------------
# 4. Detection Accuracy
# --------------------------------------------------
# Accuracy-like metric for object detection:
# Correct detections / Total actual objects
#
# Using precision and recall:
# TP = correct detections
# FN = missed detections
#
# This provides an accuracy-style reference,
# but it is NOT classification accuracy.

detection_accuracy = (precision * recall) / (
    precision + recall - (precision * recall)
) if (precision + recall - (precision * recall)) > 0 else 0.0

# --------------------------------------------------
# 5. Display results
# --------------------------------------------------
print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy*     : {detection_accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"mAP@50        : {map50:.4f}")
print(f"mAP@50-95     : {map50_95:.4f}")

print("=" * 60)

# --------------------------------------------------
# 6. Percentage values
# --------------------------------------------------
print("\nMetrics in Percentage:")
print(f"Accuracy*     : {detection_accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")
print(f"mAP@50        : {map50 * 100:.2f}%")
print(f"mAP@50-95     : {map50_95 * 100:.2f}%")

print("\n* Accuracy is an accuracy-style reference for object detection,")
print("  not standard classification accuracy.")

print("\nEvaluation completed successfully!")
print("Results saved in:")
print("runs/evaluation/test_evaluation/")