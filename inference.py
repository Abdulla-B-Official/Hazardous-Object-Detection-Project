from ultralytics import YOLO
import cv2
from pathlib import Path

# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------
MODEL_PATH = "runs/hazardous_detection/weights/best.pt"

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# 2. Input image
# --------------------------------------------------
IMAGE_PATH = IMAGE_PATH = r"test/images/Screenshot-2025-02-16-222257_png_png_jpg.rf.320b52def54def6e91ee6024bb13fa77.jpg"

# Change the above path to your actual test image.
# Example:
# IMAGE_PATH = "test/my_image.jpg"

# --------------------------------------------------
# 3. Run YOLO inference
# --------------------------------------------------
results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=False
)

# --------------------------------------------------
# 4. Process detection results
# --------------------------------------------------
for result in results:

    # Original image
    image = result.orig_img.copy()

    # Number of detected objects
    detection_count = len(result.boxes)

    print("\n" + "=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)

    print(f"Objects detected: {detection_count}")

    # --------------------------------------------------
    # 5. Display each detection
    # --------------------------------------------------
    for i, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print(
            f"Detection {i + 1}: "
            f"{class_name} | "
            f"Confidence: {confidence:.2f}"
        )

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------
    # 6. Save output image
    # --------------------------------------------------
    output_path = Path("runs/inference_result.jpg")

    cv2.imwrite(str(output_path), image)

    print("\nOutput saved to:")
    print(output_path)

    print("=" * 60)