from ultralytics import YOLO
import cv2

# --------------------------------------------------
# 1. Load trained YOLO model
# --------------------------------------------------
MODEL_PATH = "runs/hazardous_detection/weights/best.pt"

model = YOLO(MODEL_PATH)

print("=" * 60)
print("HAZARDOUS DETECTION - WEBCAM")
print("=" * 60)
print("Starting webcam...")
print("Press 'Q' to quit.")

# --------------------------------------------------
# 2. Open webcam
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# --------------------------------------------------
# 3. Webcam detection loop
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # --------------------------------------------------
    # 4. Run YOLO detection
    # --------------------------------------------------
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.90,
        device="cpu",
        verbose=False
    )

    # --------------------------------------------------
    # 5. Draw detections
    # --------------------------------------------------
    annotated_frame = results[0].plot()

    # Number of detected objects
    detection_count = len(results[0].boxes)

    # Display detection count
    cv2.putText(
        annotated_frame,
        f"Objects Detected: {detection_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------
    # 6. Display webcam
    # --------------------------------------------------
    cv2.imshow(
        "Hazardous Detection - Press Q to Exit",
        annotated_frame
    )

    # --------------------------------------------------
    # 7. Press Q to quit
    # --------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# 8. Release resources
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\nWebcam detection stopped.")