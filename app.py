import base64
import gc
import os
import cv2
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load PyTorch model weights directly from root or runs directory
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "runs/hazardous_detection/weights/best.pt"

# Initialize PyTorch YOLO model
model = YOLO(MODEL_PATH)

DEFAULT_CONFIDENCE = 0.20
# Exact 2-class mapping: Index 0 = cylinder, Index 1 = ShockAbsorber
CLASS_NAMES = ["cylinder", "ShockAbsorber"]


@app.route("/health")
def health():
    return jsonify({"status": "running", "model_loaded": True, "engine": "PyTorch (best.pt)"})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        req_confidence = float(request.form.get("threshold", DEFAULT_CONFIDENCE))

        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid or corrupted image file"}), 400

        orig_h, orig_w = image.shape[:2]

        # PyTorch YOLOv8 prediction engine
        results = model.predict(
            source=image,
            conf=req_confidence,
            iou=0.45,
            imgsz=640,
            verbose=False
        )[0]

        detections = []
        output_img = image.copy()

        # Process detections
        if len(results.boxes) > 0:
            for box in results.boxes:
                # Extract coordinates, confidence, and class ID directly from PyTorch output
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                c_id = int(box.cls[0])

                name = CLASS_NAMES[c_id] if c_id < len(CLASS_NAMES) else f"Class_{c_id}"

                detections.append({
                    "class_name": name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

                # BGR Format: Green (0, 255, 0) for cylinder | True Orange (0, 165, 255) for ShockAbsorber
                color = (0, 255, 0) if c_id == 0 else (0, 165, 255)

                # Draw bounding box on canvas
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

                label = f"{name} {conf * 100:.1f}%"
                cv2.putText(
                    output_img,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # Encode image response
        _, buffer = cv2.imencode(".jpg", output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        encoded = base64.b64encode(buffer).decode("utf-8")

        # Clean memory
        del image, output_img, buffer, file_bytes
        gc.collect()

        return jsonify({
            "success": True,
            "detection_count": len(detections),
            "detections": detections,
            "annotated_image": encoded,
            "applied_threshold": req_confidence,
        })

    except Exception as e:
        gc.collect()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)