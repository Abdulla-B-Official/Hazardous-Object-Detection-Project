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

# Point to ONNX file for optimized CPU speed
MODEL_PATH = "best.onnx" if os.path.exists("best.onnx") else "runs/hazardous_detection/weights/best.onnx"
model = YOLO(MODEL_PATH, task="detect")

DEFAULT_CONFIDENCE = 0.20


@app.route("/health")
def health():
    return jsonify({"status": "running", "model_loaded": True})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        req_confidence = float(request.form.get("threshold", DEFAULT_CONFIDENCE))

        # Decode byte stream directly into OpenCV BGR matrix
        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid or corrupted image file"}), 400

        # Match the exact 416px input dimension expected by the ONNX model
        result = model.predict(
            image,
            imgsz=416,
            conf=req_confidence,
            verbose=False,
            max_det=10,
        )[0]

        detections = []

        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < req_confidence:
                continue

            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        # Draw annotations directly onto BGR frame
        output = image.copy()
        for d in detections:
            x1, y1, x2, y2 = (
                d["bbox"]["x1"],
                d["bbox"]["y1"],
                d["bbox"]["x2"],
                d["bbox"]["y2"],
            )
            label = f'{d["class_name"]} {d["confidence"] * 100:.1f}%'

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                label,
                (x1, max(18, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Encode to JPEG in memory
        _, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        encoded = base64.b64encode(buffer).decode("utf-8")

        # Free memory instantly
        del image, output, buffer, file_bytes
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