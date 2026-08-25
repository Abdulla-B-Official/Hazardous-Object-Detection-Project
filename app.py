import base64
import gc
import io
import os
import cv2
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# --- MEMORY & CPU OPTIMIZATIONS FOR RENDER ---
# Allow PyTorch to use available CPU threads dynamically
torch.set_grad_enabled(False)  # Disable autograd calculations (inference only)

app = Flask(__name__)
CORS(app)

# Explicitly point to the best.pt in root directory
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "runs/hazardous_detection/weights/best.pt"
model = YOLO(MODEL_PATH)

DEFAULT_CONFIDENCE = 0.35


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

        # Read image
        image_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Fast inference with YOLOv8n
        result = model.predict(
            image,
            imgsz=416,
            conf=req_confidence,
            device="cpu",
            verbose=False,
            max_det=20,
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

        # Draw detections on output copy
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

        # Convert annotated image back to Base64
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(output)

        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=60)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Cleanup RAM
        del image, output, buffer, image_bytes
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