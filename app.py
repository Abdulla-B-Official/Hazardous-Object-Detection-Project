import base64
import gc
import os
import cv2
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import torch
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Locate model weights
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "runs/hazardous_detection/weights/best.pt"
model = YOLO(MODEL_PATH)  # type: ignore

DEFAULT_CONFIDENCE = 0.40

# Correct Index Alignment: Index 0 -> ShockAbsorber, Index 1 -> cylinder
CLASS_NAMES = ["ShockAbsorber", "cylinder"]


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

        # 1. Fast binary buffer decode
        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid or corrupted image file"}), 400

        # Store original dimensions for rescaling bounding boxes
        orig_h, orig_w = image.shape[:2]
        frame_area = orig_w * orig_h

        # Downscale immediately to 320x320 for sub-second CPU inference
        image_resized = cv2.resize(image, (320, 320))

        # 2. Optimized Torch inference pass without gradient tracking
        with torch.no_grad():
            results = model.predict(
                source=image_resized,
                conf=req_confidence,
                iou=0.45,
                imgsz=320,
                verbose=False
            )[0]

        detections = []
        output_img = image.copy()
        
        scale_x = orig_w / 320.0
        scale_y = orig_h / 320.0

        if len(results.boxes) > 0:
            for box in results.boxes:
                # Scale coordinates back to original frame size
                rx1, ry1, rx2, ry2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                c_id = int(box.cls[0])

                x1, y1 = int(rx1 * scale_x), int(ry1 * scale_y)
                x2, y2 = int(rx2 * scale_x), int(ry2 * scale_y)
                box_area = (x2 - x1) * (y2 - y1)

                # Filter out frame-filling background false positives
                if box_area > (0.50 * frame_area):
                    continue

                name = CLASS_NAMES[c_id] if c_id < len(CLASS_NAMES) else f"Class_{c_id}"

                detections.append({
                    "class_name": name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

                # Green (0, 255, 0) for ShockAbsorber (Index 0), Orange (0, 165, 255) for cylinder (Index 1)
                color = (0, 255, 0) if c_id == 0 else (0, 165, 255)
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

        # Encode compressed JPEG base64 payload
        _, buffer = cv2.imencode(".jpg", output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        encoded = base64.b64encode(buffer).decode("utf-8")  # type: ignore

        # Clean RAM references explicitly
        del image, image_resized, output_img, buffer, file_bytes, results
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