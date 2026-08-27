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

MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "runs/hazardous_detection/weights/best.pt"
model = YOLO(MODEL_PATH)

DEFAULT_CONFIDENCE = 0.40
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

        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid or corrupted image file"}), 400

        orig_h, orig_w = image.shape[:2]
        frame_area = orig_w * orig_h

        # Disable gradient computation to prevent RAM spikes and memory leaks
        with torch.no_grad():
            results = model.predict(
                source=image,
                conf=req_confidence,
                iou=0.45,
                imgsz=320,
                verbose=False
            )[0]

        detections = []
        output_img = image.copy()

        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                c_id = int(box.cls[0])

                box_area = (x2 - x1) * (y2 - y1)

                # Filter out frame-filling false positives (walls, background)
                if box_area > (0.50 * frame_area):
                    continue

                name = CLASS_NAMES[c_id] if c_id < len(CLASS_NAMES) else f"Class_{c_id}"

                detections.append({
                    "class_name": name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

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

        _, buffer = cv2.imencode(".jpg", output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoded = base64.b64encode(buffer).decode("utf-8")

        # Explicit RAM cleanup
        del image, output_img, buffer, file_bytes, results
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