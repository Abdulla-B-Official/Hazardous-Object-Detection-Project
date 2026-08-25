from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("runs/hazardous_detection/weights/best.pt")

# Default minimum confidence if not supplied by the frontend
DEFAULT_CONFIDENCE = 0.35


@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": True
    })


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Dynamically retrieve confidence threshold from frontend (slider)
        req_confidence = float(request.form.get("threshold", DEFAULT_CONFIDENCE))

        # Read image
        image = Image.open(
            io.BytesIO(request.files["image"].read())
        ).convert("RGB")

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # YOLO prediction using the dynamic confidence threshold
        result = model.predict(
            image,
            imgsz=640,
            conf=req_confidence,
            device="cpu",
            verbose=False
        )[0]

        detections = []

        # Get detections
        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < req_confidence:
                continue

            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

        # Draw only accepted detections
        output = image.copy()

        for d in detections:

            x1 = d["bbox"]["x1"]
            y1 = d["bbox"]["y1"]
            x2 = d["bbox"]["x2"]
            y2 = d["bbox"]["y2"]

            label = f'{d["class_name"]} {d["confidence"] * 100:.1f}%'

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                label,
                (x1, max(18, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Convert image to Base64
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(output)

        buffer = io.BytesIO()
        output.save(buffer, format="JPEG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "detection_count": len(detections),
            "detections": detections,
            "annotated_image": encoded,
            "applied_threshold": req_confidence
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )