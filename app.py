import base64
import gc
import os
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from ultralytics import YOLO

# =========================================================
# SYSTEM & HARDWARE OPTIMIZATIONS
# =========================================================
# Limit PyTorch CPU threads to prevent CPU thrashing and RAM spikes on free host tiers
torch.set_num_threads(1)

# =========================================================
# FLASK APPLICATION SETUP
# =========================================================
app = Flask(__name__)
CORS(app)

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

DEFAULT_CONFIDENCE = 0.40
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 416
MAX_BOX_AREA_RATIO = 0.50  # Filter bounding boxes covering > 50% of the total frame

# Class label fallbacks and visual color mappings (BGR format)
FALLBACK_CLASS_NAMES = {0: "ShockAbsorber", 1: "cylinder"}
COLOR_MAP = {
    0: (0, 255, 0),     # ShockAbsorber -> Bright Green
    1: (0, 165, 255)    # cylinder -> Orange
}

# =========================================================
# MODEL VERIFICATION & LOADING
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at expected path:\n{MODEL_PATH}"
    )

print("\n" + "=" * 60)
print("             HAZARDOUS WASTE DETECTION API")
print("=" * 60)
print(f"Base Path    : {BASE_DIR}")
print(f"Model Path   : {MODEL_PATH}")

try:
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model Size   : {model_size_mb:.2f} MB")
except Exception:
    print("Model Size   : Unknown")

print("Loading YOLO model into CPU memory...")
model = YOLO(MODEL_PATH)

# Extract trained class names directly from model if present
CLASS_NAMES = getattr(model, "names", FALLBACK_CLASS_NAMES)
print(f"Model Classes: {CLASS_NAMES}")

# Perform a cold-start dummy prediction to warm up PyTorch/YOLO engine
print("Warming up inference engine...")
try:
    dummy_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    with torch.inference_mode():
        _ = model.predict(source=dummy_img, imgsz=IMAGE_SIZE, verbose=False)
    del dummy_img
    print("Inference engine warmed up successfully.")
except Exception as e:
    print(f"Warmup warning: {e}")

print("=" * 60 + "\n")


# =========================================================
# ROUTES
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for status monitoring."""
    return jsonify({
        "status": "running",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "image_size": IMAGE_SIZE,
        "default_confidence": DEFAULT_CONFIDENCE,
        "iou_threshold": IOU_THRESHOLD
    })


@app.route("/", methods=["GET"])
def home():
    """Renders the main web page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Object detection endpoint for processing uploaded images and webcam frames."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    image = None
    file_bytes = None
    results = None
    output_img = None

    try:
        # 1. Request parameters parsing
        try:
            req_confidence = float(request.form.get("threshold", DEFAULT_CONFIDENCE))
        except ValueError:
            req_confidence = DEFAULT_CONFIDENCE

        req_confidence = max(0.0, min(1.0, req_confidence))
        is_webcam = request.form.get("is_webcam", "false").lower() == "true"

        # 2. Decode image stream efficiently
        uploaded_file = request.files["image"]
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "error": "Invalid or corrupted image format"}), 400

        orig_h, orig_w = image.shape[:2]
        frame_area = orig_h * orig_w

        # 3. Model Inference inside no-grad context
        with torch.inference_mode():
            results = model.predict(
                source=image,
                conf=req_confidence,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False
            )[0]

        # 4. Extract and filter bounding boxes
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes_data = results.boxes

            for box in boxes_data:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                # Clamp bounding boxes inside valid frame boundaries
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                box_area = max(0, x2 - x1) * max(0, y2 - y1)

                # Ignore massive background false positives
                if frame_area > 0 and box_area > (MAX_BOX_AREA_RATIO * frame_area):
                    continue

                # Obtain class name safely
                if isinstance(CLASS_NAMES, dict):
                    class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
                elif isinstance(CLASS_NAMES, list) and 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                else:
                    class_name = f"Class_{class_id}"

                detections.append({
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "c_id": class_id,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

        # Sort detections by confidence score (descending)
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        # 5. Image annotation and Base64 Encoding (Skipped for webcam feeds to preserve bandwidth)
        encoded_image = ""

        if not is_webcam:
            output_img = image.copy()

            for det in detections:
                bbox = det["bbox"]
                cid = det["c_id"]
                c_name = det["class_name"]
                conf_val = det["confidence"]

                color = COLOR_MAP.get(cid, (255, 255, 0))  # Default Cyan if unknown class

                # Draw bounding box
                cv2.rectangle(output_img, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), color, 2)

                # Draw class label background & text
                label_text = f"{c_name} {conf_val * 100:.1f}%"
                label_y = max(20, bbox["y1"] - 10)
                
                cv2.putText(
                    output_img,
                    label_text,
                    (bbox["x1"], label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # JPEG encoding with compression quality = 80
            success, buffer = cv2.imencode(".jpg", output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                encoded_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "detection_count": len(detections),
            "detections": detections,
            "annotated_image": encoded_image,
            "applied_threshold": req_confidence
        })

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        # Explicit garbage collection to free memory on small RAM instances
        del image, file_bytes, results, output_img
        gc.collect()


# =========================================================
# APPLICATION LAUNCHER
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting Flask Server on Port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)