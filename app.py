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
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

# Get the folder where this app.py file is located.
# This makes the code work on both:
#   Windows / VS Code
#   Render / Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Expected model location:
# Hazardous_Detection/
# ├── app.py
# └── best.pt
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# YOLO inference settings
DEFAULT_CONFIDENCE = 0.40
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 416

# Class names must match the order used during YOLO training.
# 0 -> ShockAbsorber
# 1 -> cylinder
CLASS_NAMES = [
    "ShockAbsorber",
    "cylinder"
]


# =========================================================
# CHECK MODEL BEFORE LOADING
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found!\n"
        f"Expected location:\n{MODEL_PATH}"
    )


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("\n" + "=" * 60)
print("              HAZARDOUS DETECTION API")
print("=" * 60)
print(f"Base directory : {BASE_DIR}")
print(f"Model path     : {MODEL_PATH}")
print(f"Model exists   : {os.path.exists(MODEL_PATH)}")

try:
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model size     : {model_size_mb:.2f} MB")
except Exception:
    print("Model size     : Unable to determine")

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")
print(f"Classes        : {CLASS_NAMES}")
print(f"Image size     : {IMAGE_SIZE}")
print(f"Default conf.  : {DEFAULT_CONFIDENCE}")
print(f"IoU threshold  : {IOU_THRESHOLD}")
print("=" * 60 + "\n")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "default_confidence": DEFAULT_CONFIDENCE,
        "iou_threshold": IOU_THRESHOLD
    })


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    image = None
    file_bytes = None
    results = None
    output_img = None
    buffer = None

    try:

        # -------------------------------------------------
        # GET REQUEST PARAMETERS
        # -------------------------------------------------

        try:
            req_confidence = float(
                request.form.get(
                    "threshold",
                    DEFAULT_CONFIDENCE
                )
            )
        except ValueError:
            req_confidence = DEFAULT_CONFIDENCE

        # Keep confidence within valid range
        req_confidence = max(
            0.0,
            min(1.0, req_confidence)
        )

        is_webcam = (
            request.form.get(
                "is_webcam",
                "false"
            ).lower() == "true"
        )


        # -------------------------------------------------
        # READ IMAGE
        # -------------------------------------------------

        uploaded_file = request.files["image"]

        file_bytes = np.frombuffer(
            uploaded_file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify({
                "success": False,
                "error": "Invalid or corrupted image file"
            }), 400


        # -------------------------------------------------
        # ORIGINAL IMAGE DIMENSIONS
        # -------------------------------------------------

        orig_h, orig_w = image.shape[:2]

        frame_area = orig_w * orig_h


        # -------------------------------------------------
        # YOLO INFERENCE
        #
        # IMPORTANT:
        # Do NOT manually resize the image to 416x416.
        #
        # Ultralytics performs its own preprocessing/
        # letterboxing, which helps preserve the original
        # image geometry.
        # -------------------------------------------------

        with torch.inference_mode():

            results = model.predict(
                source=image,
                conf=req_confidence,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )[0]


        # -------------------------------------------------
        # PROCESS DETECTIONS
        # -------------------------------------------------

        detections = []

        if results.boxes is not None and len(results.boxes) > 0:

            for box in results.boxes:

                # Bounding box returned by Ultralytics
                # is already mapped appropriately for the
                # original image.
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                conf = float(box.conf[0])

                class_id = int(box.cls[0])


                # -------------------------------------------------
                # CLAMP COORDINATES
                # -------------------------------------------------

                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))


                # -------------------------------------------------
                # BOX AREA
                # -------------------------------------------------

                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)

                box_area = box_width * box_height


                # -------------------------------------------------
                # FILTER HUGE BACKGROUND FALSE POSITIVES
                #
                # Ignore a detection if it covers more than
                # 50% of the complete image.
                # -------------------------------------------------

                if frame_area > 0:

                    if box_area > (0.50 * frame_area):
                        continue


                # -------------------------------------------------
                # CLASS NAME
                # -------------------------------------------------

                if 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                else:
                    class_name = f"Class_{class_id}"


                # -------------------------------------------------
                # STORE DETECTION
                # -------------------------------------------------

                detections.append({
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "c_id": class_id,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })


        # -------------------------------------------------
        # SORT DETECTIONS
        # Highest confidence first
        # -------------------------------------------------

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )


        # -------------------------------------------------
        # ENCODE ANNOTATED IMAGE
        #
        # Only do this for normal image uploads.
        #
        # Webcam requests receive JSON detections only.
        # This saves bandwidth and improves responsiveness.
        # -------------------------------------------------

        encoded = ""

        if not is_webcam:

            output_img = image.copy()


            for detection in detections:

                bbox = detection["bbox"]

                class_id = detection["c_id"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]


                # -------------------------------------------------
                # COLORS
                # -------------------------------------------------

                # ShockAbsorber -> Green
                # Cylinder -> Orange
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)


                # -------------------------------------------------
                # DRAW BOUNDING BOX
                # -------------------------------------------------

                cv2.rectangle(
                    output_img,
                    (
                        bbox["x1"],
                        bbox["y1"]
                    ),
                    (
                        bbox["x2"],
                        bbox["y2"]
                    ),
                    color,
                    2
                )


                # -------------------------------------------------
                # LABEL
                # -------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                label_x = bbox["x1"]

                label_y = max(
                    20,
                    bbox["y1"] - 10
                )


                cv2.putText(
                    output_img,
                    label,
                    (
                        label_x,
                        label_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )


            # -------------------------------------------------
            # JPEG ENCODING
            # -------------------------------------------------

            success, buffer = cv2.imencode(
                ".jpg",
                output_img,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80
                ]
            )

            if success:

                encoded = base64.b64encode(
                    buffer
                ).decode("utf-8")

            else:

                encoded = ""


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        response = {
            "success": True,
            "detection_count": len(detections),
            "detections": detections,
            "annotated_image": encoded,
            "applied_threshold": req_confidence
        }


        return jsonify(response)


    except Exception as e:

        print(
            "\nPrediction error:",
            str(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


    finally:

        # -------------------------------------------------
        # MEMORY CLEANUP
        # -------------------------------------------------

        image = None
        file_bytes = None
        results = None
        output_img = None
        buffer = None

        gc.collect()


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print(f"Port: {port}")
    print(f"Model: {MODEL_PATH}")
    print("=" * 60 + "\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )