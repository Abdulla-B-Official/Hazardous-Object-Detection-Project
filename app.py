import base64
import gc
import os
import cv2
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# Load lightweight ONNX Runtime session with strict memory management
MODEL_PATH = "best.onnx" if os.path.exists("best.onnx") else "runs/hazardous_detection/weights/best.onnx"

opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False  # Prevents memory hoarding
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.intra_op_num_threads = 2

session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

DEFAULT_CONFIDENCE = 0.20
# Corrected 2-class list matching your Roboflow project setup
CLASS_NAMES = ["cylinder", "ShockAbsorber"]


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

        # Pre-process image for ONNX model shape [1, 3, 640, 640] or [1, 3, 416, 416]
        # Using 640x640 resolution matching your trained standard
        input_size = 640
        img_resized = cv2.resize(image, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension

        # Run direct ONNX inference
        outputs = session.run([output_name], {input_name: input_tensor})[0]

        # Process YOLO output tensor
        preds = np.squeeze(outputs)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # Transpose to [num_boxes, 4 + num_classes]

        boxes, confidences, class_ids = [], [], []
        
        # Scaling factors to map 640x640 predictions back to original image dimensions
        scale_x = orig_w / float(input_size)
        scale_y = orig_h / float(input_size)

        for pred in preds:
            scores = pred[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= req_confidence:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                
                # Rescale center coordinates to original image size
                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                box_w = int(w * scale_x)
                box_h = int(h * scale_y)

                boxes.append([x1, y1, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to clear duplicate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, req_confidence, 0.45)

        detections = []
        output_img = image.copy()

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                # Clip box coordinates within image bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(orig_w, x + w), min(orig_h, y + h)
                
                conf = confidences[i]
                c_id = class_ids[i]
                name = CLASS_NAMES[c_id] if c_id < len(CLASS_NAMES) else f"Class_{c_id}"

                detections.append({
                    "class_name": name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

                # Draw bounding box and label
                color = (0, 255, 0) if name == "cylinder" else (255, 165, 0)
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

        _, buffer = cv2.imencode(".jpg", output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        encoded = base64.b64encode(buffer).decode("utf-8")

        # Memory Cleanup
        del image, img_resized, img_rgb, input_tensor, output_img, buffer, file_bytes
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