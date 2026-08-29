# ♻️ HazWaste Vision AI — Intelligent Waste Detection & Classification

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://hazardous-object-detection.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9900?style=for-the-badge)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Data_Prep-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)

**Live Application**

### 🔗 [hazardous-object-detection.onrender.com](https://hazardous-object-detection.onrender.com)

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8s-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Classes-2-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/mAP%4050-99.17%25-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Precision-98.45%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Recall-97.13%25-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/F1--Score-97.78%25-purple?style=for-the-badge" />
</p>

<p align="center">
  <b>Real-time computer vision system for detecting and classifying hazardous waste objects from images and live camera input.</b>
</p>

---

##  Overview

**Hazardous Waste Detection** is an AI-powered computer vision system designed to automatically identify and classify hazardous waste objects using a custom-trained **YOLOv8s object detection model**.

The system focuses on two hazardous waste categories:

* 🔩 **Shock Absorber**
* 🛢️ **Cylinder**

Instead of relying on manual inspection, the system uses deep learning to locate hazardous objects, draw bounding boxes around them, classify them, and provide a confidence score for every detection.

The trained model is integrated into a **Flask API** and deployed as a live web application using **Render**, allowing users to perform detection directly from a browser.

---

##  Live Demo

###  Try the Application

** https://hazardous-object-detection.onrender.com**

The web application provides:

*  Image upload
*  Automatic hazardous-object detection
*  Bounding-box visualization
*  Confidence scores
*  Live webcam detection
*  API/model status monitoring
*  Confidence-threshold control

---

##  Project Objective

The primary objective is to build an automated hazardous-waste detection system that can:

* Automatically detect hazardous waste objects from images
* Classify objects into **Shock Absorber** and **Cylinder**
* Locate objects using bounding boxes
* Provide confidence scores for predictions
* Reduce dependence on manual inspection
* Support real-time webcam detection
* Provide a practical web-based interface
* Deploy the trained AI model as an online service
* Provide a foundation for future industrial conveyor-belt inspection

---

##  Problem Statement

Hazardous waste such as cylinders and shock absorbers can create significant **safety and environmental risks** when they are not correctly identified and handled.

Traditional waste inspection often depends on manual identification, which can be:

*  Time-consuming
*  Labor-intensive
*  Inconsistent
*  Difficult to scale
*  Prone to human error

###  Proposed Solution

This project introduces an AI-powered object detection pipeline that automatically:

**Detects → Classifies → Localizes → Reports**

each hazardous object in an image or camera frame.

For every detection, the system produces:

```text
Object Class
Bounding Box
Confidence Score
```

---

#  How It Works

```text
              INPUT
                │
                ▼
       ┌─────────────────┐
       │ Image / Webcam  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │   Flask API     │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │    YOLOv8s      │
       │   best.pt       │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Object Detection│
       └────────┬────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
   Classification    Bounding Box
        │                │
        └───────┬────────┘
                ▼
       ┌─────────────────┐
       │ Confidence Score│
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Web Application │
       └─────────────────┘
```

---

#  Methodology

The complete project follows a **10-step machine-learning pipeline**.

```text
01. Data Collection
        ↓
02. Annotation
        ↓
03. Augmentation
        ↓
04. Dataset Split
        ↓
05. Model Selection
        ↓
06. Model Training
        ↓
07. Evaluation
        ↓
08. Deployment
        ↓
09. Web Application
        ↓
10. Real-Time Detection
```

---

##  Dataset

Approximately **205 raw images** were initially collected for the two target classes.

### Target Classes

| Class ID | Class          |
| -------: | -------------- |
|        0 | Shock Absorber |
|        1 | Cylinder       |

The dataset was subsequently expanded through augmentation to approximately **1,798 images**, increasing the diversity of training examples.

---

##  Annotation

Images were annotated using the **YOLO bounding-box format**.

Each annotation contains:

```text
class_id
x_center
y_center
width
height
```

Example:

```text
0 0.512 0.487 0.220 0.365
1 0.183 0.622 0.140 0.310
```

Coordinates are normalized between `0` and `1`.

Every object was manually labeled and verified before training.

---

#  Data Preprocessing & Augmentation

**Roboflow** was used for preprocessing and augmentation.

### Preprocessing

* Auto-Orient
* Resize to `640 × 640`

### Augmentation Techniques

#### Augmentation Set 1

* 90° rotation
* Clockwise / counter-clockwise rotation
* 180° rotation
* Exposure variation
* Image noise

#### Augmentation Set 2

* 90° rotation
* Brightness variation

#### Augmentation Set 3

* 90° rotation
* Grayscale applied to a portion of images

### Dataset Growth

```text
205 Raw Images
       ↓
Roboflow Augmentation
       ↓
1,798 Training Images
```

This provided approximately **8.8× growth in dataset diversity**.

---

#  Dataset Split

The final train/validation split was performed in **Google Colab** after shuffling the dataset using a fixed random seed.

```text
80% → Training
20% → Validation

Random Seed → 42
```

### `data.yaml`

```yaml
train: ../train/images
val: ../valid/images

nc: 2

names:
  0: 'shock absorber'
  1: 'cylinder'
```

---

#  Model Selection

The project uses **YOLOv8s** from Ultralytics.

YOLOv8 provides multiple model sizes:

| Model       | Parameters | Characteristics         |
| ----------- | ---------: | ----------------------- |
| YOLOv8n     |       3.2M | Fastest, lightweight    |
| **YOLOv8s** |  **11.2M** | ⭐ Selected              |
| YOLOv8m     |      25.9M | Higher accuracy, slower |
| YOLOv8l     |      43.7M | Heavy                   |
| YOLOv8x     |      68.2M | Largest                 |

### Why YOLOv8s?

YOLOv8s was selected because it provides a practical balance between:

*  Detection accuracy
*  Model size
*  Inference speed
*  Feature representation
*  Potential real-time deployment

The pretrained `yolov8s.pt` model was fine-tuned on the two-class hazardous-waste dataset.

---

#  Training Configuration

The model was trained using **Google Colab GPU acceleration**.

| Parameter             | Configuration    |
| --------------------- | ---------------- |
| Model                 | YOLOv8s          |
| Epochs                | 150              |
| Image Size            | 640 × 640        |
| Batch Size            | 16               |
| Optimizer             | AdamW            |
| Initial Learning Rate | 0.001            |
| LR Schedule           | Cosine           |
| Warmup Epochs         | 5                |
| Patience              | 30               |
| Mosaic                | 1.0              |
| Mixup                 | 0.1              |
| Platform              | Google Colab GPU |

### Training

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001
)
```

The best-performing model checkpoint was saved as:

```text
best.pt
```

---

#  Model Performance

The final evaluation was performed using the best model checkpoint.

| Metric       |      Score |
| ------------ | ---------: |
|  Precision | **98.45%** |
|  Recall    | **97.13%** |
|  F1-Score  | **97.78%** |
|  mAP@50    | **99.17%** |
|  mAP@50–95 | **94.67%** |

### Performance Summary

```text
Precision     ████████████████████  98.45%
Recall        ███████████████████   97.13%
F1 Score      ████████████████████  97.78%
mAP@50        ████████████████████  99.17%
mAP@50-95     ███████████████████   94.67%
```

> **Note:** These metrics represent the evaluation results reported in the project presentation for the trained `best.pt` model.

---

#  Web Application

The trained YOLOv8 model was integrated into a Flask-based web application.

### Application Features

####  Image Detection

Users can upload an image through the browser.

```text
Upload Image
      ↓
Send Image to API
      ↓
YOLOv8 Inference
      ↓
Detect Objects
      ↓
Draw Bounding Boxes
      ↓
Display Results
```

####  Live Webcam Detection

The application also supports real-time webcam detection.

The webcam continuously provides frames to the detection pipeline, allowing hazardous objects to be identified automatically.

####  API Status

The application provides system-status monitoring to indicate whether the API and trained model are available.

---

#  API

The deployed Flask application exposes the following endpoints:

| Method | Endpoint   | Purpose                    |
| ------ | ---------- | -------------------------- |
| `GET`  | `/`        | Serves the web application |
| `GET`  | `/health`  | API/model health status    |
| `POST` | `/predict` | Image prediction           |
| `WS`   | `/webcam`  | Real-time webcam stream    |

---

##  Example Prediction

### Request

```http
POST /predict
Content-Type: multipart/form-data
```

Example:

```text
file = cylinders.jpg
confidence = 0.55
```

### Response

```json
{
  "detections": [
    {
      "class": "cylinder",
      "confidence": 0.97,
      "bbox": [112, 44, 289, 401]
    },
    {
      "class": "shock absorber",
      "confidence": 0.91,
      "bbox": [310, 88, 410, 388]
    }
  ]
}
```

Each detection provides:

* Object class
* Confidence score
* Bounding-box coordinates

---

#  System Architecture

```text
┌──────────────────────────────────────────┐
│              CLIENT LAYER                │
│                                          │
│       HTML / CSS / JavaScript            │
│       Image Upload / Webcam              │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│                API LAYER                 │
│                                          │
│             Flask Backend                │
│     /predict  /health  /webcam           │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│         MODEL / INFERENCE LAYER          │
│                                          │
│              YOLOv8s                    │
│               best.pt                    │
│                                          │
│   Preprocess → Inference → NMS           │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│            INFRASTRUCTURE                │
│                                          │
│               Render                     │
│           HTTPS Deployment               │
│          GitHub Integration              │
└──────────────────────────────────────────┘
```

---

#  Industrial Application

The same detection pipeline can be extended to an automated industrial conveyor-belt inspection system.

```text
Camera
   ↓
Conveyor Belt
   ↓
Image Capture
   ↓
YOLOv8 Detection
   ↓
Classification
   ↓
Confidence Check
   ↓
Decision / Alert
   ↓
Automatic Sorting
   ↓
Logging & Audit
```

### Potential Industrial Workflow

A camera positioned above a conveyor belt continuously captures waste items.

The YOLOv8 model detects hazardous objects and determines their class and confidence.

If a hazardous object satisfies the required confidence threshold:

```text
Hazardous Object
       ↓
Detection
       ↓
Decision
       ↓
Alert / Sorting Signal
       ↓
Robotic Sorting
```

This creates a foundation for automated hazardous-waste management.

---

#  Technology Stack

| Technology                 | Purpose                      |
| -------------------------- | ---------------------------- |
|  Python                  | Programming                  |
|  PyTorch                 | Deep Learning                |
|  YOLOv8 / Ultralytics    | Object Detection             |
|  OpenCV                 | Computer Vision              |
|  Roboflow               | Preprocessing & Augmentation |
|  Google Colab            | Model Training               |
|  Google Drive            | Storage                      |
|  Flask                   | Backend API                  |
|  HTML / CSS / JavaScript | Frontend                     |
|  Render                  | Deployment                   |
|  Git / GitHub            | Version Control              |

---

#  Project Structure

A typical project structure is:

```text
Hazardous_Detection/
│
├── app.py
├── evaluate.py
├── inference.py
├── webcam.py
├── fix_labels.py
├── check_labels.py
├── data.yaml
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── runs/
│   └── hazardous_detection/
│       └── weights/
│           ├── best.pt
│           └── last.pt
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── README.md
```

---

#  Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Hazardous_Detection.git
cd Hazardous_Detection
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the core packages:

```bash
pip install ultralytics flask opencv-python torch torchvision
```

## 4. Run the Application

```bash
python app.py
```

The application can then be accessed locally through the Flask server.

---

#  Running Inference

You can use the trained model directly with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.55,
    imgsz=640
)

for result in results:
    print(result.boxes)
```

---

#  Real-Time Detection

For webcam-based detection:

```text
Webcam
   ↓
Frame Capture
   ↓
YOLOv8s
   ↓
Object Detection
   ↓
Confidence Filtering
   ↓
Bounding Boxes
   ↓
Live Display
```

This can be extended to industrial cameras and CCTV systems.

---

#  Future Enhancements

The current system provides a foundation for a larger intelligent waste-management platform.

###  More Hazardous Waste Categories

Future versions can include:

* Batteries
* Chemical containers
* E-waste
* Oil containers
* Pressurized tanks
* Additional hazardous materials

###  Real-Time Conveyor Detection

Deploy industrial cameras above conveyor belts for continuous monitoring.

###  Automated Sorting

Connect model predictions to robotic arms or automated sorting mechanisms.

###  Alerts & Notifications

Generate automatic alerts when hazardous objects are detected.

###  Edge AI

Deploy the model to edge devices such as:

* NVIDIA Jetson
* Industrial edge computers
* Embedded AI systems

###  Larger Dataset

Improve robustness by collecting more images across:

* Different lighting conditions
* Different viewing angles
* Object occlusion
* Different backgrounds
* Real industrial environments

---

#  Project Highlights

```text
              HAZARDOUS DETECTION
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     YOLOv8s      2 Classes    150 Epochs
        │            │            │
        └────────────┼────────────┘
                     ▼
              1,798 Images
                     │
                     ▼
               best.pt
                     │
                     ▼
          ┌───────────────────┐
          │   98.45%          │
          │   Precision        │
          ├───────────────────┤
          │   97.13% Recall    │
          ├───────────────────┤
          │   97.78% F1        │
          ├───────────────────┤
          │   99.17% mAP@50    │
          └───────────────────┘
                     │
                     ▼
               Live Web App
```

---

#  Key Outcomes

The project successfully delivers a deployed AI-based hazardous-waste detection system capable of:

* ✅ Detecting cylinders
* ✅ Detecting shock absorbers
* ✅ Detecting multiple objects in an image
* ✅ Drawing bounding boxes
* ✅ Providing confidence scores
* ✅ Performing image-based detection
* ✅ Supporting live webcam detection
* ✅ Providing a web interface
* ✅ Serving predictions through a Flask API
* ✅ Operating through a publicly deployed application

---

#  Conclusion

**Hazardous Waste Detection** demonstrates how computer vision and deep learning can be applied to automate hazardous-waste identification.

Starting from approximately **205 raw images**, the dataset was expanded through augmentation, annotated in YOLO format, and used to fine-tune a **YOLOv8s** model for two hazardous-waste categories.

The resulting model achieved:

> **98.45% Precision · 97.13% Recall · 97.78% F1-Score · 99.17% mAP@50**

The model was then integrated with a **Flask API**, deployed using **Render**, and exposed through a web interface supporting image and real-time webcam detection.

The project provides a foundation for future **industrial conveyor-belt monitoring, automated waste sorting, edge AI, and intelligent hazardous-waste management systems.**

---

#  Live Application

<p align="center">

###  Try Hazardous Detection AI

** [Open the Live Application](https://hazardous-object-detection.onrender.com)**

</p>

---

#  Project

**Hazardous Waste Detection & Classification System**

**Built with:** Python · YOLOv8 · PyTorch · OpenCV · Flask · Roboflow · Google Colab · Render · GitHub

---

<p align="center">

###  Detect Smarter. Handle Safer. Automate the Future.

⭐ If you found this project useful, consider giving the repository a star!

</p>

