# ♻️ HazWaste Vision AI — Intelligent Waste Detection & Classification

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://hazardous-object-detection.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9900?style=for-the-badge)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Data_Prep-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)

An **AI-powered computer vision web application** engineered for automated hazardous waste identification and classification using a custom-trained **YOLOv8s object detection model**.

The application analyzes static images and live camera streams to detect, bound, and classify hazardous items (such as gas cylinders and mechanical shock absorbers) in real time. It delivers visual bounding box overlays, confidence score breakdowns, processing latency metrics, and API status monitoring.

**Live Application:** [hazardous-object-detection.onrender.com](https://hazardous-object-detection.onrender.com)

---

## Key Features

* **Image Quality Inspection:** Upload hazardous waste images for single-frame or batch detection.
* **Real-Time Camera Stream:** Live video feed integration for automated inspection.
* **YOLOv8 Deep Learning Engine:** Custom-trained **YOLOv8s** architecture optimized for high accuracy and fast inference.
* **Target Localization & Bounding Boxes:** Precise visual bounding box overlays with class labels and confidence scores.
* **Performance Analytics & Metrics:** High precision and recall tracking across target hazardous waste categories.
* **REST API & Backend Endpoints:** Flask web server providing endpoints for image prediction and status monitoring.
* **Industrial Sorting Concept:** Designed as a foundational vision system for conveyor-belt waste inspection and automated sorting.
* **Modern Interface:** User-friendly UI built with HTML, CSS, JavaScript, and Flask.

---

## Hazardous Waste Detection Classes

The custom-trained YOLO model identifies two key hazardous waste categories:

| Class | Status | Risk Level | Description |
| :--- | :---: | :---: | :--- |
| 🟡 **Shock Absorber** | Target | Warning | Pressurized mechanical component requiring controlled handling and disposal. |
| 🔴 **Cylinder** | Target | Critical | High-pressure gas container posing potential explosive or chemical hazards. |

---

## Dataset & Augmentation

* **Original Dataset:** ~205 raw images annotated in YOLO format.
* **Augmentation Pipeline (Roboflow):** Dataset expanded to **~1,798 images** across 3 distinct augmentation sets to improve generalization:
  * **Set 1:** 90° Rotations (Clockwise, Counter-Clockwise, Upside Down), Exposure ($\pm 14\%$), Pixel Noise (up to 1.98%).
  * **Set 2:** 90° Rotations, Brightness adjustment ($\pm 22\%$).
  * **Set 3:** 90° Rotations, Grayscale conversion (applied to 15% of images).
* **Train/Validation Split:** 80% Training / 20% Validation split performed using Google Colab (Random Seed: 42).

---

## Model Training & Performance Metrics

The model was trained in Google Colab using GPU acceleration with **AdamW**, **Cosine Learning Rate Scheduling**, and **150 Epochs**. Evaluated on the validation split using optimal weights (`best.pt`):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Precision** | 0.9845 | **98.45%** |
| **Recall** | 0.9713 | **97.13%** |
| **F1-Score** | 0.9778 | **97.78%** |
| **mAP@50** | 0.9917 | **99.17%** |
| **mAP@50–95** | 0.9467 | **94.67%** |

---

## Tools & Technologies

* **Computer Vision & AI:** YOLOv8 (Ultralytics), PyTorch, OpenCV, Python
* **Data Processing & Augmentation:** Roboflow, Google Colab (GPU Acceleration)
* **Backend Framework:** Flask REST API, Gunicorn
* **Frontend Interface:** HTML5, CSS3, JavaScript
* **Deployment & Hosting:** Render, Git/GitHub

---

## Project Structure

```text
Hazardous_Waste_Project/
├── app.py                  # Flask server routes & REST API endpoints
├── requirements.txt        # Python backend dependencies
├── Procfile                # Render deployment execution rules
├── runtime.txt             # Python runtime environment specification
├── data.yaml               # YOLO class configuration file
├── models/
│   └── best.pt             # Trained YOLOv8s best model weights
├── static/
│   ├── css/                # User interface styles
│   ├── js/                 # Web application & frontend script logic
│   └── uploads/            # Temporary storage for upload processing
├── templates/
│   └── index.html          # Frontend interface dashboard HTML
├── notebooks/
│   └── training_colab.ipynb# Model training, split, and validation notebook
└── dataset/
    ├── train/              # Training images and YOLO labels (80%)
    └── val/                # Validation images and YOLO labels (20%)
