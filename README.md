# ♻️ AI-Based Hazardous Waste Detection & Classification System

[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-blue?logo=ultralytics)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Deployment](https://img.shields.io/badge/Render-Live_App-46E3B7?logo=render)](https://hazardous-object-detection.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An automated computer-vision system designed to detect and classify hazardous waste objects (such as pressurized cylinders and shock absorbers) from images and live camera feeds using **YOLOv8s**.

🌐 **Live Demo:** [hazardous-object-detection.onrender.com](https://hazardous-object-detection.onrender.com)

---

## 📌 Problem Statement
Hazardous waste items like gas cylinders and mechanical shock absorbers present significant safety and environmental risks if improperly identified or handled. Traditional inspection relies on manual checks, which are time-consuming, prone to human error, and difficult to scale in high-volume waste processing facilities.

This project addresses the problem by providing a high-precision, real-time deep learning pipeline that automatically identifies, bounds, and classifies hazardous items to reduce manual labor and improve safety.

---

## 🎯 Key Objectives
* **Automated Detection:** Identify hazardous items accurately from static images or webcam streams.
* **Target Classes:** Detect and distinguish between **Shock Absorbers** and **Cylinders**.
* **Localization & Scoring:** Return normalized bounding box coordinates and model confidence scores for every detection.
* **Production Ready:** Deploy an accessible web interface and REST API backend.
* **Industrial Application:** Lay the technical foundation for conveyor-belt deployment and automated sorting.

---

## 📊 Dataset & Augmentation

* **Original Dataset:** ~205 raw images manually annotated in YOLO format.
* **Augmentation Pipeline (Roboflow):** Expanded the dataset to **~1,798 images** across 3 distinct augmentation sets to enhance generalization against noise, orientation, and lighting variations:
  * **Set 1:** 90° Rotations, Exposure adjustment ($\pm 14\%$), Pixel Noise (up to 1.98%).
  * **Set 2:** 90° Rotations, Brightness adjustment ($\pm 22\%$).
  * **Set 3:** 90° Rotations, Grayscale conversion (15% probability).
* **Train/Val Split:** 80/20 train-validation split executed in Google Colab (Random Seed: 42).

---

## ⚙️ Model Training & Configurations

The lightweight **YOLOv8s** (`yolov8s.pt`) model was chosen to maintain high inference speeds while preserving feature detection capacity.

| Hyperparameter | Value |
| :--- | :--- |
| **Architecture** | YOLOv8s |
| **Input Image Size** | 640 × 640 |
| **Epochs** | 150 |
| **Batch Size** | 16 |
| **Optimizer** | AdamW |
| **Initial Learning Rate ($lr_0$)** | 0.001 (Cosine Decay) |
| **Warmup Epochs / Patience** | 5 / 30 |
| **Mosaic / Mixup** | 1.0 / 0.1 |

---

## 📈 Evaluation Metrics

Evaluated on the validation set using the best weight parameters (`best.pt`):

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Precision** | 0.9845 | **98.45%** |
| **Recall** | 0.9713 | **97.13%** |
| **F1-Score** | 0.9778 | **97.78%** |
| **mAP@50** | 0.9917 | **99.17%** |
| **mAP@50–95** | 0.9467 | **94.67%** |

---

## 🏗️ System Architecture

  ┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
  │   Input Source   │ ───► │ Image Preprocessing │ ───► │  YOLOv8s Model   │
  │ (Image / Webcam) │      │ (Resize 640x640)    │      │ (Inference Core) │
  └──────────────────┘      └─────────────────────┘      └────────┬─────────┘
                                                                  │
  ┌──────────────────┐      ┌─────────────────────┐               │
  │   Web UI Client  │ ◄─── │      Flask API      │ ◄─────────────┘
  │ (Annotated View) │      │ (/predict endpoint) │
  └──────────────────┘      └─────────────────────┘
