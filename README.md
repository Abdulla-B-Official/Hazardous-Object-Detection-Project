Here is a complete, beautifully formatted `README.md` for your **AI-Based Hazardous Waste Detection & Classification System**, designed to match the sleek, high-visibility style of your WeldVision reference:

```markdown
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

* **Image Detection & Processing:** Upload single or batch images for instant object localization.
* **Real-Time Camera Feed:** Integrated webcam support for continuous monitoring scenarios.
* **YOLOv8 Deep Learning Engine:** Custom-trained **YOLOv8s** architecture optimized for real-time edge/web inference.
* **Bounding Box Overlay:** Visual localization showing exact target position, class label, and confidence score.
* **High-Precision Evaluation:** Trained and validated to achieve high mAP (Mean Average Precision) across target categories.
* **REST API Endpoints:** Web service backend for sending raw frames and receiving structured JSON predictions.
* **Industrial Deployment Architecture:** Built to serve as the baseline vision engine for industrial conveyor belt sorting systems.
* **Web User Interface:** Clean, responsive UI for easy interaction and live detection rendering.

---

## Hazardous Waste Detection Classes

The custom-trained model detects and categorizes key hazardous waste items:

| Class | Status | Risk Level | Description |
| :--- | :---: | :---: | :--- |
| 🟡 **Shock Absorber** | Target | Warning | Pressurized mechanical component requiring controlled handling and disposal. |
| 🔴 **Cylinder** | Target | Critical | High-pressure gas container posing potential explosive or chemical hazards. |

---

## Performance & Metrics

Evaluated on the 20% validation split using the optimal checkpoint (`best.pt`):

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
├── app.py                  # Flask web server routes & prediction REST API
├── requirements.txt        # Backend dependencies & libraries
├── Procfile                # Render deployment execution commands
├── runtime.txt             # Environment Python runtime specification
├── data.yaml               # YOLOv8 class names & dataset paths config
├── models/
│   └── best.pt             # Trained YOLOv8s weight checkpoint
├── static/
│   ├── css/                # Custom styling stylesheet files
│   ├── js/                 # Web interface logic & API communication
│   └── uploads/            # Temporary storage for processed frames
├── templates/
│   └── index.html          # Main HTML web application page
├── notebooks/
│   └── training_colab.ipynb# Model training, validation & export pipeline
└── dataset/
    ├── train/              # 80% Training split (images & YOLO labels)
    └── val/                # 20% Validation split (images & YOLO labels)

```

---

## System Architecture

```text
  ┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
  │   Input Source   │ ───► │ Image Preprocessing │ ───► │  YOLOv8s Model   │
  │ (Image / Webcam) │      │ (Resize 640x640)    │      │ (Inference Core) │
  └──────────────────┘      └─────────────────────┘      └────────┬─────────┘
                                                                  │
  ┌──────────────────┐      ┌─────────────────────┐               │
  │   Web UI Client  │ ◄─── │      Flask API      │ ◄─────────────┘
  │ (Annotated View) │      │ (/predict endpoint) │
  └──────────────────┘      └─────────────────────┘

```

### Industrial Sorting Workflow

```text
[Industrial Camera] ──► [Conveyor Stream] ──► [YOLOv8 Inference] ──► [Hazard Classification] ──► [Automated Actuator / Sorting]

```

---

## Quick Start & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/hazardous-waste-detection.git](https://github.com/your-username/hazardous-waste-detection.git)
cd hazardous-waste-detection

```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
python app.py

```

Open your browser and navigate to `http://localhost:5000`.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

```

```
