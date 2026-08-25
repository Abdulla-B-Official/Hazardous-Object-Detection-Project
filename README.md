# Hazardous Object Detection & Identification Project

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Small-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Live_Demo-brightgreen?style=for-the-badge&logo=render)](https://render.com/)

A comprehensive collection of **Hazardous Object Detection Web-App concepts, AI-powered computer vision pipelines, full-stack implementations, and cloud deployment workflows** documenting my journey in building **production-ready, real-time object detection systems**.

This repository serves as a structured blueprint covering key engineering concepts used to develop **memory-optimized, scalable, interactive, and AI-integrated computer vision applications**. The project focuses on real-world industrial safety scenarios by combining custom dataset training, intelligent inference optimization, web API architecture, containerization, and cloud deployment.

---

## Topics Covered

### Web Development & API Architecture
* Responsive Web Interfaces & Dynamic File Upload Workflows
* RESTful API Endpoint Architecture (`POST /predict`, `GET /health`)
* Multi-part Payload Parsing & Image File Stream Handling
* Base64 Image Encoding/Decoding & JSON Serialization
* Cross-Origin Resource Sharing (CORS) & Flask Integration
* Interactive Thresholding & Dynamic Client-Side Visualization

### Computer Vision & AI Engineering
* Custom Deep Learning Object Detection Pipelines (YOLOv8 Small)
* Dataset Annotation, Stratified Splitting (80/15/5), & Class Balancing
* Data Augmentation Strategies (Mosaic, Mixup, Rotations)
* Inference Optimization for Low-Memory & CPU-Only Environments
* Image Preprocessing (RGB/BGR Color Space Conversions, Resizing)
* Post-Processing (Bounding Box Drawing, Class Labels, Dynamic Confidence Scores)

### Backend Performance & Resource Optimization
* Memory-Constrained Deep Learning Inference (<512MB RAM Limits)
* Disable Autograd Calculations (`torch.set_grad_enabled(False)`)
* Dynamic Memory Cleanup & Python Garbage Collection (`gc.collect()`)
* Thread-Safe Model Loading & PyTorch CPU Execution
* Single-Core Cloud CPU Throughput Optimization

### Containerization, DevOps & Deployment
* Docker Containerization & Multi-Stage Environment Isolation
* WSGI Production Server Configuration (Gunicorn / Flask Engine)
* Production Cloud Hosting & Continuous Integration via Render
* Environment Configuration & Dependency Management (`requirements.txt`)
* Version Control with Git, Remote Repository Linking, & GitHub Workflows

---

## Featured Real-World Projects

* **Hazardous Object Detection System:** Production-ready computer vision web application to identify, localize, and bounding-box annotate industrial hazards (gas cylinders, shock absorbers, industrial waste) in real time.
* **Low-Memory Inference Pipeline:** Containerized PyTorch CPU execution service engineered specifically to run custom YOLO models smoothly under cloud free-tier RAM caps.
* **RESTful Detection Engine API:** Modular Flask microservice delivering structured JSON detection metadata alongside base64-encoded annotated image payloads.

---

## Repository Goal

The purpose of this repository is to strengthen end-to-end computer vision engineering skills by unifying **Artificial Intelligence, Deep Learning, Full-Stack Web Development, Docker Containerization, and Cloud Deployment** to deliver scalable, reliable, and real-time safety inspection systems.

---

## Tools & Technologies

* **Languages:** Python 3.10+, JavaScript (ES6+), HTML5, CSS3
* **Backend Frameworks:** Flask, Flask-CORS, Gunicorn
* **AI & Computer Vision:** PyTorch, Ultralytics YOLOv8, OpenCV (`opencv-python-headless`), Pillow, NumPy
* **Environments & Dev Tools:** Google Colab (GPU Training), VS Code, Git, GitHub, Docker
* **Deployment Platforms:** Render, Docker Engine

---

*This repository will continue to evolve as I explore advanced object detection architectures, real-time video stream processing, optimized model quantizations (ONNX/OpenVINO), edge AI deployment, and industrial safety automation.*
