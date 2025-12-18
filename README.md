---
title: Traffic AI Analytics – FastAPI
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

# Traffic AI Analytics – FastAPI

This project is an end-to-end **AI-powered traffic analytics system** built with **FastAPI** and deployed on **Hugging Face Spaces**.

## 🚘 Features
- Vehicle detection and tracking (ID-based)
- Helmet violation detection
- Weapon detection
- Fight detection
- Speed estimation
- Event fusion with severity logic
- Annotated output video
- Deduplicated CSV event logs

## 🧠 Architecture
- **Backend:** FastAPI
- **Models:** YOLO (Ultralytics)
- **Pipeline:** Modular, frame-by-frame AI processing
- **Deployment:** Hugging Face Spaces (Docker)

## ▶️ How It Works
1. Upload a video using the `/analyze` API
2. The system processes each frame through:
   - Vehicle tracking
   - Helmet / weapon / fight detection
   - Speed estimation
   - Event fusion
3. Download:
   - Annotated video
   - CSV report of detected events

## 📡 API Endpoints
- `POST /analyze` – Upload a video for analysis
- `GET /download/video/{filename}` – Download processed video
- `GET /download/csv/{filename}` – Download CSV results

## 🛠 Requirements
All dependencies are listed in `requirements.txt`.

## 🚀 Deployment
This Space runs automatically using:
