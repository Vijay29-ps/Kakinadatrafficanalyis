# app.py — FINAL FastAPI entrypoint for HF Spaces
import os
import uuid
import shutil
import cv2
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# ----------------- PIPELINE IMPORTS -----------------
from pipeline.vehicle_tracker import VehicleTracker
from pipeline.helmet_detector import HelmetDetector
from pipeline.weapon_detector import WeaponDetector
from pipeline.fight_detector import FightDetector
from pipeline.speed_estimator import SpeedEstimator
from pipeline.event_manager import EventManager

from utils.config import OUTPUT_DIR
from utils.csv_logger import init_csv, log_event
from utils.dedup import EventDeduplicator
from utils.drawing import draw_box

# ----------------- FASTAPI APP -----------------
app = FastAPI(
    title="Traffic AI Analytics – FastAPI",
    description="Vehicle, Helmet, Weapon, Fight, Speed analytics",
    version="1.0"
)

# ----------------- LOAD MODELS ONCE (IMPORTANT) -----------------
vehicle_tracker = VehicleTracker()
helmet_detector = HelmetDetector()
weapon_detector = WeaponDetector()
fight_detector = FightDetector()
speed_estimator = SpeedEstimator()
event_manager = EventManager()
dedup = EventDeduplicator()

# ----------------- HEALTH CHECK -----------------
@app.get("/")
def health():
    return {"status": "running", "service": "Traffic AI FastAPI"}

# ----------------- MAIN ANALYSIS ENDPOINT -----------------
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    run_id = str(uuid.uuid4())[:8]

    input_path = OUTPUT_DIR / f"input_{run_id}.mp4"
    output_path = OUTPUT_DIR / f"output_{run_id}.mp4"
    csv_path = OUTPUT_DIR / f"events_{run_id}.csv"

    # Save uploaded file
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    init_csv(csv_path)
    frame_id = 0

    # ----------------- SINGLE VIDEO LOOP -----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        tracks = vehicle_tracker.process_frame(frame)

        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            cls_name = track["class_name"]

            helmet = helmet_detector.check(frame, track)
            weapon = weapon_detector.check(frame, track)
            fight  = fight_detector.check(frame, track)
            speed  = speed_estimator.update(track_id, bbox)

            events = event_manager.fuse(
                track=track,
                helmet=helmet,
                weapon=weapon,
                fight=fight,
                speed=speed
            )

            for ev in events:
                dedup_key = f"{track_id}-{ev['type']}"
                if dedup.allow(dedup_key):
                    log_event(csv_path, [
                        frame_id,
                        ev["type"],
                        track_id,
                        ev["confidence"],
                        ev["details"]
                    ])

                draw_box(
                    frame,
                    bbox,
                    ev["label"],
                    ev["color"]
                )

        writer.write(frame)

    cap.release()
    writer.release()

    return JSONResponse({
        "status": "success",
        "run_id": run_id,
        "outputs": {
            "video": f"/download/video/{output_path.name}",
            "csv": f"/download/csv/{csv_path.name}"
        }
    })

# ----------------- DOWNLOAD ENDPOINTS -----------------
@app.get("/download/video/{filename}")
def download_video(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4")

@app.get("/download/csv/{filename}")
def download_csv(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv")
