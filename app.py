import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import uuid
import shutil
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from pipeline.vehicle_tracker import VehicleTracker
from pipeline.helmet_detector import HelmetDetector
from pipeline.weapon_detector import WeaponDetector
from pipeline.fight_detector import FightDetector
from pipeline.event_manager import EventManager

from utils.config import OUTPUT_DIR
from utils.csv_logger import init_csv, log_event
from utils.dedup import EventDeduplicator
from utils.drawing import draw_box

app = FastAPI(title="Kakinada Traffic AI Analytics")

vehicle_tracker = VehicleTracker()
helmet_detector = HelmetDetector()
weapon_detector = WeaponDetector()
fight_detector = FightDetector()
event_manager = EventManager()
dedup = EventDeduplicator()

@app.get("/")
def health():
    return {"status": "running", "service": "Traffic AI FastAPI"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    run_id = str(uuid.uuid4())[:8]
    input_path = OUTPUT_DIR / f"input_{run_id}.mp4"
    output_path = OUTPUT_DIR / f"output_{run_id}.mp4"
    csv_path = OUTPUT_DIR / f"events_{run_id}.csv"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    init_csv(csv_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        tracks = vehicle_tracker.process_frame(frame)

        for t in tracks:
            helmet = helmet_detector.check(frame, t)
            weapon = weapon_detector.check(frame, t)
            fight = fight_detector.check(frame, t)

            events = event_manager.fuse(t, helmet, weapon, fight)

            for ev in events:
                key = f"{t['track_id']}-{ev['type']}"
                if dedup.allow(key):
                    log_event(csv_path, [
                        frame_id,
                        ev["type"],
                        t["track_id"],
                        ev["confidence"],
                        ev["details"]
                    ])

                draw_box(frame, t["bbox"], ev["label"], ev["color"])

        writer.write(frame)

    cap.release()
    writer.release()

    return JSONResponse({
        "status": "success",
        "video": f"/download/video/{output_path.name}",
        "csv": f"/download/csv/{csv_path.name}"
    })

@app.get("/download/video/{filename}")
def download_video(filename: str):
    return FileResponse(OUTPUT_DIR / filename)

@app.get("/download/csv/{filename}")
def download_csv(filename: str):
    return FileResponse(OUTPUT_DIR / filename)
