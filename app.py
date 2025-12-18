import os
import uuid
import shutil
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

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

app = FastAPI(title="Traffic AI Analytics System")

# ------------------ LOAD MODELS ONCE ------------------
vehicle_tracker = VehicleTracker()
helmet_detector = HelmetDetector()
weapon_detector = WeaponDetector()
fight_detector = FightDetector()
speed_estimator = SpeedEstimator()
event_manager = EventManager()
dedup = EventDeduplicator()

# -----------------------------------------------------

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())[:8]
    input_path = OUTPUT_DIR / f"input_{video_id}.mp4"
    output_path = OUTPUT_DIR / f"output_{video_id}.mp4"
    csv_path = OUTPUT_DIR / f"events_{video_id}.csv"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    init_csv(csv_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        tracks = vehicle_tracker.process_frame(frame)

        for t in tracks:
            track_id = t["track_id"]
            bbox = t["bbox"]
            cls = t["class_name"]

            # Helmet check
            helmet = helmet_detector.check(frame, t)

            # Weapon + fight
            weapon = weapon_detector.check(frame, t)
            fight = fight_detector.check(frame, t)

            # Speed
            speed = speed_estimator.update(track_id, bbox)

            events = event_manager.fuse(
                track_id, cls, helmet, weapon, fight, speed
            )

            for e in events:
                if dedup.allow(f"{track_id}-{e['type']}"):
                    log_event(csv_path, [
                        frame_id,
                        e["type"],
                        track_id,
                        e["confidence"],
                        e["details"]
                    ])

                draw_box(
                    frame,
                    bbox,
                    e["label"],
                    e["color"]
                )

        writer.write(frame)

    cap.release()
    writer.release()

    return {
        "status": "success",
        "video": f"/download/video/{output_path.name}",
        "csv": f"/download/csv/{csv_path.name}"
    }

@app.get("/download/video/{filename}")
def download_video(filename: str):
    return FileResponse(OUTPUT_DIR / filename, media_type="video/mp4")

@app.get("/download/csv/{filename}")
def download_csv(filename: str):
    return FileResponse(OUTPUT_DIR / filename, media_type="text/csv")
