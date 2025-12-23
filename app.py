import os
import uuid
import shutil
import cv2
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from pipeline.vehicle_tracker import VehicleTracker
from pipeline.helmet_detector import HelmetDetector
from pipeline.weapon_detector import WeaponDetector
from pipeline.fight_detector import FightDetector
from pipeline.event_manager import EventManager

from utils.csv_logger import init_csv, log_event
from utils.json_logger import init_json, log_json_event
from utils.config import OUTPUT_DIR
from utils.drawing import draw_box
from utils.dedup import EventDeduplicator

app = FastAPI(title="Traffic & Public Safety AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

vehicle_tracker = VehicleTracker()
helmet_detector = HelmetDetector()
weapon_detector = WeaponDetector()
fight_detector = FightDetector()
event_manager = EventManager()
dedup = EventDeduplicator()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported video format")

    run_id = str(uuid.uuid4())[:8]

    input_path = OUTPUT_DIR / f"input_{run_id}.mp4"
    output_path = OUTPUT_DIR / f"output_{run_id}.mp4"
    csv_path = OUTPUT_DIR / f"events_{run_id}.csv"
    json_path = OUTPUT_DIR / f"events_{run_id}.json"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(500, "Video open failed")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    init_csv(csv_path)
    init_json(json_path)

    frame_id = 0
    CAMERA_ID = "CAM_01"

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
                if not dedup.allow(key):
                    continue

                common = dict(
                    detection_uuid=str(uuid.uuid4()),
                    file_name=input_path.name,
                    camera_id=CAMERA_ID,
                    frame_id=frame_id,
                    object_id=t["track_id"],
                    lane=t.get("lane"),
                    speed_kmph=t.get("speed"),
                    overspeed=ev["type"] == "overspeed",
                    label=ev["label"],
                    confidence=ev["confidence"],
                    color=ev["color"],
                    helmet_violation=ev["type"] == "helmet_violation",
                    weapon_detected=ev["type"] == "weapon",
                    fight_detected=ev["type"] == "fight",
                    fight_confidence=ev.get("confidence"),
                    fight_participants=ev.get("participants"),
                    fight_severity=ev.get("severity"),
                    fight_duration_sec=ev.get("duration"),
                    violation_type=ev["type"],
                    anpr_plate=ev.get("plate"),
                    anpr_confidence=ev.get("plate_conf"),
                    hotlist_match=ev.get("hotlist", False),
                    violation_image=ev.get("evidence_path"),
                    meta={"pipeline": "traffic-ai-v1"},
                )

                log_event(csv_path, **common)
                log_json_event(json_path, **common)

                draw_box(frame, t["bbox"], ev["label"], ev["color"])

        writer.write(frame)

    cap.release()
    writer.release()

    return {
        "status": "success",
        "video": f"/download/video/{output_path.name}",
        "csv": f"/download/csv/{csv_path.name}",
        "json": f"/download/json/{json_path.name}",
    }

@app.get("/download/video/{name}")
def download_video(name: str):
    return FileResponse(OUTPUT_DIR / name, media_type="video/mp4")

@app.get("/download/csv/{name}")
def download_csv(name: str):
    return FileResponse(OUTPUT_DIR / name, media_type="text/csv")

@app.get("/download/json/{name}")
def download_json(name: str):
    return FileResponse(OUTPUT_DIR / name, media_type="application/json")
