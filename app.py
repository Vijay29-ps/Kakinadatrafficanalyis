import os
import sys
import uuid
import shutil
import cv2
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------
# Environment fixes for Hugging Face
# ------------------------------------------------------------------
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
from pipeline.vehicle_tracker import VehicleTracker
from pipeline.helmet_detector import HelmetDetector
from pipeline.weapon_detector import WeaponDetector
from pipeline.fight_detector import FightDetector
from pipeline.event_manager import EventManager

from utils.config import OUTPUT_DIR
from utils.csv_logger import init_csv, log_event
from utils.dedup import EventDeduplicator
from utils.drawing import draw_box

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="Kakinada Traffic AI Analytics",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ------------------------------------------------------------------
# CORS (REQUIRED for Vercel frontend)
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://video-ten-silk.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Ensure output directory exists
# ------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# Load pipelines once (important for performance)
# ------------------------------------------------------------------
vehicle_tracker = VehicleTracker()
helmet_detector = HelmetDetector()
weapon_detector = WeaponDetector()
fight_detector = FightDetector()
event_manager = EventManager()
dedup = EventDeduplicator()

# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Traffic AI FastAPI",
    }

# ------------------------------------------------------------------
# Main analysis endpoint
# ------------------------------------------------------------------
@app.post("/analyze", summary="Analyze traffic video")
async def analyze_video(
    file: UploadFile = File(..., description="Upload MP4/AVI/MOV/MKV/WEBM video"),
):
    if not file.filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ):
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
        raise HTTPException(status_code=500, detail="Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    init_csv(csv_path)
    frame_id = 0

    # --------------------- frame loop ---------------------
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
                    log_event(
                        csv_path,
                        [
                            frame_id,
                            ev["type"],
                            t["track_id"],
                            ev["confidence"],
                            ev["details"],
                        ],
                    )

                draw_box(frame, t["bbox"], ev["label"], ev["color"])

        writer.write(frame)

    cap.release()
    writer.release()

    return JSONResponse(
        {
            "status": "success",
            "video": f"/download/video/{output_path.name}",
            "csv": f"/download/csv/{csv_path.name}",
        }
    )

# ------------------------------------------------------------------
# Download endpoints
# ------------------------------------------------------------------
@app.get("/download/video/{filename}")
def download_video(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")

@app.get("/download/csv/{filename}")
def download_csv(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(path, media_type="text/csv")
