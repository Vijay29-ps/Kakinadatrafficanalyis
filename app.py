# app.py - simplified stable server for HF Space
import os
os.environ.setdefault("YOLO_CONFIG_DIR", os.environ.get("YOLO_CONFIG_DIR", "/app/.ultralytics"))
from pathlib import Path
import shutil, time, uuid, csv, json, tempfile, zipfile, math
from collections import defaultdict, deque
from datetime import datetime

# create config dir
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import webcolors
import requests

# ultralytics (import after YOLO_CONFIG_DIR)
from ultralytics import YOLO

# optional hf helper
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

# --- CONFIG ---
OUTPUT_DIR = Path("./outputs"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("./model"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
GENERAL_MODEL_PATH = "yolov8n.pt"  # let Ultralytics download if missing
HELMET_CANDIDATES = [MODELS_DIR / "best.pt", Path("/mnt/data/best.pt"), Path("")]

PX_PER_METER = 20.0
OVERSPEED_THRESHOLD_KMPH = 40.0

ALLOWED_ORIGINS = ["https://video-ten-silk.vercel.app", "http://localhost:3000", "http://127.0.0.1:3000"]

CSV_HEADERS = ["file_name","camera_id","frame_id","vehicle_id","vehicle_type","conf","color_hex","color_name","lane",
               "speed_kmph","overspeed","helmet_violation","anpr_plate","anpr_confidence","hotlist_match","weapon_detected",
               "violation_type","violation_image","timestamp","meta","crowd_count","density_level"]

# --- utility functions (kept minimal) ---
def make_vehicle_id(): return str(uuid.uuid4())
def centroid_from_bbox(bbox): x1,y1,x2,y2=bbox; return ((x1+x2)/2.0, (y1+y2)/2.0)
def euclidean(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

# minimal mean color (fast)
def mean_color_hex(img, bbox):
    x1,y1,x2,y2 = [int(round(x)) for x in bbox]
    x1,y1,x2,y2 = max(0,x1), max(0,y1), min(img.shape[1]-1,x2), min(img.shape[0]-1,y2)
    crop = img[y1:y2, x1:x2]
    if crop.size==0: return "#000000"
    avg = crop.mean(axis=(0,1))
    r,g,b = int(avg[2]), int(avg[1]), int(avg[0])
    return "#{:02X}{:02X}{:02X}".format(r,g,b)

def hex_to_name(hexc):
    try: return webcolors.hex_to_name(hexc, spec='css3')
    except: return hexc

# --- MODEL LOADING (once) ---
def find_helmet_model():
    for cand in HELMET_CANDIDATES:
        try:
            if cand.exists(): return str(cand)
        except Exception:
            continue
    # optional env URL or HF repo fallback (not required)
    url = os.environ.get("HELMET_MODEL_URL")
    if url:
        try:
            dest = MODELS_DIR / "best.pt"
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk: f.write(chunk)
            return str(dest)
        except Exception as e:
            print("helmet URL download failed:", e)
    hf_repo = os.environ.get("HELMET_HF_REPO")
    if hf_repo and hf_hub_download is not None:
        try:
            token = os.environ.get("HF_TOKEN")
            path = hf_hub_download(repo_id=hf_repo, filename=os.environ.get("HELMET_HF_FILENAME","best.pt"), use_auth_token=token)
            dest = MODELS_DIR / Path(path).name
            shutil.copy(path, dest)
            return str(dest)
        except Exception as e:
            print("hf hub download failed:", e)
    return None

print("Loading YOLO general model (may download if needed)...")
GENERAL = YOLO(GENERAL_MODEL_PATH)
HELMET_MODEL = None
helmet_path = find_helmet_model()
if helmet_path:
    try:
        HELMET_MODEL = YOLO(helmet_path)
        print("✔ Helmet model loaded:", helmet_path)
    except Exception as e:
        print("Failed loading helmet model:", e)
else:
    print("No helmet model found; continuing without helmet checks.")

# --- FASTAPI app ---
app = FastAPI(title="Helmet Vehicle Processor (stable)")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status":"ok","helmet_loaded": bool(HELMET_MODEL)}

@app.post("/process")
async def process(file: UploadFile = File(...), camera_id: str = Form("camera_0")):
    tmp = Path(tempfile.mkdtemp())
    try:
        in_path = tmp / file.filename
        in_path.write_bytes(await file.read())
        suffix = in_path.suffix.lower()
        is_image = suffix in (".jpg",".jpeg",".png",".bmp",".webp")
        # small helper to run detector once and make records
        def run_detection_img(img):
            recs = []
            res = GENERAL(img)[0]
            helmet_boxes = []
            if HELMET_MODEL is not None:
                try:
                    hr = HELMET_MODEL(img)[0]
                    for hb in hr.boxes:
                        helmet_boxes.append({"bbox": hb.xyxy[0].cpu().numpy().tolist(), "cls": int(hb.cls[0]), "conf": float(hb.conf[0])})
                except Exception:
                    helmet_boxes = []
            for b in res.boxes:
                bbox = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                vehicle_type = GENERAL.names[cls]
                color_hex = mean_color_hex(img, bbox)
                color_name = hex_to_name(color_hex)
                helmet_violation = False
                if helmet_boxes:
                    x1,y1,x2,y2 = bbox
                    for hb in helmet_boxes:
                        hx1,hy1,hx2,hy2 = hb["bbox"]
                        if max(0, min(x2,hx2) - max(x1,hx1))>0 and max(0, min(y2,hy2) - max(y1,hy1))>0:
                            helmet_violation = (hb["cls"]==1)
                            break
                recs.append({"vehicle_type": vehicle_type, "conf": conf, "color_hex": color_hex,
                             "color_name": color_name, "helmet_violation": helmet_violation, "bbox": bbox})
            return recs

        if is_image:
            img = cv2.imread(str(in_path))
            out_img_path = OUTPUT_DIR / f"annotated_{int(time.time()*1000)}.jpg"
            recs = run_detection_img(img)
            # annotate simple
            for r in recs:
                x1,y1,x2,y2 = [int(round(x)) for x in r["bbox"]]
                clr = (0,0,255) if r["helmet_violation"] else (255,255,0)
                cv2.rectangle(img, (x1,y1),(x2,y2), clr, 2)
            cv2.imwrite(str(out_img_path), img)
            # write simple json + csv + zip
            ts = int(time.time()*1000)
            json_path = OUTPUT_DIR / f"result_{ts}.json"
            csv_path = OUTPUT_DIR / f"results_{ts}.csv"
            with open(json_path, "w") as jf: json.dump(recs, jf, indent=2)
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                import csv as _csv
                w = _csv.DictWriter(cf, fieldnames=["vehicle_type","conf","color_hex","color_name","helmet_violation"])
                w.writeheader(); 
                for r in recs: w.writerow({"vehicle_type":r["vehicle_type"],"conf":r["conf"],"color_hex":r["color_hex"],"color_name":r["color_name"],"helmet_violation":r["helmet_violation"]})
            zipname = OUTPUT_DIR / f"results_{ts}.zip"
            with zipfile.ZipFile(zipname, "w") as zf:
                zf.write(out_img_path, arcname=out_img_path.name)
                zf.write(json_path, arcname=json_path.name)
                zf.write(csv_path, arcname=csv_path.name)
            return FileResponse(path=str(zipname), filename=zipname.name, media_type="application/zip")
        else:
            # for video: use simpler approach (frame-sample detection to avoid heavy processing)
            cap = cv2.VideoCapture(str(in_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_vid = OUTPUT_DIR / f"annotated_{int(time.time()*1000)}.mp4"
            writer = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
            frame_id = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_id += 1
                    # run detection every Nth frame to save CPU
                    if frame_id % max(1,int(fps//2)) == 0:
                        recs = run_detection_img(frame)
                        for r in recs:
                            x1,y1,x2,y2 = [int(round(x)) for x in r["bbox"]]
                            clr = (0,0,255) if r["helmet_violation"] else (255,255,0)
                            cv2.rectangle(frame, (x1,y1),(x2,y2), clr, 2)
                    writer.write(frame)
            finally:
                cap.release(); writer.release()
            ts = int(time.time()*1000)
            json_path = OUTPUT_DIR / f"result_{ts}.json"
            csv_path = OUTPUT_DIR / f"results_{ts}.csv"
            # simple placeholders
            with open(json_path,"w") as jf: json.dump({"message":"video processed"}, jf)
            with open(csv_path,"w") as cf: cf.write("file,info\n")
            zipname = OUTPUT_DIR / f"results_{ts}.zip"
            with zipfile.ZipFile(zipname, "w") as zf:
                zf.write(out_vid, arcname=out_vid.name)
                zf.write(json_path, arcname=json_path.name)
                zf.write(csv_path, arcname=csv_path.name)
            return FileResponse(path=str(zipname), filename=zipname.name, media_type="application/zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: shutil.rmtree(tmp)
        except: pass

# For local debug (not used by Spaces)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info", workers=1)
