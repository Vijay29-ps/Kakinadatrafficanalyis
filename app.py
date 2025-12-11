# main.py - Single-file FastAPI server with integrated YOLO processing (video/image -> annotated + deduped outputs)
# Usage:
#   pip install fastapi "uvicorn[standard]" python-multipart ultralytics opencv-python-headless numpy webcolors aiofiles
#   uvicorn main:app --host 0.0.0.0 --port 7860
#
# Endpoint:
#   POST /process  (multipart file upload field name "file", optional form field "camera_id")
#   Returns: application/zip containing annotated output + results CSV + result JSON + summary JSON

import os
import io
import time
import shutil
import zipfile
import tempfile
import math
import uuid
import csv
import json
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import webcolors

# ultralytics YOLO
from ultralytics import YOLO

# ------------------ CONFIG (tweak) ------------------
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# models (you can place your helmet model here)
GENERAL_MODEL_PATH = "./models/yolov8n.pt"   # if not present, ultralytics will download from the internet
HELMET_MODEL_CANDIDATES = [Path("./models/best.pt"), Path("/app/models/best.pt"), Path("https://huggingface.co/spaces/psv12/Kakinadatrafficanalyis/resolve/main/best.pt")]

# detection/tracking/speed settings
PX_PER_METER = 20.0
OVERSPEED_THRESHOLD_KMPH = 40.0
MAX_MATCH_DISTANCE_PX = 120
TRACK_HISTORY_LEN = 6
SPEED_SMOOTH_ALPHA = 0.4

NUM_LANES = 3
DENSITY_THRESHOLDS = {"none": 0, "low": 5, "medium": 15}

KEEP_BY = "best_conf"  # dedupe mode

CSV_HEADERS = [
    "file_name","camera_id","frame_id","vehicle_id",
    "vehicle_type","conf","color_hex","color_name","lane","speed_kmph","overspeed","helmet_violation",
    "anpr_plate","anpr_confidence","hotlist_match","weapon_detected",
    "violation_type","violation_image","timestamp","meta","crowd_count","density_level"
]

# CORS - add your frontend origin(s)
ALLOWED_ORIGINS = [
    "https://video-ten-silk.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# ------------------ HELPERS (from your Colab) ------------------

def make_vehicle_id():
    return str(uuid.uuid4())

def save_violation_crop(img, bbox, prefix="violation"):
    x1,y1,x2,y2 = [int(round(x)) for x in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    fname = f"{prefix}_{int(time.time()*1000)}_{make_vehicle_id()}.jpg"
    outpath = OUTPUT_DIR / fname
    cv2.imwrite(str(outpath), crop)
    return str(outpath)

def mean_color_hex(img, bbox, center_ratio=0.6, min_sat=30, min_val=40, k=3, max_pixels=20000):
    x1,y1,x2,y2 = [int(round(x)) for x in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x2 <= x1 or y2 <= y1:
        return "#000000"
    bw = x2 - x1
    bh = y2 - y1
    cx1 = x1 + int((1 - center_ratio) / 2.0 * bw)
    cy1 = y1 + int((1 - center_ratio) / 2.0 * bh)
    cx2 = x2 - int((1 - center_ratio) / 2.0 * bw)
    cy2 = y2 - int((1 - center_ratio) / 2.0 * bh)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "#000000"
    ch, cw = crop.shape[:2]
    total_pixels = ch * cw
    if total_pixels > max_pixels:
        scale = (max_pixels / float(total_pixels)) ** 0.5
        crop = cv2.resize(crop, (max(1,int(cw*scale)), max(1,int(ch*scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mask = (s_ch >= min_sat) & (v_ch >= min_val)
    # mask selection
    valid = crop[mask]
    if valid.shape[0] < 50:
        mask2 = (s_ch >= max(10, min_sat//2)) & (v_ch >= max(20, min_val//2))
        valid = crop[mask2]
    if valid.shape[0] < 30:
        # fallback average central area
        fcrop = img[max(0, cy1 + int(0.2*(cy2-cy1))):max(0, cy2 - int(0.2*(cy2-cy1))),
                    max(0, cx1 + int(0.2*(cx2-cx1))):max(0, cx2 - int(0.2*(cx2-cx1)))]
        if fcrop.size == 0:
            return "#000000"
        avg = fcrop.mean(axis=(0,1))
        r,g,b = int(avg[2]), int(avg[1]), int(avg[0])
        return "#{:02X}{:02X}{:02X}".format(r,g,b)
    samples = valid.reshape(-1,3).astype(np.float32)
    if samples.shape[0] > max_pixels:
        idx = np.random.choice(samples.shape[0], max_pixels, replace=False)
        samples = samples[idx]
    K = min(k, max(1, samples.shape[0] // 20))
    if K < 1:
        avg = samples.mean(axis=0)
        b,g,r = int(avg[0]), int(avg[1]), int(avg[2])
        return "#{:02X}{:02X}{:02X}".format(r,g,b)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(samples, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=K)
    dominant_idx = int(np.argmax(counts))
    dominant_color = centers[dominant_idx]
    b, g, r = [int(max(0, min(255, c))) for c in dominant_color]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def hex_to_nearest_color_name(hex_color):
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return str(hex_color)
    try:
        return webcolors.hex_to_name(hex_color, spec='css3')
    except Exception:
        pass
    try:
        mapping = getattr(webcolors, "CSS3_NAMES_TO_HEX", None)
        if mapping is None or not isinstance(mapping, dict):
            mapping = {
                "black":"#000000","white":"#FFFFFF","red":"#FF0000","green":"#008000",
                "blue":"#0000FF","gray":"#808080","silver":"#C0C0C0","yellow":"#FFFF00",
                "orange":"#FFA500","brown":"#A52A2A"
            }
        r,g,b = tuple(int(hex_color[i:i+2], 16) for i in (1,3,5))
        min_dist = None
        min_name = None
        for name, hexv in mapping.items():
            if not isinstance(hexv, str) or not hexv.startswith("#"):
                continue
            rr,gg,bb = tuple(int(hexv[i:i+2], 16) for i in (1,3,5))
            d = (r-rr)**2 + (g-gg)**2 + (b-bb)**2
            if min_dist is None or d < min_dist:
                min_dist = d
                min_name = name
        if min_name:
            return min_name
    except Exception:
        pass
    return hex_color

def centroid_from_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def euclidean(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def match_detections_to_tracks(detections_centroids, existing_tracks, max_dist=MAX_MATCH_DISTANCE_PX):
    matches = []
    used_tracks = set()
    for i, det_c in enumerate(detections_centroids):
        best_tid = None
        best_d = None
        for tid, t in existing_tracks.items():
            if tid in used_tracks:
                continue
            d = euclidean(det_c, t['centroid'])
            if d <= max_dist and (best_d is None or d < best_d):
                best_d = d
                best_tid = tid
        if best_tid is not None:
            matches.append((i, best_tid))
            used_tracks.add(best_tid)
        else:
            matches.append((i, None))
    return matches

def compute_speed_from_history(history_deque, fps, px_per_meter=PX_PER_METER):
    if len(history_deque) < 2:
        return 0.0
    old_frame, old_cent = history_deque[0]
    new_frame, new_cent = history_deque[-1]
    frame_delta = new_frame - old_frame
    if frame_delta <= 0 or fps <= 0:
        return 0.0
    dist_px = euclidean(old_cent, new_cent)
    dist_m = dist_px / px_per_meter if px_per_meter and px_per_meter>0 else 0.0
    time_s = frame_delta / fps
    if time_s == 0 or dist_m == 0:
        return 0.0
    speed_m_s = dist_m / time_s
    return speed_m_s * 3.6

def density_from_count(count):
    if count <= DENSITY_THRESHOLDS["none"]:
        return "none"
    if count <= DENSITY_THRESHOLDS["low"]:
        return "low"
    if count <= DENSITY_THRESHOLDS["medium"]:
        return "medium"
    return "high"

def build_record(file_name, camera_id, frame_id, vehicle_id, vehicle_type, conf, color_hex, color_name,
                 lane="", speed_kmph=None, overspeed=False, helmet_violation=False,
                 anpr_plate="", anpr_confidence=0.0, hotlist_match=False,
                 weapon_detected=False, violation_type="", violation_image="", meta=None,
                 crowd_count=0, density_level="none"):
    ts = datetime.utcnow().isoformat() + "Z"
    record = {
        "file_name": file_name,
        "camera_id": camera_id,
        "frame_id": frame_id,
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle_type,
        "conf": float(conf),
        "color_hex": color_hex,
        "color_name": color_name,
        "lane": lane,
        "speed_kmph": round(float(speed_kmph),3) if speed_kmph is not None else None,
        "overspeed": bool(overspeed),
        "helmet_violation": bool(helmet_violation),
        "anpr_plate": anpr_plate,
        "anpr_confidence": float(anpr_confidence),
        "hotlist_match": bool(hotlist_match),
        "weapon_detected": bool(weapon_detected),
        "violation_type": violation_type,
        "violation_image": violation_image,
        "timestamp": ts,
        "meta": meta or {},
        "crowd_count": int(crowd_count),
        "density_level": density_level
    }
    return record

# ------------------ MODEL LOADING ------------------

def load_models():
    # ensure models dir exists
    Path("./models").mkdir(parents=True, exist_ok=True)
    gen_path = GENERAL_MODEL_PATH
    if not Path(gen_path).exists():
        # fall back to default name (ultralytics will download)
        gen_path = "yolov8n.pt"
    print("Loading general model:", gen_path)
    general = YOLO(gen_path)
    helmet = None
    for cand in HELMET_MODEL_CANDIDATES:
        if cand.exists():
            try:
                print("Found helmet model at", str(cand), "— loading automatically.")
                helmet = YOLO(str(cand))
                print("✔ Helmet model loaded:", str(cand))
                break
            except Exception as e:
                print("Helmet model load failed for", cand, ":", e)
    if helmet is None:
        print("No helmet model auto-found. Helmet detection disabled.")
    return general, helmet

GENERAL_MODEL, HELMET_MODEL = load_models()

# ------------------ PROCESSING FUNCTIONS ------------------

def process_image(path, camera_id="camera_0"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Cannot read image")
    ts = time.strftime("%Y%m%d_%H%M%S")
    file_name = Path(path).name
    records = []
    res = GENERAL_MODEL(img)[0]
    crowd_count = sum(1 for b in res.boxes if GENERAL_MODEL.names[int(b.cls[0])].lower()=="person")
    density = density_from_count(crow_count)
    helmet_boxes = []
    if HELMET_MODEL is not None:
        try:
            hr = HELMET_MODEL(img)[0]
            for hb in hr.boxes:
                helmet_boxes.append({"bbox": hb.xyxy[0].cpu().numpy().tolist(), "cls": int(hb.cls[0]), "conf": float(hb.conf[0])})
        except Exception as e:
            print("Helmet inference failed on image:", e)
            helmet_boxes = []
    for b in res.boxes:
        bbox = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        vehicle_type = GENERAL_MODEL.names[cls]
        color_hex = mean_color_hex(img, bbox)
        color_name = hex_to_nearest_color_name(color_hex)
        helmet_violation = False
        if helmet_boxes:
            x1,y1,x2,y2 = bbox
            for hb in helmet_boxes:
                hx1,hy1,hx2,hy2 = hb["bbox"]
                inter_x = max(0, min(x2,hx2) - max(x1,hx1))
                inter_y = max(0, min(y2,hy2) - max(y1,hy1))
                if inter_x>0 and inter_y>0:
                    helmet_violation = (hb["cls"]==1)
                    break
        cx,cy = centroid_from_bbox(bbox)
        img_w = img.shape[1]
        lane_idx = max(1, min(NUM_LANES, int(cx // (img_w/NUM_LANES)) + 1))
        vehicle_id = f"image_{make_vehicle_id()}"
        violation_type = "helmet" if helmet_violation else ""
        violation_image = save_violation_crop(img, bbox, prefix="violation") if violation_type else ""
        rec = build_record(file_name, camera_id, 0, vehicle_id, vehicle_type, conf, color_hex, color_name,
                           lane=f"lane_{lane_idx}", speed_kmph=None, overspeed=False,
                           helmet_violation=helmet_violation, violation_type=violation_type,
                           violation_image=violation_image, meta={"detector":"yolov8","source":"image"},
                           crowd_count=crowd_count, density_level=density)
        records.append(rec)
        # annotate
        color_draw = (0,0,255) if violation_type else (255,255,0)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color_draw, 2)
        label_txt = vehicle_type if not violation_type else f"{vehicle_type} | {violation_type}"
        cv2.putText(img, f"{label_txt} {conf:.2f}", (int(bbox[0]), int(max(0,bbox[1])-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    out_img = OUTPUT_DIR / f"annotated_{ts}.jpg"
    cv2.imwrite(str(out_img), img)
    csv_path = OUTPUT_DIR / f"results_{ts}.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    json_path = OUTPUT_DIR / f"result_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(records, jf, indent=2, default=str)
    counts = defaultdict(int)
    for r in records:
        counts[r['vehicle_type']] += 1
    summary = {"file": file_name, "vehicle_count": sum(counts.values()), "vehicle_counts_by_type": dict(counts)}
    summary_path = OUTPUT_DIR / f"summary_{ts}.json"
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    return str(out_img), str(json_path), str(csv_path), str(summary_path)

def process_video(path, camera_id="camera_0"):
    TRACKS = {}
    NEXT_TRACK_ID = 1
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_vid_path = OUTPUT_DIR / f"annotated_{ts}.mp4"
    writer = cv2.VideoWriter(str(out_vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    all_records = []
    frame_id = 0
    print("Processing video FPS:", fps, "PX_PER_METER:", PX_PER_METER)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            res = GENERAL_MODEL(frame)[0]
            crowd_count = sum(1 for b in res.boxes if GENERAL_MODEL.names[int(b.cls[0])].lower()=="person")
            density = density_from_count(crowd_count)
            helm_boxes = []
            if HELMET_MODEL is not None:
                try:
                    hr = HELMET_MODEL(frame)[0]
                    for hb in hr.boxes:
                        helm_boxes.append({"bbox": hb.xyxy[0].cpu().numpy().tolist(), "cls": int(hb.cls[0]), "conf": float(hb.conf[0])})
                except Exception as e:
                    if frame_id == 1:
                        print("Helmet inference failed:", e)
                    helm_boxes = []
            detections = []
            for b in res.boxes:
                bbox = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                vehicle_type = GENERAL_MODEL.names[cls]
                cx, cy = centroid_from_bbox(bbox)
                detections.append({"bbox": bbox, "centroid": (cx,cy), "vehicle_type": vehicle_type, "conf": conf})
            det_centroids = [d['centroid'] for d in detections]
            matches = match_detections_to_tracks(det_centroids, TRACKS, max_dist=MAX_MATCH_DISTANCE_PX)
            det_to_tid = {}
            for det_idx, tid in matches:
                det_to_tid[det_idx] = tid
            # update/create tracks
            for i, d in enumerate(detections):
                tid = det_to_tid.get(i)
                if tid is None:
                    tid = NEXT_TRACK_ID
                    NEXT_TRACK_ID += 1
                    history = deque(maxlen=TRACK_HISTORY_LEN)
                    history.append((frame_id, d['centroid']))
                    TRACKS[tid] = {"centroid": d['centroid'], "last_frame": frame_id, "history": history, "speed": 0.0}
                else:
                    track = TRACKS[tid]
                    track['history'].append((frame_id, d['centroid']))
                    speed = compute_speed_from_history(track['history'], fps, px_per_meter=PX_PER_METER)
                    track['speed'] = (1.0 - SPEED_SMOOTH_ALPHA) * track.get('speed', 0.0) + SPEED_SMOOTH_ALPHA * speed
                    track['centroid'] = d['centroid']
                    track['last_frame'] = frame_id
            # clear stale tracks
            stale = [tid for tid,t in TRACKS.items() if (frame_id - t['last_frame']) > (fps * 6)]
            for tid in stale:
                del TRACKS[tid]
            # build records for this frame
            for i,d in enumerate(detections):
                tid = det_to_tid.get(i)
                if tid is None:
                    tid = NEXT_TRACK_ID
                    NEXT_TRACK_ID += 1
                    TRACKS[tid] = {"centroid": d['centroid'], "last_frame": frame_id, "history": deque([(frame_id,d['centroid'])], maxlen=TRACK_HISTORY_LEN), "speed": 0.0}
                track = TRACKS[tid]
                speed_kmph = track.get('speed', 0.0)
                overspeed = speed_kmph > OVERSPEED_THRESHOLD_KMPH
                helmet_violation = False
                if helm_boxes:
                    x1,y1,x2,y2 = d['bbox']
                    for hb in helm_boxes:
                        hx1,hy1,hx2,hy2 = hb['bbox']
                        inter_x = max(0, min(x2,hx2) - max(x1,hx1))
                        inter_y = max(0, min(y2,hy2) - max(y1,hy1))
                        if inter_x>0 and inter_y>0:
                            helmet_violation = (hb['cls']==1)
                            break
                violation_type = ""
                if helmet_violation and overspeed:
                    violation_type = "helmet,overspeed"
                elif helmet_violation:
                    violation_type = "helmet"
                elif overspeed:
                    violation_type = "overspeed"
                color_hex = mean_color_hex(frame, d['bbox'])
                color_name = hex_to_nearest_color_name(color_hex)
                cx, cy = d['centroid']
                lane_idx = max(1, min(NUM_LANES, int(cx // (w/NUM_LANES)) + 1))
                violation_image = save_violation_crop(frame, d['bbox'], prefix="violation") if violation_type else ""
                vehicle_id = f"track_{tid}"
                rec = build_record(
                    file_name=Path(path).name,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    vehicle_id=vehicle_id,
                    vehicle_type=d['vehicle_type'],
                    conf=d['conf'],
                    color_hex=color_hex,
                    color_name=color_name,
                    lane=f"lane_{lane_idx}",
                    speed_kmph=speed_kmph,
                    overspeed=overspeed,
                    helmet_violation=helmet_violation,
                    violation_type=violation_type,
                    violation_image=violation_image,
                    meta={"detector":"yolov8","source":"video","track_id": tid},
                    crowd_count=crowd_count,
                    density_level=density
                )
                all_records.append(rec)
                # annotate
                color_draw = (0,0,255) if violation_type else (255,255,0)
                tl = (int(d['bbox'][0]), int(d['bbox'][1]))
                br = (int(d['bbox'][2]), int(d['bbox'][3]))
                cv2.rectangle(frame, tl, br, color_draw, 2)
                label_txt = d['vehicle_type'] if not violation_type else f"{d['vehicle_type']} | {violation_type}"
                cv2.putText(frame, f"{label_txt} {speed_kmph:.1f}km/h", (tl[0], max(0, tl[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    # dedupe by vehicle_id (track)
    dedup_map = {}
    for r in all_records:
        vid = r['vehicle_id']
        if vid not in dedup_map:
            dedup_map[vid] = r
        else:
            if KEEP_BY == "best_conf":
                if r['conf'] > dedup_map[vid]['conf']:
                    dedup_map[vid] = r
            else:
                pass
    deduped_records = list(dedup_map.values())
    csv_path = OUTPUT_DIR / f"results_{ts}.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in deduped_records:
            writer.writerow(r)
    json_path = OUTPUT_DIR / f"result_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(deduped_records, jf, indent=2, default=str)
    counts = defaultdict(int)
    for r in deduped_records:
        counts[r['vehicle_type']] += 1
    summary = {"file": Path(path).name, "vehicle_count": sum(counts.values()), "vehicle_counts_by_type": dict(counts)}
    summary_path = OUTPUT_DIR / f"summary_{ts}.json"
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    return str(out_vid_path), str(json_path), str(csv_path), str(summary_path)

# ------------------ FASTAPI APP ------------------

app = FastAPI(title="Helmet+Vehicle Processor (single-file)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "general_model": getattr(GENERAL_MODEL, "model", "yolo").__class__.__name__ if GENERAL_MODEL else None, "helmet_loaded": HELMET_MODEL is not None}

@app.post("/process")
async def process(file: UploadFile = File(...), camera_id: str = Form("camera_0")):
    # save uploaded file to a temp dir
    suffix = Path(file.filename).suffix.lower()
    tmpdir = Path(tempfile.mkdtemp())
    try:
        input_path = tmpdir / file.filename
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        is_image = suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        # process
        if is_image:
            out_img, out_json, out_csv, out_summary = process_image(str(input_path), camera_id=camera_id)
            files_to_zip = [out_img, out_json, out_csv, out_summary]
        else:
            out_vid, out_json, out_csv, out_summary = process_video(str(input_path), camera_id=camera_id)
            files_to_zip = [out_vid, out_json, out_csv, out_summary]
        # make zip
        zip_name = OUTPUT_DIR / f"results_{int(time.time()*1000)}.zip"
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files_to_zip:
                if p and Path(p).exists():
                    zf.write(p, arcname=Path(p).name)
        # return zip
        return FileResponse(path=str(zip_name), filename=Path(zip_name).name, media_type="application/zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup tempdir
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# ------------------ END ------------------
