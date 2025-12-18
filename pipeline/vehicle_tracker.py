from ultralytics import YOLO
from utils.config import MODEL_DIR

VEHICLE_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class VehicleTracker:
    def __init__(self):
        self.model = YOLO(str(MODEL_DIR / "yolov8n.pt"))

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        tracks = []

        if not results or results[0].boxes is None:
            return tracks

        for b in results[0].boxes:
            cid = int(b.cls[0])
            if cid not in VEHICLE_CLASSES:
                continue

            tracks.append({
                "track_id": int(b.id[0]) if b.id is not None else -1,
                "class": VEHICLE_CLASSES[cid],
                "bbox": b.xyxy[0].cpu().numpy().tolist()
            })

        return tracks
