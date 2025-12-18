from ultralytics import YOLO
from utils.config import MODEL_DIR

class HelmetDetector:
    def __init__(self):
        self.model = YOLO(str(MODEL_DIR / "best.pt"))

    def check(self, frame, track):
        if track["class"] != "motorcycle":
            return None

        x1, y1, x2, y2 = map(int, track["bbox"])
        crop = frame[y1:y2, x1:x2]

        res = self.model(crop, verbose=False)
        if not res or not res[0].boxes:
            return None

        cls = int(res[0].boxes[0].cls[0])
        conf = float(res[0].boxes[0].conf[0])

        return {
            "helmet": cls == 0,
            "confidence": conf
        }
