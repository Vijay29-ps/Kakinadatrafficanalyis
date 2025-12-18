from ultralytics import YOLO
from utils.config import MODEL_DIR

class WeaponDetector:
    def __init__(self):
        self.model = YOLO(str(MODEL_DIR / "All_weapon .pt"))

    def check(self, frame, track):
        res = self.model(frame, verbose=False)
        if not res or not res[0].boxes:
            return None

        box = res[0].boxes[0]
        return {
            "weapon": True,
            "confidence": float(box.conf[0])
        }
