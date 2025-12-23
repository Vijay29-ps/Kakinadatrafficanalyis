import json
from datetime import datetime

def init_json(path):
    payload = {
        "meta": {
            "schema": "traffic-ai-v1",
            "created_at": datetime.utcnow().isoformat(),
            "total_events": 0,
        },
        "events": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def log_json_event(
    path,
    *,
    detection_uuid,
    file_name,
    camera_id,
    frame_id,
    object_id,
    lane,
    speed_kmph,
    overspeed,
    label,
    confidence,
    color,
    helmet_violation,
    weapon_detected,
    fight_detected,
    fight_confidence,
    fight_participants,
    fight_severity,
    fight_duration_sec,
    violation_type,
    anpr_plate,
    anpr_confidence,
    hotlist_match,
    violation_image,
    meta,
    timestamp=None,
):
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    event = {
        "detection_uuid": detection_uuid,
        "file_name": file_name,
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp": timestamp,

        "object_id": object_id,
        "lane": lane,
        "speed_kmph": speed_kmph,
        "overspeed": overspeed,

        "label": label,
        "confidence": round(float(confidence), 4),
        "color": color,

        "helmet_violation": helmet_violation,
        "weapon_detected": weapon_detected,

        "fight": {
            "detected": fight_detected,
            "confidence": fight_confidence,
            "participants": fight_participants or [],
            "severity": fight_severity,
            "duration_sec": fight_duration_sec,
        },

        "violation_type": violation_type,

        "anpr": {
            "plate": anpr_plate,
            "confidence": anpr_confidence,
            "hotlist_match": hotlist_match,
        },

        "violation_image": violation_image,
        "meta": meta or {},
    }

    with open(path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["events"].append(event)
        data["meta"]["total_events"] += 1
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
