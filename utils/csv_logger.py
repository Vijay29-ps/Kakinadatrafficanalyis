import csv
from datetime import datetime

CSV_HEADER = [
    "detection_uuid",
    "file_name",
    "camera_id",
    "frame_id",
    "timestamp",

    "object_id",
    "lane",
    "speed_kmph",
    "overspeed",

    "label",
    "confidence",
    "color",

    "helmet_violation",
    "weapon_detected",

    "fight_detected",
    "fight_confidence",
    "fight_participants",
    "fight_severity",
    "fight_duration_sec",

    "violation_type",

    "anpr_plate",
    "anpr_confidence",
    "hotlist_match",

    "violation_image",
    "meta",
]


def init_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)


def log_event(
    path,
    *,
    detection_uuid,
    file_name,
    camera_id,
    frame_id,
    object_id,
    label,
    confidence,
    color,

    lane=None,
    speed_kmph=None,
    overspeed=False,

    helmet_violation=False,
    weapon_detected=False,

    fight_detected=False,
    fight_confidence=None,
    fight_participants=None,
    fight_severity=None,
    fight_duration_sec=None,

    violation_type=None,

    anpr_plate=None,
    anpr_confidence=None,
    hotlist_match=False,

    violation_image=None,
    meta=None,
    timestamp=None,
):
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    if meta is None:
        meta = {}

    if fight_participants is None:
        fight_participants = []

    row = [
        detection_uuid,
        file_name,
        camera_id,
        frame_id,
        timestamp,

        object_id,
        lane,
        speed_kmph,
        overspeed,

        label,
        round(float(confidence), 4),
        color,

        helmet_violation,
        weapon_detected,

        fight_detected,
        fight_confidence,
        ",".join(map(str, fight_participants)),
        fight_severity,
        fight_duration_sec,

        violation_type,

        anpr_plate,
        anpr_confidence,
        hotlist_match,

        violation_image,
        meta,
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
