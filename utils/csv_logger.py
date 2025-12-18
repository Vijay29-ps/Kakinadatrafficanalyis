import csv

def init_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "event", "track_id", "confidence", "details"])

def log_event(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
