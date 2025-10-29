import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import supervision as sv


# Paths & Configurations

VIDEO_PATH = "/content/drive/MyDrive/Office_tailgating/test_videos/test1.mp4"
OUTPUT_PATH = "/content/drive/MyDrive/Office_tailgating/output/test1_tailgating_output.mp4"
LOG_PATH = "/content/drive/MyDrive/Office_tailgating/output/tailgating_events.csv"
MODEL_PATH = "/content/drive/MyDrive/Office_tailgating/runs/tailgating_model/weights/best.pt"

# Load trained YOLO model
model = YOLO(MODEL_PATH)

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()


# Define Entry Line (door zone)

line_start = sv.Point(200, 400)  # adjust based on your door position
line_end   = sv.Point(800, 400)
line_zone = sv.LineZone(start=line_start, end=line_end)

line_annotator = sv.LineZoneAnnotator(thickness=4)
box_annotator = sv.BoxAnnotator(thickness=2)


# Video setup

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))


# Tailgating detection variables

tailgating_log = []
frame_count = 0
cross_log = {}  # {person_id: timestamp_of_crossing}
window_sec = 3  # if 2+ people cross within 3s → tailgating


# Main Processing Loop

print(" Starting tailgating detection...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / fps

    # Run YOLO detection + tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # Convert detections
    detections = sv.Detections.from_ultralytics(results[0])
    detections = detections[detections.class_id == 0]  # only 'person'

    if detections.tracker_id is None:
        out.write(frame)
        continue

    # Line crossing check
    triggered = line_zone.trigger(detections=detections)

    # Draw boxes and line
    labels = [f"ID {tracker_id}" for tracker_id in detections.tracker_id]
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = line_annotator.annotate(annotated, line_counter=line_zone)

    # Tailgating logic
    if triggered:
        for _, _, _, _, cls_id, track_id in detections:
            if cls_id == 0 and track_id is not None:
                cross_log[track_id] = current_time

        # Find all people who crossed within the last few seconds
        recent = [pid for pid, t in cross_log.items() if current_time - t <= window_sec]
        if len(set(recent)) > 1:
            cv2.putText(annotated, " TAILGATING DETECTED", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            tailgating_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame": frame_count,
                "persons_crossed": len(set(recent)),
                "event": "Tailgating Detected"
            })

    out.write(annotated)

cap.release()
out.release()

# Save logs
pd.DataFrame(tailgating_log).to_csv(LOG_PATH, index=False)
print(f" Processing complete. Output saved to: {OUTPUT_PATH}")
print(f" Log file: {LOG_PATH}")
