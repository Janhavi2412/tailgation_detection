import cv2
import os
from tqdm import tqdm

VIDEO_DIR = "/content/drive/MyDrive/Office_tailgating/raw_videos"
OUTPUT_DIR = "/content/drive/MyDrive/Office_tailgating/dataset/train/images"
FRAME_RATE = 2  # frames per second

os.makedirs(OUTPUT_DIR, exist_ok=True)

for video_name in os.listdir(VIDEO_DIR):
    if not video_name.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / FRAME_RATE)
    frame_count, saved_count = 0, 0

    print(f" Extracting from {video_name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_name = f"{os.path.splitext(video_name)[0]}_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f" Extracted {saved_count} frames from {video_name}")

print(" Frame extraction complete.")
