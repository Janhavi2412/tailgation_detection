from ultralytics import YOLO
import os

MODEL_NAME = "yolo11n.pt"  # Pretrained on COCO
IMAGE_DIR = "/content/drive/MyDrive/Office_tailgating/dataset/train/images"
LABEL_DIR = "/content/drive/MyDrive/Office_tailgating/dataset/train/labels"
CONF_THRESHOLD = 0.3

os.makedirs(LABEL_DIR, exist_ok=True)
model = YOLO(MODEL_NAME)

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

for img in images:
    img_path = os.path.join(IMAGE_DIR, img)
    model.predict(img_path, conf=CONF_THRESHOLD, save_txt=True, project=LABEL_DIR, name="", exist_ok=True)

print(" Auto-labeling complete.")
