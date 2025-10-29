import os, shutil, random

BASE_DIR = "/content/drive/MyDrive/Office_tailgating/dataset"
train_img = os.path.join(BASE_DIR, "train/images")
val_img = os.path.join(BASE_DIR, "val/images")
train_lbl = os.path.join(BASE_DIR, "train/labels")
val_lbl = os.path.join(BASE_DIR, "val/labels")

os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

train_files = [f for f in os.listdir(train_img) if f.endswith(".jpg")]
random.shuffle(train_files)
val_split = int(0.2 * len(train_files))

for f in train_files[:val_split]:
    shutil.move(os.path.join(train_img, f), val_img)
    lbl = f.replace(".jpg", ".txt")
    if os.path.exists(os.path.join(train_lbl, lbl)):
        shutil.move(os.path.join(train_lbl, lbl), val_lbl)

print("Dataset split 80/20 (train/val) complete.")
