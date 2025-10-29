from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/content/drive/MyDrive/Office_tailgating/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="tailgating_model",
    project="/content/drive/MyDrive/Office_tailgating/runs"
)
