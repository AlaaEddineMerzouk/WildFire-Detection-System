from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="graduation.png",
    save=True
)

print("Inference fbdfb completed.")