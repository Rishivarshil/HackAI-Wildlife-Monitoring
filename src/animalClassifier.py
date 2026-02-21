import os
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on your images folder (save=False initially; we'll save selectively)
results = model(r"C:\Users\bhara\Downloads\Pictures", save=False, conf=0.25)

# Process results
for result in results:
    print(f"\n📷 {result.path}")

    # Filter boxes to only those with confidence > 0.5
    high_conf_boxes = [
        box for box in (result.boxes if result.boxes is not None else [])
        if float(box.conf[0]) > 0.5
    ]

    if high_conf_boxes:
        for box in high_conf_boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"  ✓ {cls_name}: {conf:.2f} at {coords}")
        # Save only images with qualifying detections
        result.save()
    else:
        print("  No detections above 0.5 — deleting input image")
        try:
            os.remove(result.path)
            print(f"  Deleted: {result.path}")
        except OSError as e:
            print(f"  Could not delete {result.path}: {e}")

print("\n✅ Done. Results saved to runs/detect/predict")
