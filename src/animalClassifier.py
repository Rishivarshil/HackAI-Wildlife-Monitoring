import os
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Path to images
image_folder = r"C:\Users\bhara\Downloads\Pictures"

# Run inference (lower threshold here so we can manually filter)
results = model(image_folder, save=False, conf=0.25)

for result in results:
    print(f"\n📷 {result.path}")

    if result.boxes is None or len(result.boxes) == 0:
        print("  No detections — deleting image")
        os.remove(result.path)
        continue

    # Keep only boxes with confidence > 0.5
    keep_indices = [
        i for i, box in enumerate(result.boxes)
        if float(box.conf[0]) > 0.5
    ]

    if len(keep_indices) > 0:
        for i in keep_indices:
            box = result.boxes[i]
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"  ✓ {cls_name}: {conf:.2f} at {coords}")

        # Save annotated image
        result.save()

    else:
        print("  No detections above 0.5 — deleting image")
        try:
            os.remove(result.path)
            print(f"  Deleted: {result.path}")
        except OSError as e:
            print(f"  Could not delete {result.path}: {e}")

print("\n✅ Done. Results saved to runs/detect/predict")
