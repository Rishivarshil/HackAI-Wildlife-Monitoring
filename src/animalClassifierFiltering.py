import os
import csv
import subprocess
from datetime import datetime
from ultralytics import YOLO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on your images folder (save=False initially; we'll save selectively)
results = model(r"C:\Users\bhara\Downloads\Pictures", save=False, conf=0.25)

csv_path = "detections.csv"
csv_rows = []

# Process results
for result in results:
    print(f"\n📷 {result.path}")

    # Filter boxes to only those with confidence >= 0.5
    high_conf_boxes = [
        box for box in (result.boxes if result.boxes is not None else [])
        if float(box.conf[0]) >= 0.5
    ]

    if high_conf_boxes:
        # --- Get image datetime ---
        image_datetime = None
        if PIL_AVAILABLE:
            try:
                with Image.open(result.path) as img:
                    exif = img._getexif()
                    if exif:
                        # EXIF tag 36867 = DateTimeOriginal
                        raw = exif.get(36867) or exif.get(306)
                        if raw:
                            image_datetime = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
            except Exception:
                pass

        if image_datetime is None:
            # Fallback: file modification time
            mtime = os.path.getmtime(result.path)
            image_datetime = datetime.fromtimestamp(mtime)

        dt_str = image_datetime.strftime("%Y-%m-%d %H:%M:%S")

        for box in high_conf_boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"  ✓ {cls_name}: {conf:.2f} at {coords}")
            csv_rows.append({
                "image_path": result.path,
                "image_datetime": dt_str,
                "animal": cls_name,
                "confidence": round(conf, 4),
                "x1": round(coords[0], 1),
                "y1": round(coords[1], 1),
                "x2": round(coords[2], 1),
                "y2": round(coords[3], 1),
            })

        # Save annotated image
        result.save()
    else:
        print("  No detections above 0.5 — deleting input image")
        try:
            os.remove(result.path)
            print(f"  Deleted: {result.path}")
        except OSError as e:
            print(f"  Could not delete {result.path}: {e}")

# Write CSV
if csv_rows:
    fieldnames = ["image_path", "image_datetime", "animal", "confidence", "x1", "y1", "x2", "y2"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n📊 CSV saved to {csv_path} ({len(csv_rows)} rows)")

print("\n✅ Done. Results saved to runs/detect/predict")
