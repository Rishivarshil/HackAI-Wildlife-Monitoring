import os
import sys
import shutil
import cv2
from ultralytics import YOLO
import torch
# Allow the YOLO model structure to pass through PyTorch's security filter
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Input video path
mp4_input = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"

frames_dir = "frames"
output_dir = "demo_output"

# Clear and recreate working dirs
shutil.rmtree(frames_dir, ignore_errors=True)
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(frames_dir)
os.makedirs(output_dir)

# Extract frames using OpenCV
cap = cv2.VideoCapture(mp4_input)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"📽️ Input: {mp4_input} @ {fps}fps")
print("🖼️ Extracting frames...")
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg"), frame)
cap.release()
print(f"   {frame_idx} frames extracted")

# Run YOLO inference on extracted frames
results = model(frames_dir, save=False, conf=0.25)

# Process results
for result in results:
    filename = os.path.basename(result.path)
    print(f"\n📷 {filename}")

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
        # Save annotated frame (bounding box only, no label)
        annotated = result.plot(labels=False)
        cv2.imwrite(os.path.join(output_dir, filename), annotated)
    else:
        print("  No detections above 0.5 — keeping original frame")
        shutil.copy2(result.path, os.path.join(output_dir, filename))

# Stitch annotated frames back into MP4 using OpenCV
print("\n🎬 Stitching frames into MP4...")
frame_files = sorted(
    f for f in (os.path.join(output_dir, n) for n in os.listdir(output_dir))
    if f.endswith(".jpg") or f.endswith(".png")
)
if frame_files:
    first = cv2.imread(frame_files[0])
    h, w = first.shape[:2]
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fp in frame_files:
        frame = cv2.imread(fp)
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"   {len(frame_files)} frames written")

print("\n✅ Done. output.mp4 saved.")