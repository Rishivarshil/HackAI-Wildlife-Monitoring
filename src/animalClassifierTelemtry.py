import os
import csv
import time
import threading
from datetime import datetime
from pathlib import Path

import psutil
from ultralytics import YOLO

# Optional EXIF datetime
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =========================
# CONFIG (EDIT THESE)
# =========================
# IMPORTANT: On OSC/Linux, this must be a Linux path, e.g.:
# IMAGE_FOLDER = "/fs/scratch/PAS2136/image"
IMAGE_FOLDER = "/fs/scratch/PAS2136/image"  # <-- CHANGE THIS to your real image directory on OSC

MODEL_WEIGHTS = "yolov8n.pt"
CONF_MODEL = 0.25   # YOLO inference threshold
CONF_KEEP = 0.50    # keep boxes >= this confidence
SAMPLE_INTERVAL_SEC = 1.0

# Safer than deleting: set to True to MOVE images with no detections
MOVE_NO_DETECTIONS_INSTEAD_OF_DELETE = True

# If moving, they go here:
NO_DETECTIONS_DIRNAME = "no_detections"


# =========================
# CPU-ONLY ENFORCEMENT
# =========================
def enforce_cpu_only():
    # Prevent CUDA visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Ultralytics respects this too
    os.environ["ULTRALYTICS_DEVICE"] = "cpu"


# =========================
# Helpers
# =========================
def folder_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def try_get_image_datetime(image_path: str) -> datetime:
    """
    Best effort:
    - EXIF DateTimeOriginal (36867) or DateTime (306) if PIL available
    - file modification time fallback
    """
    image_datetime = None

    if PIL_AVAILABLE:
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    raw = exif.get(36867) or exif.get(306)
                    if raw:
                        image_datetime = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        except Exception:
            image_datetime = None

    if image_datetime is None:
        mtime = os.path.getmtime(image_path)
        image_datetime = datetime.fromtimestamp(mtime)

    return image_datetime


def get_disk_usage_path_for_output(output_dir: str) -> str:
    """
    Return a valid path for psutil.disk_usage():
    - On Linux: use '/'
    - On Windows: use drive root like 'C:\\'
    """
    if os.name == "nt":
        drive = os.path.splitdrive(os.path.abspath(output_dir))[0]
        return drive + "\\"
    return "/"


# =========================
# Resource Monitor (CPU/RAM/Disk + optional battery)
# =========================
class ResourceMonitor:
    def __init__(self, log_csv_path: str, output_dir: str, sample_interval_sec: float = 1.0):
        self.log_csv_path = log_csv_path
        self.sample_interval_sec = sample_interval_sec
        self.output_dir = output_dir

        self._stop_event = threading.Event()
        self._thread = None

        self.process = psutil.Process(os.getpid())
        self.start_ts = None

        # Peaks for summary
        self.peak_proc_rss_mb = 0.0
        self.peak_sys_mem_percent = 0.0
        self.peak_cpu_percent = 0.0

    def start(self):
        self.start_ts = time.time()

        with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "elapsed_s",
                    "cpu_percent_total",
                    "proc_rss_mb",
                    "sys_mem_percent",
                    "disk_used_gb",
                    "disk_free_gb",
                ],
            )
            writer.writeheader()

        # Prime CPU percent so first read isn't 0.0
        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        disk_path = get_disk_usage_path_for_output(self.output_dir)

        while not self._stop_event.is_set():
            now = datetime.now()
            elapsed = time.time() - self.start_ts if self.start_ts else 0.0

            cpu_total = psutil.cpu_percent(interval=None)

            rss_mb = self.process.memory_info().rss / (1024 * 1024)
            sys_mem_percent = psutil.virtual_memory().percent

            # Disk usage (won't crash on Linux)
            disk = psutil.disk_usage(disk_path)
            disk_used_gb = disk.used / (1024 ** 3)
            disk_free_gb = disk.free / (1024 ** 3)

            # Update peaks
            self.peak_proc_rss_mb = max(self.peak_proc_rss_mb, rss_mb)
            self.peak_sys_mem_percent = max(self.peak_sys_mem_percent, sys_mem_percent)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_total)

            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "elapsed_s",
                        "cpu_percent_total",
                        "proc_rss_mb",
                        "sys_mem_percent",
                        "disk_used_gb",
                        "disk_free_gb",
                    ],
                )
                writer.writerow(
                    {
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_s": round(elapsed, 3),
                        "cpu_percent_total": round(cpu_total, 2),
                        "proc_rss_mb": round(rss_mb, 2),
                        "sys_mem_percent": round(sys_mem_percent, 2),
                        "disk_used_gb": round(disk_used_gb, 3),
                        "disk_free_gb": round(disk_free_gb, 3),
                    }
                )

            time.sleep(self.sample_interval_sec)


# =========================
# Main
# =========================
def main():
    enforce_cpu_only()

    image_dir = Path(IMAGE_FOLDER)

    # Output directory relative to current working dir
    output_dir = Path(os.getcwd()) / "yolo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = output_dir / "detections.csv"
    resource_log_csv = output_dir / "resource_log_cpu_only.csv"

    print("🧱 CPU-only mode enforced (no CUDA).")
    print("📁 Image folder:", str(image_dir))
    print("📁 Output dir:", str(output_dir))
    print("📄 Detections CSV:", str(detections_csv))
    print("📈 Resource log CSV:", str(resource_log_csv))
    print("🧭 Current working directory:", os.getcwd())

    # Fail early if image folder is invalid
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(
            f"IMAGE_FOLDER does not exist on this machine: {image_dir}\n"
            f"Fix IMAGE_FOLDER to a valid Linux path on OSC, e.g. /fs/scratch/PAS2136/<your_folder>"
        )

    # Optional directory for moved no-detection images
    no_det_dir = output_dir / NO_DETECTIONS_DIRNAME
    if MOVE_NO_DETECTIONS_INSTEAD_OF_DELETE:
        no_det_dir.mkdir(parents=True, exist_ok=True)

    # Storage snapshots before
    before_input_bytes = folder_size_bytes(str(image_dir))
    before_output_bytes = folder_size_bytes(str(output_dir))

    # Start resource monitor
    monitor = ResourceMonitor(str(resource_log_csv), str(output_dir), sample_interval_sec=SAMPLE_INTERVAL_SEC)
    monitor.start()

    t0 = time.time()

    # Load model + infer (CPU-only)
    model = YOLO(MODEL_WEIGHTS)
    results = model(str(image_dir), save=False, conf=CONF_MODEL, device="cpu")

    csv_rows = []

    for result in results:
        print(f"\n📷 {result.path}")

        high_conf_boxes = [
            box for box in (result.boxes if result.boxes is not None else [])
            if float(box.conf[0]) >= CONF_KEEP
        ]

        if high_conf_boxes:
            image_dt = try_get_image_datetime(result.path)
            dt_str = image_dt.strftime("%Y-%m-%d %H:%M:%S")

            for box in high_conf_boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()

                print(f"  ✓ {cls_name}: {conf:.2f} at {coords}")

                csv_rows.append(
                    {
                        "image_path": result.path,
                        "image_datetime": dt_str,
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "x1": round(coords[0], 1),
                        "y1": round(coords[1], 1),
                        "x2": round(coords[2], 1),
                        "y2": round(coords[3], 1),
                    }
                )

            # Save annotated image (Ultralytics default: runs/detect/predict)
            result.save()

        else:
            src = Path(result.path)
            if MOVE_NO_DETECTIONS_INSTEAD_OF_DELETE:
                dst = no_det_dir / src.name
                print(f"  No detections >= {CONF_KEEP:.2f} — moving to {dst}")
                try:
                    src.rename(dst)
                except Exception as e:
                    print(f"  Could not move {src}: {e}")
            else:
                print(f"  No detections >= {CONF_KEEP:.2f} — deleting")
                try:
                    os.remove(result.path)
                    print(f"  Deleted: {result.path}")
                except OSError as e:
                    print(f"  Could not delete {result.path}: {e}")

    # Write detections CSV
    if csv_rows:
        fieldnames = ["image_path", "image_datetime", "class", "confidence", "x1", "y1", "x2", "y2"]
        with open(detections_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n📊 Detections CSV saved: {detections_csv} ({len(csv_rows)} rows)")
    else:
        print("\n📊 No qualifying detections; detections CSV not written.")

    t1 = time.time()

    # Stop monitor
    monitor.stop()

    # Storage snapshots after
    after_input_bytes = folder_size_bytes(str(image_dir))
    after_output_bytes = folder_size_bytes(str(output_dir))

    print("\n=========================")
    print("✅ RUN SUMMARY (CPU ONLY)")
    print("=========================")
    print(f"⏱ Runtime (s): {round(t1 - t0, 2)}")
    print(f"🧠 Peak process RAM (MB): {round(monitor.peak_proc_rss_mb, 2)}")
    print(f"🧠 Peak system memory (%): {round(monitor.peak_sys_mem_percent, 2)}")
    print(f"🖥 Peak CPU (% total): {round(monitor.peak_cpu_percent, 2)}")

    print("\n💾 STORAGE (bytes)")
    print(f"Input folder before:  {before_input_bytes}")
    print(f"Input folder after:   {after_input_bytes}")
    print(f"Output folder before: {before_output_bytes}")
    print(f"Output folder after:  {after_output_bytes}")
    print(f"Δ Input bytes:        {after_input_bytes - before_input_bytes}")
    print(f"Δ Output bytes:       {after_output_bytes - before_output_bytes}")

    print("\n📈 Resource log CSV:", str(resource_log_csv))
    print("📄 Detections CSV:", str(detections_csv))
    print("\n✅ Done. Annotated images saved under runs/detect/predict (Ultralytics default).")


if __name__ == "__main__":
    main()
