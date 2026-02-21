# EdgeGuard Wildlife Monitoring System

EdgeGuard is a **low-power, multimodal edge AI pipeline** for wildlife monitoring deployed on a Raspberry Pi.

It performs:

1. **Animal detection** using YOLOv8  
2. **Species verification** using BioCLIP  
3. **Conditional audio health classification** using a CNN on spectrograms  
4. **Event logging and media storage** (JPG, MP4, CSV)  
5. **Edge vs Desktop benchmarking**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Decision Logic](#decision-logic)
- [Outputs](#outputs)
- [CSV Output Schema](#csv-output-schema)
- [Full Pipeline Example](#full-pipeline-example)
- [Benchmarking](#benchmarking)
- [Energy Optimization](#energy-optimization)
- [Repository Structure](#repository-structure)
- [Success Metrics](#success-metrics)
- [Non-Goals](#non-goals)
- [Long-Term Vision](#long-term-vision)

---

# Overview

Traditional camera traps require:

- Manual review  
- Centralized cloud processing  
- High transmission costs  
- No real-time alerting  

**EdgeGuard moves intelligence directly to the edge device.**

### Key Characteristics

- Event-driven (PIR triggered)
- No continuous inference
- No required cloud connection
- Local SD card storage
- Quantized models for efficiency
- Measurable latency + accuracy comparison

> [!IMPORTANT]
> All inference runs locally. No cloud dependency is required.

---

# System Architecture

## Hardware

- Raspberry Pi 4 or 5
- PIR motion sensor
- USB camera
- USB microphone
- MicroSD card (32GB+ recommended)
- Optional battery pack

---

## Software Stack

- Python 3.9+
- PyTorch
- Ultralytics YOLOv8
- open_clip (BioCLIP support)
- OpenCV
- Librosa
- NumPy
- Matplotlib

---

# Installation

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python pillow numpy
pip install open_clip_torch
pip install librosa matplotlib
```

---

# Pipeline Workflow

## Step 1: Motion Trigger

When PIR detects motion:

1. Capture image  
2. Record 5–10 second audio clip  
3. Launch inference  
4. Return to idle  

> [!TIP]
> Audio inference only runs if deer is detected in the image to conserve energy.

---

## Step 2: Object Detection (YOLOv8 Lite)

```bash
python edgeguard_pipeline.py \
  --weights yolov8n.pt \
  --image ./input/frame.jpg \
  --conf 0.25
```

Output:

- Bounding boxes  
- Detection confidence  
- Class IDs  

---

## Step 3: Species Filtering (BioCLIP)

Each bounding box is classified using a predefined label list.

```bash
--labels "white-tailed deer,deer,bird,other"
```

### Filtering Logic

- If label ≠ deer → discard detection  
- If no deer remain → skip audio stage  
- If deer detected → run audio model  

---

## Step 4: Audio Spectrogram & CNN

If deer is confirmed:

1. Convert waveform → Mel spectrogram  
2. Normalize  
3. Run CNN classifier  

```bash
python edgeguard_pipeline.py \
  --audio ./input/audio.wav \
  --run-audio-classifier
```

Outputs:

- `anomaly_score`
- `health_status`

---

# Decision Logic

```
IF deer_detected == TRUE:
    Run audio CNN
    IF anomaly_score > threshold:
        health_status = "poor_health_indicator"
    ELSE:
        health_status = "normal"
ELSE:
    Skip audio classification
```

---

# Outputs

All outputs are stored locally on the SD card.

---

## 1️⃣ Annotated Image

Includes:

- Bounding boxes  
- YOLO confidence  
- BioCLIP label  
- Health classification  
- Timestamp  

Saved to:

```
/outputs/images/frame_annotated.jpg
```

---

## 2️⃣ Annotated MP4 Clip

Includes:

- Species label  
- Health status  
- Confidence scores  
- Timestamp  

Saved to:

```
/outputs/videos/event_001.mp4
```

---

## 3️⃣ CSV Log

Saved to:

```
/outputs/logs/edgeguard_events.csv
```

---

# CSV Output Schema

One row per confirmed deer event.

| Column | Type | Description |
|--------|------|------------|
| timestamp | ISO datetime | Event time |
| image_path | String | Annotated image |
| video_path | String | MP4 clip |
| yolo_conf | Float | Detection confidence |
| bioclip_label | String | Species prediction |
| bioclip_score | Float | Similarity score |
| anomaly_score | Float | Audio model output |
| health_status | String | Classification result |
| device | String | edge / desktop |
| inference_latency_ms | Float | Total latency |

---

# Full Pipeline Example

```bash
python edgeguard_pipeline.py \
  --weights yolov8n.pt \
  --image ./input/frame.jpg \
  --audio ./input/audio.wav \
  --conf 0.25 \
  --labels "white-tailed deer,deer,bird" \
  --output-dir ./outputs \
  --device edge
```

---

# Benchmarking

The same pipeline runs on:

- Raspberry Pi (edge)
- Desktop CPU/GPU (baseline)

### Logged Metrics

- Vision latency (ms)
- Audio latency (ms)
- Total pipeline latency
- RAM usage (MB)
- Model size (MB)
- CPU utilization
- Precision / Recall
- Accuracy degradation

Accuracy degradation formula:

```
((Desktop Accuracy - Edge Accuracy) / Desktop Accuracy) * 100
```

> [!NOTE]
> Benchmarking is automatically logged per event when `--device` flag is used.

---

# Energy Optimization

- Event-driven execution  
- Batch size = 1  
- INT8 quantization  
- Reduced input resolution  
- Skip audio when no deer detected  
- Immediate return to idle  

---

# Repository Structure

```
EdgeGuard/
│
├── edgeguard_pipeline.py
├── models/
│   ├── yolov8_lite.pt
│   ├── bioclip_model.pt
│   └── audio_cnn.pt
│
├── outputs/
│   ├── images/
│   ├── videos/
│   └── logs/
│
└── README.md
```

---

# Success Metrics

## Detection

- ≥ 80% precision  
- ≥ 75% recall  

## Audio

- Detect deviation from baseline grazing  

## Edge Optimization

- Inference latency under 500ms (vision)
- Minimal accuracy degradation
- Reduced memory footprint

---

# Non-Goals

- No cloud infrastructure  
- No live GPS tracking  
- No multi-species classification  
- No medical diagnosis  
- No ecological forecasting  

---

# Long-Term Vision

Future expansions may include:

- Solar-powered deployment  
- Multi-species classification  
- Real-time conservation alerts  
- Federated learning  
- Habitat stress analytics  

---

# Why EdgeGuard Matters

EdgeGuard demonstrates:

- Deployable edge AI in remote ecosystems  
- Low-cost conservation monitoring  
- Multimodal intelligence (vision + audio)  
- Measurable optimization tradeoffs  

The innovation is not just detection —  
it is **optimized, measurable, deployable AI for environmental impact**.
