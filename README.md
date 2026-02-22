# EdgeGuard Wildlife Monitoring System

EdgeGuard is a **low-power, multimodal edge AI pipeline** for wildlife monitoring deployed on a Raspberry Pi.

It performs:

1. **Animal detection** using YOLOv8  
2. **Conditional audio health classification** using a CNN on spectrograms  
3. **Event logging and media storage** (JPG, MP4, CSV)  
4. **Edge vs Desktop benchmarking**

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
- [Audio CNN Architecture](#audio-cnn-architecture)
- [Benchmarking](#benchmarking)
- [Benchmark Results: Intel Xeon vs Raspberry Pi 5](#benchmark-results-intel-xeon-vs-raspberry-pi-5)
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
- TensorFlow/Keras
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
pip install librosa matplotlib
pip install tensorflow
```

---

# Pipeline Workflow

<img width="2604" height="836" alt="Cropped-logiclow" src="https://github.com/user-attachments/assets/e9358140-db2a-478e-9668-823a9be25400" />

## Step 1: Motion Trigger

When PIR detects motion:

1. Capture image  
2. Record 5–10 second audio clip  
3. Launch inference  
4. Return to idle  

> [!TIP]
> Audio inference only runs if deer is detected in the image to conserve energy.

---

## Step 2: Object Detection (YOLOv8)

```bash
python edgeguard_pipeline.py \
  --weights yolov8n.pt \
  --image ./input/frame.jpg \
  --conf 0.25
```

Output:

- Bounding boxes  
- Detection confidence  
- Class labels (from YOLO's 80 COCO classes)

---

## Step 3: Species Filtering

Detections are filtered using YOLO's native class labels.

### Filtering Logic

- If class ≠ target animal (e.g., deer) → discard detection  
- If no target animals remain → skip audio stage  
- If target animal detected → run audio model  

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
- Class label  
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
| yolo_class | String | YOLO class label |
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
  --output-dir ./outputs \
  --device edge
```

---

# Audio CNN Architecture

The audio health classification model uses a **lightweight CNN architecture optimized for ARM-based edge devices** like the Raspberry Pi 5.

## Model Design

```python
model = models.Sequential([
    # Separable Conv is easier on ARM CPUs
    layers.SeparableConv1D(32, kernel_size=3, padding='same', input_shape=input_shape),
    layers.Activation('relu'),
    layers.MaxPooling1D(2),
    
    layers.SeparableConv1D(64, kernel_size=3, padding='same'),
    layers.Activation('relu'),
    
    # Flatten is faster for tiny feature sets (24 features)
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
```

## Architecture Decisions

### 1. Separable Convolutions (`SeparableConv1D`)

Standard convolutions perform a single operation that combines spatial filtering and channel mixing. **Separable convolutions** split this into two steps:

1. **Depthwise convolution**: Applies a single filter per input channel
2. **Pointwise convolution**: 1×1 convolution to combine channels

**Why this matters for edge deployment:**

| Metric | Standard Conv | Separable Conv | Benefit |
|--------|--------------|----------------|---------|
| Parameters | k × C_in × C_out | k × C_in + C_in × C_out | ~3-9x fewer |
| Multiply-Adds | O(k × C_in × C_out × L) | O(k × C_in × L + C_in × C_out × L) | Significantly reduced |
| ARM Efficiency | Memory-bound | Compute-efficient | Better cache utilization |

For our model with 24 acoustic features:
- **Standard Conv (3×1×32)**: 96 parameters per layer
- **Separable Conv**: 3 + 32 = 35 parameters (~63% reduction)

ARM Cortex-A76 cores (Pi5) have smaller caches than x86. Separable convolutions reduce memory bandwidth requirements, leading to faster inference.

### 2. Flatten vs Global Pooling

For tiny feature sets (24 acoustic features), **Flatten** outperforms GlobalAveragePooling:

| Approach | Operation | Best For |
|----------|-----------|----------|
| GlobalAveragePooling | Averages across spatial dimension | Large feature maps (images) |
| Flatten | Preserves all features | Small feature sets (<100 features) |

With only 24 input features and 2 conv layers, the intermediate feature map is small enough that:
- Flatten adds minimal overhead
- No information loss from averaging
- Dense layers can learn fine-grained patterns

### 3. Model Summary

```
Layer (type)                Output Shape         Param #
================================================================
separable_conv1d            (None, 24, 32)       67
activation                  (None, 24, 32)       0
max_pooling1d               (None, 12, 32)       0
separable_conv1d_1          (None, 12, 64)       2,176
activation_1                (None, 12, 64)       0
flatten                     (None, 768)          0
dense                       (None, 32)           24,608
dropout                     (None, 32)           0
dense_1                     (None, 1)            33
================================================================
Total params: 26,884
Trainable params: 26,884
```

### 4. Input Features (24 Acoustic Features)

The CNN processes 24 pre-extracted acoustic features:

| Category | Features |
|----------|----------|
| **Temporal** | duration_s, time_to_peak_s |
| **Signal Quality** | snr_db |
| **Pitch (F0)** | f0_mean_hz, f0_min_hz, f0_max_hz, voiced_ratio |
| **Intensity** | intensity_min_db, intensity_max_db |
| **Formants** | f1_mean_hz, f2_mean_hz |
| **Energy** | rms_energy_mean, rms_energy_std |
| **Spectral** | spectral_centroid_mean, spectral_bandwidth_mean, spectral_rolloff_95 |
| **Zero-Crossing** | zcr_mean, zcr_std |
| **MFCCs** | mfcc1_mean, mfcc1_std, mfcc2_mean, mfcc2_std, mfcc3_mean, mfcc3_std |

### 5. Performance Impact

The optimized architecture achieved a **36% runtime reduction** on Raspberry Pi 5:

| Version | Runtime | Peak RAM |
|---------|---------|----------|
| Standard CNN | 94.10 s | 1,173.83 MB |
| **Optimized (Separable)** | **60.58 s** | **1,013.39 MB** |

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

# Benchmark Results: Intel Xeon vs Raspberry Pi 5

We conducted comparative benchmarks between a high-performance server CPU and an edge device to quantify the tradeoffs between compute power and energy efficiency.

## Test Environment

### Server Platform (OSC - Ohio Supercomputer Center)

| Specification | Details |
|--------------|---------|
| **CPU** | Intel Xeon Gold 6148 |
| **Clock Speed** | 2.4 GHz (base) / 3.7 GHz (turbo) |
| **Cores** | 20 cores / 40 threads |
| **Architecture** | Skylake-SP (14nm) |
| **L3 Cache** | 27.5 MB |
| **Memory Bandwidth** | 128 GB/s (6-channel DDR4-2666) |
| **TDP** | 150W |
| **Compute Mode** | CPU-only (CUDA disabled) |

### Edge Platform (Raspberry Pi 5)

| Specification | Details |
|--------------|---------|
| **CPU** | Broadcom BCM2712 (Arm Cortex-A76) |
| **Clock Speed** | 2.4 GHz |
| **Cores** | 4 cores / 4 threads |
| **Architecture** | Arm v8.2-A (16nm) |
| **L3 Cache** | 2 MB shared |
| **Memory** | 8 GB LPDDR4X-4267 |
| **TDP** | ~12W (under load) |
| **Compute Mode** | CPU-only |

---

## Benchmark Configuration

```yaml
Model: YOLOv8n (nano)
Inference Mode: CPU-only
Confidence Threshold: 0.25
Keep Threshold: 0.50
Input Resolution: Native (no preprocessing resize)
Batch Size: 1 (per-image inference)
Resource Sampling: 1 second intervals (OSC) / 0.5 second intervals (Pi)
```

---

## Performance Results

### Computer Vision (YOLOv8) Pipeline

| Metric | Intel Xeon Gold 6148 | Raspberry Pi 5 | Comparison |
|--------|---------------------|----------------|------------|
| **Total Runtime** | 40.46 s | 34.03 s | Pi 16% faster |
| **Peak Process RAM** | 2,802.58 MB | 2,461.81 MB | Pi uses 12% less |
| **Peak System Memory** | 7.1% | 44.2% | Xeon has more headroom |
| **Peak CPU Utilization** | 6.1% | 44.5% | Pi runs near capacity |

### Audio CNN Pipeline

| Metric | Intel Xeon Gold 6148 | Raspberry Pi 5 (Optimized) |
|--------|---------------------|---------------------------|
| **Total Runtime** | 112.95 s | 60.58 s |
| **Peak Process RAM** | 12.19 MB | 1,013.39 MB |
| **Peak CPU Utilization** | 17.2% | 46.5% |

### Combined Pipeline Performance

| Metric | Intel Xeon Gold 6148 | Raspberry Pi 5 |
|--------|---------------------|----------------|
| **Total Pipeline Time** | ~153.4 s | ~94.6 s |
| **Vision + Audio Combined** | CV + CNN | CV + CNN (Optimized) |

---

## Resource Utilization Over Time

### Intel Xeon Gold 6148 (OSC)

**Computer Vision (YOLO):**
- Startup RAM: 404.8 MB → Peak: 2,802.58 MB
- CPU remained low (0.3% – 6.1%) due to 20-core architecture
- Disk usage stable at ~29.1 GB used / 160.9 GB free

**Audio CNN:**
- Extremely lightweight: constant 12.19 MB RAM
- CPU: ~10-17% utilization
- Two-phase execution visible: preprocessing (~70s) + inference (~35s)

### Raspberry Pi 5

**Computer Vision (YOLO):**
- Startup RAM: 1.05 MB → Peak: 2,461.81 MB
- CPU consistently high (35-45%) — fully utilizing 4 cores
- System memory peaked at 44.2%

**Audio CNN:**
- RAM: peaked at 1,013.39 MB (optimized)
- CPU: 36-48% utilization
- Optimized version reduced runtime by 36%

---

## Analysis

### Performance Gap

Both platforms operate at identical base clock speeds (2.4 GHz), making this a direct comparison of:

1. **Core count**: 20 cores (Xeon) vs 4 cores (Pi5)
2. **Architecture efficiency**: Skylake x86 vs Cortex-A76 ARM
3. **Memory subsystem**: 128 GB/s bandwidth vs ~34 GB/s
4. **Cache hierarchy**: 27.5 MB L3 vs 2 MB L3

### Key Findings

**Surprising Result: Raspberry Pi 5 outperformed Intel Xeon on both pipelines**

This counterintuitive result is explained by:

1. **Single-threaded workload**: YOLOv8n inference is largely single-threaded, negating the Xeon's core count advantage
2. **Memory efficiency**: ARM architecture's tighter memory integration benefits inference workloads
3. **CPU-only mode**: The Xeon's strengths (AVX-512, massive parallelism) are underutilized without GPU offload
4. **Optimized CNN architecture**: Separable convolutions are specifically designed for ARM efficiency

**Intel Xeon Gold 6148 Advantages:**
- Massive memory headroom (6.5-7.1% system usage vs 44%)
- Better suited for parallel batch processing
- Lower per-core utilization leaves room for concurrent tasks
- Superior for training workloads

**Raspberry Pi 5 Advantages:**
- Faster single-stream inference
- ~12.5x lower power consumption (12W vs 150W)
- Deployable in remote/off-grid locations
- Lower cost per node ($80 vs $3,000+)
- Sufficient for real-time single-image inference

### Energy Efficiency

| Metric | Intel Xeon Gold 6148 | Raspberry Pi 5 |
|--------|---------------------|----------------|
| **TDP** | 150W | ~12W |
| **CV Pipeline Energy** | ~6,069 J | ~408 J |
| **Energy Ratio** | 1x | **14.9x more efficient** |

> [!IMPORTANT]
> Both platforms produce **identical inference results** when using the same model weights. The accuracy degradation is **0%** — only latency and resource usage differ.

---

## Running the Benchmark

### On OSC (Intel Xeon)

```bash
# SSH into OSC
ssh username@owens.osc.edu

# Navigate to project directory
cd /fs/scratch/PAS2136/EdgeGuard

# Run benchmark script
python src/osc_monitor.py
```

### On Raspberry Pi 5

```bash
# Run benchmark script
python src/animalClassifierTelemtry.py
```

### Output Files

Both scripts generate:

| File | Description |
|------|-------------|
| `detections.csv` | Detection results with confidence scores |
| `resource_log_cpu_only.csv` | Time-series resource utilization |

---

## Resource Log Schema

| Column | Type | Description |
|--------|------|-------------|
| timestamp | ISO datetime | Sample time |
| elapsed_s | Float | Seconds since start |
| cpu_percent_total | Float | System CPU utilization |
| proc_rss_mb | Float | Process memory (MB) |
| sys_mem_percent | Float | System memory utilization |
| disk_used_gb | Float | Disk space used |
| disk_free_gb | Float | Disk space available |

---

## Visualization

To generate comparison charts from benchmark logs:

```bash
python scripts/plot_benchmarks.py \
  --xeon results/xeon_resource_log.csv \
  --pi5 results/pi5_resource_log.csv \
  --output benchmark_comparison.png
```

---

# Energy Optimization

- Event-driven execution  
- Batch size = 1  
- INT8 quantization  
- Reduced input resolution  
- Skip audio when no deer detected  
- Immediate return to idle
- Separable convolutions for ARM efficiency

---

# Repository Structure

```
EdgeGuard/
│
├── edgeguard_pipeline.py
├── src/
│   ├── main.py                    # Audio feature extraction + CNN
│   ├── osc_monitor.py             # OSC benchmark script
│   ├── animalClassifierTelemtry.py # Pi benchmark script
│   └── Requirements.txt
│
├── models/
│   ├── yolov8n.pt
│   ├── lightweight_cnn_model.h5   # Optimized audio CNN
│   └── scaler.pkl
│
├── outputs/
│   ├── images/
│   ├── videos/
│   └── logs/
│
├── results/
│   ├── xeon_resource_log.csv
│   └── pi5_resource_log.csv
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

## Benchmark Goals

- Document performance gap between server and edge
- Validate identical accuracy across platforms
- Quantify energy efficiency advantage of edge deployment

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
