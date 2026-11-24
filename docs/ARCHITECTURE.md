# System Architecture & Technical Specifications

**Project Title:** Degradation-Aware Image Enhancement with Perceptual Quality and Downstream Task Preservation  
**Academic Context:** SIC AIML Academic Research & Evaluation Project  

---

## 1. High-Level System Overview

The system is architected as an end-to-end, full-stack AI research platform capable of processing real-world degraded images, diagnosing underlying corruption characteristics, dynamically configuring enhancement trajectories, and rigorously assessing both human perceptual fidelity and machine task preservation.

```
+-----------------------------------------------------------------------------------+
|                                  USER / CLIENT                                    |
|                      React.js 18 + Vite (SPA, Port 5173)                          |
+-----------------------------------------------------------------------------------+
                                    |
                    REST APIs / JSON / Multipart HTTP
                                    v
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND                                    |
|                          Uvicorn Server (Port 8000)                               |
+-----------------------------------------------------------------------------------+
         |                        |                         |
         v                        v                         v
+------------------+    +-------------------+    +--------------------+
|   DEGRADATION    |    |     ADAPTIVE      |    |     DOWNSTREAM     |
|     ANALYSIS     |--->|    ENHANCEMENT    |--->|     EVALUATION     |
|  Signal Metrics  |    | Zero-DCE (PyTorch)|    |  YOLOv8 Detection  |
| Lum/Noise/Blur   |    | Light-Enhancement |    |   COCO / ExDark    |
+------------------+    +-------------------+    +--------------------+
         |                        |                         |
         +------------------------+-------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------------------+
|                        PERCEPTUAL QUALITY & METRICS                               |
|                  PSNR (dB) | SSIM (0-1) | LPIPS Distance                          |
|                     SQLite3 Session Persistence & Logging                         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Components

### A. Frontend Layer (React 18 + Vite)
- **State Management & Persistence:** Cross-view state continuity maintained via local session synchronization (`storage.js`).
- **Interactive UI Modules:**
  1. *Dashboard:* Live hardware device detection, model lifecycle status indicators, and recent session table.
  2. *Enhancement Workspace:* Drag-and-drop file ingestion, real-time degradation diagnosis, and execution trigger.
  3. *Comparison Studio:* Hardware-accelerated split-slider canvas comparing degraded and restored images.
  4. *Downstream Task Detection:* Visual bounding-box comparison overlay with confidence threshold controls.
  5. *Performance Analytics:* Recharts-powered distribution breakdowns and quantitative quality scores.
  6. *Benchmark Datasets:* Specifications and directory verification for LOL and ExDark corpora.
  7. *Academic Project Overview:* Mathematical formulation and viva presentation documentation.

### B. Backend Engine (FastAPI + Async Python)
- **Lifespan Manager:** Models (Zero-DCE DCENet and Ultralytics YOLOv8n) are instantiated during startup, preventing latency spikes on individual inference requests.
- **Static Asset Serving:** Outputs (uploaded source, restored artifacts, annotated detections, and generated reports) mounted securely at `/images`.
- **API Contracts:** Fully typed endpoints supporting asynchronous multipart uploads, JSON payloads, and error boundary handling.

### C. Deep Learning & Computer Vision Core
- **Zero-DCE (Deep Curve Estimation):**
  - Architecture: Lightweight 7-layer convolutional network (`DCENet`) with symmetrical skip connections.
  - Parameter Formulation: Generates 24 feature maps representing channel-wise parameter maps across 8 progressive iterations.
  - Recursion: $LE_n(x) = LE_{n-1}(x) + \mathcal{A}_n \cdot LE_{n-1}(x)(1 - LE_{n-1}(x))$.
  - Efficiency: Fully executable on consumer Intel x86 CPU hardware without dedicated CUDA GPU requirements.
- **YOLOv8 Downstream Detector:**
  - Evaluates target preservation on both degraded and restored pairs at identical confidence thresholds ($default=0.25$).
  - Quantifies object discoverability, classification consistency, and bounding-box stability.

---

## 3. Data Flow & Execution Sequence

```
1. Image Ingestion      --> Client verifies format (JPEG/PNG) & file size (<= 10MB)
2. Signal Analysis      --> Luminance mean, Laplacian variance, and high-frequency noise computation
3. Adaptive Strategy    --> Selection of DCE curves / iteration depth or filtering pipeline
4. Restoration Pass     --> Zero-DCE tensor forward pass (float32, torch.no_grad())
5. Dual YOLOv8 Pass     --> Original and Enhanced inference with annotation generation
6. Metric Computation   --> PSNR, SSIM, and brightness differential calculation
7. Persistence & Report --> Metadata written to SQLite database; CSV/TXT report compiled
```

---

## 4. Hardware and Portability Guarantees

- **Target OS:** Windows 10/11 x64 (Native PowerShell and VS Code).
- **Compute Architecture:** Intel Core i5 with Integrated Graphics (CPU inference mode enforced; auto-detects CUDA if available).
- **RAM Footprint:** Designed to run comfortably within 16 GB RAM by processing images streaming via disk caching.
