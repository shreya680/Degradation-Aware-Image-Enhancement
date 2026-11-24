# Degradation-Aware Image Enhancement with Perceptual Quality and Downstream Task Preservation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **SIC AIML Final Academic Research & Evaluation Project**  
> An intelligent computer vision system that diagnoses image corruption, dynamically tailors enhancement strategies using Zero-Reference Deep Curve Estimation (**Zero-DCE**), and quantifies downstream computer vision preservation with **YOLOv8**.

---

## 1. Problem Background & Research Objectives

Real-world images captured in adverse environments frequently suffer from multifaceted degradations:
- Low-light & non-uniform illumination
- Additive Gaussian / sensor noise
- Motion and defocus blur
- Compression artifacts (JPEG block boundaries)

Standard enhancement methods apply static filters or global histogram equalization, resulting in unnatural color casts, amplified noise, and degraded downstream AI task performance.

### Proposed System Objectives
1. **Degradation Detection & Severity Estimation:** Automated signal analysis classifying severity (Low, Medium, High).
2. **Adaptive Image Enhancement:** Dynamic configuration of Zero-DCE Light-Enhancement curves without over-saturation.
3. **Perceptual Quality Preservation:** Reference-based (PSNR, SSIM, LPIPS) evaluation on paired benchmarks.
4. **Downstream Task Preservation:** Validating object detection stability and discovery using YOLOv8 across 80 COCO categories.

---

## 2. System Architecture

```
[ Input Image (JPEG/PNG) ]
           │
           ▼
[ Degradation Analyzer ] ──► (Luminance, Laplacian variance, Noise estimate)
           │
           ▼
[ Adaptive Controller ]  ──► (Selects curve iterations & strategy parameters)
           │
           ▼
[ Zero-DCE Enhancer ]    ──► (7-layer DCENet applies 8 iterative LE curves)
           │
           ├───────────────────────────────┐
           ▼                               ▼
[ Downstream YOLOv8 Detection ]   [ Perceptual Quality Evaluation ]
   Raw vs. Enhanced Detections        PSNR, SSIM, LPIPS, Brightness
           │                               │
           └───────────────┬───────────────┘
                           ▼
[ Interactive SaaS Web Dashboard & Reports (React + Recharts) ]
```

---

## 3. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend** | React 18, Vite | Component-based modern SPA |
| **Visuals & Charts**| Lucide React, Recharts | Icons, analytics dashboards, bar charts |
| **Backend Framework**| FastAPI, Uvicorn | High-performance asynchronous REST API |
| **Deep Learning** | PyTorch (CPU-compatible) | Zero-DCE DCENet tensor inference |
| **Object Detection**| Ultralytics YOLOv8n | Downstream task preservation evaluation |
| **Computer Vision** | OpenCV, Pillow, NumPy | Preprocessing, annotation drawing, filters |
| **Quality Metrics** | Scikit-Image, LPIPS | PSNR, SSIM, perceptual patch distance |
| **Persistence** | SQLite3 | Session history and processing logs |

---

## 4. Benchmark Datasets

- **LOL Dataset (v1):** Paired low-light dataset (485 training pairs, 15 evaluation pairs) captured under identical conditions for supervised and full-reference perceptual quality evaluation.
- **ExDark (Exclusively Dark Dataset):** 7,363 low-light images across 12 object classes (Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table) for downstream object detection validation.

*See [`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md) for download links and configuration.*

---

## 5. Project Directory Structure

```
sic_projj/
├── backend/
│   ├── main.py                  # FastAPI server entry point & lifespan loader
│   ├── requirements.txt         # Python dependencies
│   ├── api/
│   │   └── routes.py            # REST API endpoints (/upload, /analyze, /enhance, etc.)
│   ├── degradation/
│   │   └── analyzer.py          # Signal-processing degradation detection
│   ├── enhancement/
│   │   ├── zero_dce.py          # DCENet architecture & enhancement inference
│   │   └── adaptive.py          # Adaptive enhancement controller
│   ├── detection/
│   │   └── yolov8_detector.py   # YOLOv8 downstream object detection wrapper
│   ├── evaluation/
│   │   └── metrics.py           # PSNR, SSIM, LPIPS, and brightness metrics
│   ├── datasets/
│   │   ├── lol_dataset.py       # LOL dataset loader & pair matcher
│   │   └── exdark_dataset.py    # ExDark loader & annotation parser
│   └── services/
│       ├── image_service.py     # File validation and preprocessing
│       ├── history_service.py   # SQLite processing history tracker
│       └── report_service.py    # CSV and TXT diagnostic report generator
├── frontend/
│   ├── package.json             # NPM dependencies
│   ├── vite.config.js           # Vite server & backend API proxy configuration
│   └── src/
│       ├── App.jsx              # Main React router & layout
│       ├── pages/               # 8 Complete pages (Landing, Dashboard, Workspace, etc.)
│       ├── components/          # Reusable UI widgets & comparison slider
│       ├── services/            # Axios API client & localStorage persistence
│       └── styles/              # Dark SaaS CSS design system
├── ml/
│   ├── models/                  # Standalone DCENet model definition
│   ├── training/                # Zero-DCE trainer on LOL dataset (4 losses)
│   ├── inference/               # Standalone CLI inference script
│   ├── preprocessing/           # Transforms and normalization
│   └── evaluation/              # Batch evaluation scripts
├── data/                        # Dataset mount point (LOL, ExDark)
├── checkpoints/                 # Model checkpoints directory (zero_dce.pth, yolov8n.pt)
├── outputs/                     # Uploads, enhanced outputs, annotated images, reports
├── docs/                        # Architecture, weights guide, and dataset docs
└── README.md
```

---

## 6. Windows Execution Instructions (VS Code)

### Prerequisites
- **OS:** Windows 10 or 11 (64-bit)
- **Python:** 3.10+ (Verified with Python 3.12)
- **Node.js:** 18+ (Verified with Node.js 20+)
- **Hardware:** Intel Core i5 with Integrated Graphics (CPU inference mode supported out-of-the-box).

---

### Step 1: Clone or Open in VS Code
Open the project root directory in VS Code:
```powershell
cd C:\Users\Admin\Desktop\sic_projj
```

---

### Step 2: Set Up Backend (Python Virtual Environment)
Open a PowerShell terminal in VS Code:
```powershell
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Set Up Frontend (Node.js)
Open a second PowerShell terminal in VS Code:
```powershell
# Navigate to frontend
cd frontend

# Install dependencies (use cmd /c if running with restricted execution policy)
cmd /c "npm install"
```

---

### Step 4: Run the Application

#### Terminal 1 — Start Backend Server:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Backend runs at: [http://localhost:8000](http://localhost:8000)*  
*Interactive Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)*

#### Terminal 2 — Start Frontend Application:
```powershell
cd frontend
cmd /c "npm run dev"
```
*Frontend runs at: [http://localhost:5173](http://localhost:5173)*

---

## 7. Interactive Features & Navigation

1. **Main Dashboard (`/`):** Hardware detection, live Zero-DCE and YOLOv8 status cards, session history log.
2. **Image Enhancement Workspace (`/enhance`):** Drag-and-drop image upload, degradation diagnosis, and Zero-DCE execution.
3. **Interactive Comparison Studio (`/compare`):** Side-by-side or draggable split-slider comparing original vs. restored image.
4. **Downstream Object Detection (`/detect`):** YOLOv8 visual annotations on both original and enhanced images, confidence threshold slider, and class count breakdown chart.
5. **Performance Analytics (`/analytics`):** Real computed PSNR, SSIM, and brightness gain charts.
6. **Benchmark Datasets (`/datasets`):** LOL and ExDark specifications, loading checks, and local directory guide.
7. **Academic Project Info (`/about`):** Detailed research objectives, mathematical formulation, and system architecture for academic viva demonstration.

---

## 8. REST API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health status and loaded model hardware device info |
| `GET` | `/api/models/status` | Detailed loading status for Zero-DCE and YOLOv8 |
| `GET` | `/api/datasets/status`| Local counts and directory availability for LOL and ExDark |
| `GET` | `/api/history` | List of last 20 processed image sessions |
| `POST` | `/api/upload` | Upload an image; returns filename and metadata |
| `POST` | `/api/analyze` | Run degradation analysis (luminance, noise, blur) |
| `POST` | `/api/enhance` | Run Zero-DCE adaptive enhancement |
| `POST` | `/api/detect` | Run YOLOv8 on original and/or enhanced images |
| `POST` | `/api/evaluate` | Calculate PSNR, SSIM, LPIPS, and brightness metrics |
| `POST` | `/api/process` | All-in-one pipeline (Upload ➔ Analyze ➔ Enhance ➔ Detect ➔ Evaluate) |

---

## 9. Academic Citations

```bibtex
@inproceedings{guo2020zero,
  title={Zero-reference deep curve estimation for low-light image enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1780--1789},
  year={2020}
}

@inproceedings{wei2018deep,
  title={Deep Retinex Decomposition for Low-Light Enhancement},
  author={Wei, Chen and Wang, Wenjing and Yang, Wenhan and Liu, Jiaying},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2018}
}

@article{loh2020getting,
  title={Getting to know low-light images with the Exclusively Dark dataset},
  author={Loh, Yuen Peng and Chan, Chee Seng},
  journal={Computer Vision and Image Understanding (CVIU)},
  volume={198},
  pages={103017},
  year={2020}
}
```

---

## 10. License
This project is open-source under the [MIT License](LICENSE).
