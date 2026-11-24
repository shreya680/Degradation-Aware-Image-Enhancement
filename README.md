# Degradation-Aware Image Enhancement with Perceptual Quality and Downstream Task Preservation

**SIC AIML Academic Research Project**

## 1. Overview

This project aims to develop an adaptive image enhancement system that identifies image degradation and improves visual quality while preserving important information required for downstream computer vision tasks.

The system focuses on analyzing degraded images, selecting suitable enhancement strategies, and evaluating the impact of enhancement on image quality and object detection performance.

## 2. Problem Statement

Images captured in real-world environments often suffer from degradation such as:

- Low illumination
- Image noise
- Motion and defocus blur
- Compression artifacts

Conventional enhancement methods may improve visual appearance while losing important information needed by computer vision models.

Our project aims to address this challenge through degradation analysis, adaptive enhancement, and downstream task evaluation.

## 3. Proposed Approach

The proposed system follows these stages:

1. **Image Input:** Accepts degraded images.
2. **Degradation Analysis:** Identifies image quality issues and estimates their severity.
3. **Adaptive Enhancement:** Selects an appropriate enhancement strategy based on the detected degradation.
4. **Perceptual Evaluation:** Measures image quality using metrics such as PSNR, SSIM, and LPIPS, where applicable.
5. **Downstream Task Evaluation:** Examines object detection performance before and after enhancement.

## 4. Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, CSS |
| Backend | Python, FastAPI |
| Image Processing | OpenCV, Pillow, NumPy |
| Deep Learning | PyTorch, Zero-DCE |
| Object Detection | YOLOv8 |
| Evaluation | PSNR, SSIM, LPIPS |
| Database | SQLite |

## 5. Datasets

- **LOL Dataset:** Paired low-light and normal-light images for image enhancement evaluation.
- **ExDark Dataset:** Real-world low-light images for downstream object detection evaluation.

Dataset setup and configuration details are available in `docs/DATASET_SETUP.md`.

## 6. System Architecture

```text
       Input Image
            |
            v
   Degradation Analysis
            |
            v
   Adaptive Enhancement
            |
            v
     Enhanced Image
            |
       +----+----+
       |         |
       v         v
  Quality     Object Detection
  Evaluation   Evaluation
       |         |
       +----+----+
            |
            v
     Results Dashboard
```

## 7. Project Structure

```text
sic_projj/
├── backend/       # FastAPI backend and image processing
├── frontend/      # React user interface
├── ml/            # Model training, inference and evaluation
├── docs/          # Project documentation
├── data/          # Dataset directory
├── checkpoints/   # Model weights directory
└── outputs/       # Generated results
```

## 8. Current Development Status

The initial project structure and backend API have been created. The backend server is running, and API documentation is accessible.

Frontend integration, trained model loading, and end-to-end enhancement and evaluation testing are ongoing.

## 9. Future Scope

- Improve enhancement across multiple degradation types.
- Integrate and validate trained enhancement models.
- Evaluate downstream object detection preservation.
- Conduct benchmark experiments using LOL and ExDark datasets.
- Compare enhancement methods using perceptual and task-based metrics.

---

Project Category: Artificial Intelligence and Machine Learning  