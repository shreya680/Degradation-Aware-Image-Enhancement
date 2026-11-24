# Dataset Setup & Configuration Guide (Windows)

This document provides exact instructions for configuring the two academic benchmark datasets used in this research project:
1. **LOL (Low-Light Dataset)** - BMVC 2018 (Paired Enhancement)
2. **ExDark (Exclusively Dark Dataset)** - CVIU 2020 (Object Detection)

---

## 1. LOL Dataset (Low-Light Image Enhancement)

### Purpose
Used for supervised model evaluation, paired reference comparisons, and ground-truth metrics (PSNR, SSIM, LPIPS).

### Source
- Official Webpage: [https://daooshee.github.io/BMVC2018website/](https://daooshee.github.io/BMVC2018website/)
- Alternative (Baidu/Google Drive links provided on author's page).

### Folder Structure
Unzip the downloaded LOL dataset archive so that files match this structure:
```
data/
└── LOL/
    ├── our485/
    │   ├── low/      (485 low-light training images)
    │   └── high/     (485 normal-light ground truth training images)
    └── eval15/
        ├── low/      (15 low-light test images)
        └── high/     (15 normal-light ground truth test images)
```

Each image in `low/` has a matching filename in `high/` (e.g. `1.png`, `2.png`).

---

## 2. ExDark Dataset (Exclusively Dark Image Dataset)

### Purpose
Used for downstream computer vision validation and object detection evaluation in low-light conditions.

> **CRITICAL ACADEMIC DISTINCTION:**  
> ExDark is an object detection dataset and does NOT have paired normal-light reference images. It is evaluated via YOLOv8 detection counts and bounding-box preservation, NOT full-reference PSNR/SSIM.

### Source
- Official GitHub: [https://github.com/cs-chan/Exclusively-Dark-Image-Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

### Folder Structure
Extract the images and annotation files into `data/ExDark/`:
```
data/
└── ExDark/
    ├── Bicycle/
    ├── Boat/
    ├── Bottle/
    ├── Bus/
    ├── Car/
    ├── Cat/
    ├── Chair/
    ├── Cup/
    ├── Dog/
    ├── Motorbike/
    ├── People/
    ├── Table/
    └── Annotations/ (optional, contains .txt class and bounding boxes)
```

---

## 3. Dataset Verification

Once the folders are populated, you can verify dataset loading in two ways:
1. **Via Web Interface:** Open the **Dataset Information** tab (`/datasets`) and check that status badges display *Configured & Active*.
2. **Via REST API:**
   ```
   GET http://localhost:8000/api/datasets/status
   ```
   Returns JSON summary with counts of available images in both LOL and ExDark.
