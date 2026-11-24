# Zero-DCE Model Weights & Checkpoint Guide

## 1. Overview
Zero-DCE (Zero-Reference Deep Curve Estimation) is a self-supervised low-light image enhancement network introduced by Chongyi Li et al. (CVPR 2020). It does not require paired low/normal light images for training.

---

## 2. Checkpoint Locations

The backend automatically searches for the Zero-DCE model weights at:
```
checkpoints/zero_dce.pth
```

If `zero_dce.pth` exists in `checkpoints/`, the system loads the trained weights into the `DCENet` architecture at startup.

---

## 3. How to Obtain Pretrained Weights

### Option A: Official Repository Download
1. Navigate to the official Zero-DCE repository:
   [https://github.com/Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
2. In the repository's `snapshots/` folder, locate the pretrained checkpoint:
   - File name: `Epoch99.pth` or `zero_dce.pth`
3. Download the `.pth` file.
4. Rename the file to `zero_dce.pth` and place it into:
   ```
   C:\Users\Admin\Desktop\sic_projj\checkpoints\zero_dce.pth
   ```

### Option B: Local Training Pipeline (Included)
A complete training script with the 4 Zero-DCE loss functions is included in the project:
```bash
python ml/training/zero_dce_trainer.py --data_path data/LOL --epochs 100 --batch_size 4
```
Upon completion, the trainer saves `zero_dce.pth` directly into `checkpoints/`.

---

## 4. Behavior When Weights Are Missing (Graceful Fallback)

To ensure the application runs out-of-the-box on any new machine without crashing:
- If `checkpoints/zero_dce.pth` is not yet downloaded, the backend:
  1. Instantiates the authentic `DCENet` PyTorch architecture.
  2. Sets the network to evaluation mode (`model.eval()`).
  3. Clearly logs a status notice in the backend console:
     `"Notice: Using freshly initialized Zero-DCE architecture (checkpoints/zero_dce.pth not found)."`
  4. Executes the genuine curve estimation equations $LE_n(x)$ without throwing unhandled exceptions.
  5. The web interface displays the exact model loading status transparently on the Dashboard and Sidebar.
