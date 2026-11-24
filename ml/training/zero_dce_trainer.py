"""Zero-DCE Training Script — LOL Dataset

Train DCENet on the LOL (Low-light) dataset using the four unsupervised losses
described in the Zero-DCE paper (Guo et al., CVPR 2020).

Usage:
    python ml/training/zero_dce_trainer.py \\
        --data_path data/LOL \\
        --epochs 200 \\
        --batch_size 8 \\
        --lr 0.0001 \\
        --save_path checkpoints/zero_dce.pth

The trainer works on CPU-only machines and also accelerates on CUDA/MPS when
available. All file paths are handled via pathlib.Path.
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Optional tqdm import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zero_dce_trainer")

# ---------------------------------------------------------------------------
# Local imports — allow running from project root or ml/ directory
# ---------------------------------------------------------------------------
import sys
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml.models.zero_dce_arch import DCENet, apply_enhancement_curves
from ml.preprocessing.image_transforms import normalize_image, TrainingTransformPair


# ===========================================================================
# Dataset
# ===========================================================================

class LOLDataset(Dataset):
    """PyTorch Dataset for the LOL (Low-light) dataset.

    Expected directory layout::

        lol_root/
            our485/
                low/   ← 485 low-light training images
                high/  ← 485 corresponding normal-light images
            eval15/
                low/   ← 15 evaluation images
                high/

    Args:
        lol_root:   Root directory of the LOL dataset.
        split:      ``'train'`` (our485) or ``'eval'`` (eval15).
        crop_size:  Random crop size applied during training.
        augment:    Whether to apply random flipping / cropping augmentation.
    """

    _SPLIT_DIRS = {"train": "our485", "eval": "eval15"}

    def __init__(
        self,
        lol_root: Path,
        split: str = "train",
        crop_size: int = 256,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.lol_root = Path(lol_root)
        self.split = split
        self.augment = augment and (split == "train")

        split_dir = self._SPLIT_DIRS.get(split, "our485")
        self.low_dir = self.lol_root / split_dir / "low"
        self.high_dir = self.lol_root / split_dir / "high"

        if not self.low_dir.exists():
            raise FileNotFoundError(
                f"LOL low-light directory not found: {self.low_dir}\n"
                f"See docs/DATASET_SETUP.md for download instructions."
            )

        # Build paired file list (sorted for reproducibility)
        self.low_files: List[Path] = sorted(self.low_dir.glob("*.png")) + \
                                     sorted(self.low_dir.glob("*.jpg")) + \
                                     sorted(self.low_dir.glob("*.bmp"))
        if not self.low_files:
            raise FileNotFoundError(
                f"No images found in {self.low_dir}. "
                "Supported formats: .png, .jpg, .bmp"
            )

        self.transform = TrainingTransformPair(
            crop_size=crop_size,
            hflip=self.augment,
            vflip=False,
            rotate=False,
        )
        logger.info(
            "LOLDataset [%s]: %d image pairs found in %s",
            split, len(self.low_files), self.lol_root
        )

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_path = self.low_files[idx]
        # Match corresponding high-light image by stem name
        high_path = self.high_dir / low_path.name
        if not high_path.exists():
            # Try alternative extensions
            for ext in (".png", ".jpg", ".bmp"):
                candidate = self.high_dir / (low_path.stem + ext)
                if candidate.exists():
                    high_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No high-light pair found for: {low_path}"
                )

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        low_t, high_t = self.transform(low_img, high_img)
        return low_t, high_t


# ===========================================================================
# Loss Functions
# ===========================================================================

class SpatialConsistencyLoss(nn.Module):
    """Spatial Consistency Loss (L_spa).

    Preserves the spatial relationship between neighbouring regions of the
    input and enhanced image. Uses a 4-connectivity neighbourhood kernel.

    Reference: Zero-DCE paper Eq. (10)
    """

    def __init__(self) -> None:
        super().__init__()
        # 4-directional difference kernels (up, down, left, right)
        kernels = torch.zeros(4, 1, 3, 3)
        kernels[0, 0, 0, 1] = -1; kernels[0, 0, 1, 1] = 1  # up
        kernels[1, 0, 2, 1] = -1; kernels[1, 0, 1, 1] = 1  # down
        kernels[2, 0, 1, 0] = -1; kernels[2, 0, 1, 1] = 1  # left
        kernels[3, 0, 1, 2] = -1; kernels[3, 0, 1, 1] = 1  # right
        self.register_buffer("kernels", kernels)

    def forward(
        self, enhanced: torch.Tensor, original: torch.Tensor
    ) -> torch.Tensor:
        # Convert to grayscale (average channels)
        enh_gray = enhanced.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        ori_gray = original.mean(dim=1, keepdim=True)   # (B, 1, H, W)

        loss = torch.tensor(0.0, device=enhanced.device)
        for k in range(4):
            kernel = self.kernels[k]  # (1, 1, 3, 3)
            d_enh = F.conv2d(enh_gray, kernel, padding=1)
            d_ori = F.conv2d(ori_gray, kernel, padding=1)
            loss = loss + torch.mean(torch.abs(d_enh - d_ori) ** 2)
        return loss


class ExposureControlLoss(nn.Module):
    """Exposure Control Loss (L_exp).

    Encourages the mean value of a local region (patch) in the enhanced image
    to be close to a target exposure level E.

    Reference: Zero-DCE paper Eq. (11)

    Args:
        patch_size: Local region size (default 16).
        mean_val:   Target well-exposedness value E ∈ (0, 1) (default 0.6).
    """

    def __init__(self, patch_size: int = 16, mean_val: float = 0.6) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.mean_val = mean_val

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        gray = enhanced.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mean_patch = self.pool(gray)                # (B, 1, H', W')
        return torch.mean((mean_patch - self.mean_val) ** 2)


class ColorConstancyLoss(nn.Module):
    """Color Constancy Loss (L_col).

    Corrects potential colour deviations introduced by enhancement by
    minimising the difference between the mean values of R, G, and B channels.

    Reference: Zero-DCE paper Eq. (12)
    """

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        mean_r = enhanced[:, 0, :, :].mean(dim=[1, 2])  # (B,)
        mean_g = enhanced[:, 1, :, :].mean(dim=[1, 2])
        mean_b = enhanced[:, 2, :, :].mean(dim=[1, 2])

        loss = (
            (mean_r - mean_g) ** 2
            + (mean_r - mean_b) ** 2
            + (mean_g - mean_b) ** 2
        )
        return loss.mean()


class IlluminationSmoothnessLoss(nn.Module):
    """Illumination Smoothness Loss (L_tvA).

    Encourages spatial smoothness of the enhancement curve maps A by
    penalising large total variation.

    Reference: Zero-DCE paper Eq. (13)

    Args:
        tv_loss_weight: Weighting factor (default 1).
    """

    def __init__(self, tv_loss_weight: float = 1.0) -> None:
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curves: Curve maps (B, 3*n, H, W) output from DCENet.
        """
        batch_size = curves.shape[0]
        h_x = curves.shape[2]
        w_x = curves.shape[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)

        h_tv = torch.pow(curves[:, :, 1:, :] - curves[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(curves[:, :, :, 1:] - curves[:, :, :, :-1], 2).sum()

        return self.tv_loss_weight * (h_tv / count_h + w_tv / count_w) / batch_size


class ZeroDCELoss(nn.Module):
    """Combined Zero-DCE training loss.

    L = w_spa * L_spa + w_exp * L_exp + w_col * L_col + w_tva * L_tva

    Default weights from the paper:
        w_spa = 1, w_exp = 10, w_col = 5, w_tva = 1600
    """

    def __init__(
        self,
        w_spa: float = 1.0,
        w_exp: float = 10.0,
        w_col: float = 5.0,
        w_tva: float = 1600.0,
        exposure_patch_size: int = 16,
        exposure_mean_val: float = 0.6,
    ) -> None:
        super().__init__()
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tva = w_tva

        self.spatial_loss = SpatialConsistencyLoss()
        self.exposure_loss = ExposureControlLoss(exposure_patch_size, exposure_mean_val)
        self.color_loss = ColorConstancyLoss()
        self.smooth_loss = IlluminationSmoothnessLoss()

    def forward(
        self,
        enhanced: torch.Tensor,
        original: torch.Tensor,
        curves: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_spa = self.spatial_loss(enhanced, original)
        l_exp = self.exposure_loss(enhanced)
        l_col = self.color_loss(enhanced)
        l_tva = self.smooth_loss(curves)

        total = (
            self.w_spa * l_spa
            + self.w_exp * l_exp
            + self.w_col * l_col
            + self.w_tva * l_tva
        )

        breakdown = {
            "loss_total": total.item(),
            "loss_spa": l_spa.item(),
            "loss_exp": l_exp.item(),
            "loss_col": l_col.item(),
            "loss_tva": l_tva.item(),
        }
        return total, breakdown


# ===========================================================================
# Training Loop
# ===========================================================================

def train_one_epoch(
    model: DCENet,
    loader: DataLoader,
    criterion: ZeroDCELoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Dict[str, float]:
    """Run one complete training epoch."""
    model.train()

    totals: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_spa": 0.0,
        "loss_exp": 0.0,
        "loss_col": 0.0,
        "loss_tva": 0.0,
    }

    iterable = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}") if HAS_TQDM else loader

    for batch_idx, (low, _high) in enumerate(iterable):
        low = low.to(device)

        optimizer.zero_grad()
        curves = model(low)
        enhanced = apply_enhancement_curves(low, curves, num_iterations=model.num_iterations)

        loss, breakdown = criterion(enhanced, low, curves)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in breakdown.items():
            totals[k] += v

        if HAS_TQDM:
            iterable.set_postfix({
                "loss": f"{breakdown['loss_total']:.4f}",
                "spa": f"{breakdown['loss_spa']:.4f}",
                "exp": f"{breakdown['loss_exp']:.4f}",
                "col": f"{breakdown['loss_col']:.4f}",
            })
        elif (batch_idx + 1) % 10 == 0:
            logger.info(
                "  [%d/%d] batch %d | total=%.4f spa=%.4f exp=%.4f col=%.4f tva=%.4f",
                epoch, num_epochs, batch_idx + 1,
                breakdown["loss_total"],
                breakdown["loss_spa"],
                breakdown["loss_exp"],
                breakdown["loss_col"],
                breakdown["loss_tva"],
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def save_checkpoint(
    model: DCENet,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: Path,
) -> None:
    """Save model weights and training state."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    logger.info("Checkpoint saved → %s (epoch %d, loss=%.4f)", save_path, epoch, loss)


def train(
    data_path: Path,
    save_path: Path,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-4,
    crop_size: int = 256,
    num_workers: int = 0,
    weight_decay: float = 1e-4,
    save_every: int = 50,
    resume: Optional[Path] = None,
) -> None:
    """Main training function.

    Args:
        data_path:   Root directory of the LOL dataset.
        save_path:   Path where the final checkpoint will be saved.
        epochs:      Number of training epochs.
        batch_size:  Samples per batch.
        lr:          Initial Adam learning rate.
        crop_size:   Random crop size for data augmentation.
        num_workers: DataLoader worker processes (0 = main process, safe on Windows).
        weight_decay: Adam weight decay.
        save_every:  Save a checkpoint every N epochs.
        resume:      Path to a checkpoint to resume training from.
    """
    # ------------------------------------------------------------------
    # Device selection (CPU-safe, uses CUDA / MPS if available)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Training device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Training device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info(
            "Training device: CPU — training will be slow. "
            "Consider reducing batch_size and epochs for quick tests."
        )

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset = LOLDataset(
        lol_root=data_path,
        split="train",
        crop_size=crop_size,
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = DCENet(in_channels=3, num_filters=32, num_iterations=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = ZeroDCELoss()

    start_epoch = 1
    if resume is not None and Path(resume).is_file():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from checkpoint: %s (epoch %d)", resume, ckpt["epoch"])

    logger.info(
        "Starting training: epochs=%d  batch=%d  lr=%.0e  device=%s",
        epochs, batch_size, lr, device,
    )

    best_loss = float("inf")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            model, loader, criterion, optimizer, device, epoch, epochs
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d  |  total=%.4f  spa=%.4f  exp=%.4f  col=%.4f  tva=%.4f  "
            "lr=%.2e  time=%.1fs",
            epoch, epochs,
            metrics["loss_total"],
            metrics["loss_spa"],
            metrics["loss_exp"],
            metrics["loss_col"],
            metrics["loss_tva"],
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # Save periodic checkpoint
        if epoch % save_every == 0:
            periodic_path = save_path.parent / f"zero_dce_ep{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, metrics["loss_total"], periodic_path)

        # Save best checkpoint
        if metrics["loss_total"] < best_loss:
            best_loss = metrics["loss_total"]
            save_checkpoint(
                model, optimizer, epoch, metrics["loss_total"],
                save_path.parent / "zero_dce_best.pth"
            )

    # Final save
    save_checkpoint(model, optimizer, epochs, metrics["loss_total"], save_path)
    logger.info("Training complete. Final checkpoint: %s", save_path)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Zero-DCE on the LOL dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=Path, default=Path("data/LOL"),
        help="Root directory of the LOL dataset",
    )
    parser.add_argument(
        "--save_path", type=Path, default=Path("checkpoints/zero_dce.pth"),
        help="Output checkpoint file path",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (0 is safest on Windows)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--save_every", type=int, default=50,
        help="Save a periodic checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint to resume training from",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        resume=args.resume,
    )
