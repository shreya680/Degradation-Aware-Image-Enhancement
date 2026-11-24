"""Batch Evaluation Pipeline — LOL Dataset (eval15)

Evaluates a trained Zero-DCE model against the LOL evaluation split (eval15),
computing PSNR and SSIM for each image pair and saving a results CSV.

Usage:
    python ml/evaluation/eval_pipeline.py \\
        --lol_path data/LOL \\
        --weights  checkpoints/zero_dce.pth \\
        --output   results/eval_results.csv \\
        --device   cpu

The script also prints a summary table to stdout.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Add project root to sys.path
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml.models.zero_dce_arch import DCENet, apply_enhancement_curves, load_zero_dce
from ml.preprocessing.image_transforms import (
    normalize_image,
    denormalize_image,
    resize_for_model,
    load_image,
    save_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_pipeline")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.error(
        "scikit-image is required for evaluation. "
        "Install it with: pip install scikit-image"
    )
    sys.exit(1)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ===========================================================================
# Metric computations
# ===========================================================================

def pil_to_float_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 numpy array in [0, 1], shape (H, W, 3)."""
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def compute_psnr(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """PSNR in dB. Both arrays should be float32 in [0, 1], shape (H, W, 3)."""
    return float(_psnr(reference, enhanced, data_range=1.0))


def compute_ssim(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """SSIM. Both arrays should be float32 in [0, 1], shape (H, W, 3)."""
    return float(_ssim(reference, enhanced, data_range=1.0, channel_axis=2))


def compute_brightness_metrics(arr: np.ndarray) -> Dict[str, float]:
    """Compute mean and std of luminance from an (H, W, 3) float array."""
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return {
        "mean_brightness": float(gray.mean()),
        "std_brightness": float(gray.std()),
    }


# ===========================================================================
# Inference helper
# ===========================================================================

def enhance_image(
    model: DCENet,
    input_path: Path,
    device: torch.device,
    max_size: int = 512,
) -> Tuple[Image.Image, float]:
    """Run Zero-DCE inference on a single image.

    Returns:
        (enhanced PIL Image at original resolution, inference time in seconds)
    """
    img = load_image(input_path)
    original_size = img.size

    img_resized = resize_for_model(img, max_size=max_size)
    tensor = normalize_image(img_resized).unsqueeze(0).to(device)  # (1, 3, H, W)

    t0 = time.perf_counter()
    with torch.no_grad():
        curves = model(tensor)
        enhanced_t = apply_enhancement_curves(tensor, curves, num_iterations=model.num_iterations)
    elapsed = time.perf_counter() - t0

    enhanced_img = denormalize_image(enhanced_t.squeeze(0).cpu())
    if enhanced_img.size != original_size:
        enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)

    return enhanced_img, elapsed


# ===========================================================================
# Main evaluation function
# ===========================================================================

def evaluate_lol_dataset(
    lol_path: Path,
    model: DCENet,
    device: torch.device = torch.device("cpu"),
    split: str = "eval15",
    max_size: int = 512,
    save_enhanced: Optional[Path] = None,
) -> List[Dict]:
    """Evaluate the Zero-DCE model on the LOL eval split.

    Args:
        lol_path:      Root directory of the LOL dataset.
        model:         Loaded DCENet model in eval() mode.
        device:        Torch device.
        split:         Subdirectory to evaluate (default: 'eval15').
        max_size:      Resize limit for inference.
        save_enhanced: Optional directory to save enhanced images.

    Returns:
        List of dicts, one per image, with keys:
            filename, psnr, ssim, mean_brightness_in, mean_brightness_out,
            inference_time_ms
    """
    low_dir = lol_path / split / "low"
    high_dir = lol_path / split / "high"

    if not low_dir.exists():
        raise FileNotFoundError(
            f"LOL eval low directory not found: {low_dir}\n"
            f"See docs/DATASET_SETUP.md for download instructions."
        )
    if not high_dir.exists():
        raise FileNotFoundError(
            f"LOL eval high directory not found: {high_dir}\n"
            f"See docs/DATASET_SETUP.md for download instructions."
        )

    low_files = sorted([
        f for f in low_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not low_files:
        raise ValueError(f"No images found in {low_dir}")

    logger.info(
        "Evaluating %d image pairs from %s/%s ...", len(low_files), lol_path, split
    )

    if save_enhanced:
        save_enhanced.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    for idx, low_path in enumerate(low_files, start=1):
        # Match ground truth
        ref_path = high_dir / low_path.name
        if not ref_path.exists():
            for ext in SUPPORTED_EXTENSIONS:
                candidate = high_dir / (low_path.stem + ext)
                if candidate.exists():
                    ref_path = candidate
                    break
            else:
                logger.warning("No reference found for %s — skipping.", low_path.name)
                continue

        # Enhance
        enhanced, t = enhance_image(model, low_path, device, max_size=max_size)

        # Save if requested
        if save_enhanced:
            save_image(enhanced, save_enhanced / low_path.name)

        # Compute metrics
        ref_img = load_image(ref_path)
        # Align sizes for metric computation
        if ref_img.size != enhanced.size:
            ref_img = ref_img.resize(enhanced.size, Image.LANCZOS)

        enh_arr = pil_to_float_array(enhanced)
        ref_arr = pil_to_float_array(ref_img)
        low_arr = pil_to_float_array(load_image(low_path))

        psnr_val = compute_psnr(enh_arr, ref_arr)
        ssim_val = compute_ssim(enh_arr, ref_arr)
        br_in = compute_brightness_metrics(low_arr)["mean_brightness"]
        br_out = compute_brightness_metrics(enh_arr)["mean_brightness"]

        row = {
            "filename": low_path.name,
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "mean_brightness_in": round(br_in, 4),
            "mean_brightness_out": round(br_out, 4),
            "brightness_gain": round(br_out - br_in, 4),
            "inference_time_ms": round(t * 1000, 2),
        }
        results.append(row)

        logger.info(
            "[%2d/%d] %-30s  PSNR=%6.2f dB  SSIM=%.4f  time=%.1f ms",
            idx, len(low_files), low_path.name, psnr_val, ssim_val, t * 1000
        )

    return results


def save_results_csv(results: List[Dict], output_path: Path) -> None:
    """Write evaluation results to a CSV file."""
    if not results:
        logger.warning("No results to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results saved → %s", output_path)


def print_summary(results: List[Dict]) -> None:
    """Print a formatted summary table."""
    if not results:
        print("No results to summarise.")
        return

    psnr_vals = [r["psnr"] for r in results]
    ssim_vals = [r["ssim"] for r in results]
    times = [r["inference_time_ms"] for r in results]
    br_gains = [r["brightness_gain"] for r in results]

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION SUMMARY — {len(results)} images")
    print(sep)
    print(f"  {'Metric':<25}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("  " + "─" * 56)

    def row(name: str, vals: List[float], fmt: str = ".4f") -> None:
        print(
            f"  {name:<25}  "
            f"{np.mean(vals):>10{fmt}}  "
            f"{np.std(vals):>10{fmt}}  "
            f"{np.min(vals):>10{fmt}}  "
            f"{np.max(vals):>10{fmt}}"
        )

    row("PSNR (dB)", psnr_vals, ".2f")
    row("SSIM", ssim_vals, ".4f")
    row("Brightness Gain", br_gains, ".4f")
    row("Inference (ms)", times, ".1f")
    print(sep + "\n")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Zero-DCE on the LOL eval15 split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lol_path", type=Path, default=Path("data/LOL"),
        help="Root directory of the LOL dataset",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("checkpoints/zero_dce.pth"),
        help="Path to trained Zero-DCE checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/eval_results.csv"),
        help="CSV output file path",
    )
    parser.add_argument(
        "--split", type=str, default="eval15",
        help="LOL dataset split to evaluate (folder name inside lol_path)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device string (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--max_size", type=int, default=512,
        help="Maximum image edge length for inference",
    )
    parser.add_argument(
        "--save_enhanced", type=Path, default=None,
        help="Optional directory to save enhanced output images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load model
    model = load_zero_dce(weights_path=args.weights, device=str(device))

    # Evaluate
    results = evaluate_lol_dataset(
        lol_path=args.lol_path,
        model=model,
        device=device,
        split=args.split,
        max_size=args.max_size,
        save_enhanced=args.save_enhanced,
    )

    # Save CSV
    save_results_csv(results, args.output)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
