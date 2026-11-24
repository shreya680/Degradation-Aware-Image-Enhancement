"""Zero-DCE Inference CLI — Standalone Image Enhancement Script

Enhance one or more low-light images using a trained Zero-DCE model.
Optionally compute PSNR / SSIM if a reference (ground-truth) image is provided.

Usage Examples:
    # Basic enhancement
    python ml/inference/enhance_inference.py \\
        --input  path/to/low_light.jpg \\
        --output path/to/enhanced.jpg \\
        --weights checkpoints/zero_dce.pth

    # With quality metrics (requires a reference image)
    python ml/inference/enhance_inference.py \\
        --input    path/to/low_light.jpg \\
        --output   path/to/enhanced.jpg \\
        --weights  checkpoints/zero_dce.pth \\
        --reference path/to/ground_truth.jpg

    # Batch directory
    python ml/inference/enhance_inference.py \\
        --input    data/LOL/eval15/low \\
        --output   outputs/eval15_enhanced \\
        --weights  checkpoints/zero_dce.pth \\
        --reference data/LOL/eval15/high
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import ml.*
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
logger = logging.getLogger("enhance_inference")

# ---------------------------------------------------------------------------
# Optional metric dependencies
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.warning("scikit-image not installed — PSNR/SSIM metrics unavailable.")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ===========================================================================
# Core inference helpers
# ===========================================================================

def enhance_single(
    model: DCENet,
    input_path: Path,
    device: torch.device,
    max_size: int = 512,
) -> Tuple[Image.Image, float]:
    """Enhance a single image file.

    Args:
        model:      Loaded DCENet (in eval mode).
        input_path: Path to the input image.
        device:     Torch device.
        max_size:   Maximum edge length for inference (images are resized if larger).

    Returns:
        (enhanced_pil_image, inference_time_seconds)
    """
    img = load_image(input_path)
    original_size = img.size  # (W, H) — keep for final upscale

    # Resize for model if needed (preserves aspect ratio)
    img_resized = resize_for_model(img, max_size=max_size)
    tensor = normalize_image(img_resized).unsqueeze(0).to(device)  # (1, 3, H, W)

    t0 = time.perf_counter()
    with torch.no_grad():
        curves = model(tensor)
        enhanced_tensor = apply_enhancement_curves(
            tensor, curves, num_iterations=model.num_iterations
        )
    elapsed = time.perf_counter() - t0

    # Back to PIL
    enhanced_img = denormalize_image(enhanced_tensor.squeeze(0).cpu())

    # Restore original resolution
    if enhanced_img.size != original_size:
        enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)

    return enhanced_img, elapsed


# ===========================================================================
# Metric helpers
# ===========================================================================

def compute_psnr(enhanced: Image.Image, reference: Image.Image) -> float:
    """Compute PSNR (dB) between enhanced and reference images."""
    if not HAS_SKIMAGE:
        return float("nan")
    enh_arr = np.array(enhanced.convert("RGB"), dtype=np.float32) / 255.0
    ref_arr = np.array(reference.convert("RGB"), dtype=np.float32) / 255.0
    # Resize reference to match enhanced if sizes differ
    if enh_arr.shape != ref_arr.shape:
        ref_pil = reference.resize(enhanced.size, Image.LANCZOS)
        ref_arr = np.array(ref_pil.convert("RGB"), dtype=np.float32) / 255.0
    return float(_psnr(ref_arr, enh_arr, data_range=1.0))


def compute_ssim(enhanced: Image.Image, reference: Image.Image) -> float:
    """Compute SSIM between enhanced and reference images."""
    if not HAS_SKIMAGE:
        return float("nan")
    enh_arr = np.array(enhanced.convert("RGB"), dtype=np.float32) / 255.0
    ref_arr = np.array(reference.convert("RGB"), dtype=np.float32) / 255.0
    if enh_arr.shape != ref_arr.shape:
        ref_pil = reference.resize(enhanced.size, Image.LANCZOS)
        ref_arr = np.array(ref_pil.convert("RGB"), dtype=np.float32) / 255.0
    return float(_ssim(ref_arr, enh_arr, data_range=1.0, channel_axis=2))


def compute_brightness(img: Image.Image) -> Dict[str, float]:
    """Return mean and standard deviation of pixel brightness."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return {"mean_brightness": float(gray.mean()), "std_brightness": float(gray.std())}


def print_metrics(
    input_path: Path,
    enhanced: Image.Image,
    reference: Optional[Image.Image],
    inference_time: float,
    original: Image.Image,
) -> None:
    """Print a formatted metrics table to stdout."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Image : {input_path.name}")
    print(sep)
    print(f"  Inference time : {inference_time * 1000:.1f} ms")

    orig_br = compute_brightness(original)
    enh_br = compute_brightness(enhanced)
    print(f"  Input  brightness : {orig_br['mean_brightness']:.4f}")
    print(f"  Output brightness : {enh_br['mean_brightness']:.4f}")
    print(f"  Brightness gain   : {enh_br['mean_brightness'] - orig_br['mean_brightness']:+.4f}")

    if reference is not None and HAS_SKIMAGE:
        psnr_val = compute_psnr(enhanced, reference)
        ssim_val = compute_ssim(enhanced, reference)
        print(f"  PSNR              : {psnr_val:.2f} dB")
        print(f"  SSIM              : {ssim_val:.4f}")
    elif reference is not None and not HAS_SKIMAGE:
        print("  PSNR / SSIM : N/A (install scikit-image)")

    print(sep + "\n")


# ===========================================================================
# Single image pipeline
# ===========================================================================

def process_single_image(
    model: DCENet,
    input_path: Path,
    output_path: Path,
    reference_path: Optional[Path],
    device: torch.device,
    max_size: int,
) -> Dict[str, float]:
    """Enhance and optionally evaluate a single image."""
    original = load_image(input_path)
    enhanced, t = enhance_single(model, input_path, device, max_size=max_size)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(enhanced, output_path)
    logger.info("Saved enhanced image → %s", output_path)

    # Reference for metrics
    reference = None
    if reference_path is not None and reference_path.is_file():
        reference = load_image(reference_path)

    print_metrics(input_path, enhanced, reference, t, original)

    result: Dict[str, float] = {"inference_time_ms": t * 1000}
    if reference is not None and HAS_SKIMAGE:
        result["psnr"] = compute_psnr(enhanced, reference)
        result["ssim"] = compute_ssim(enhanced, reference)
    return result


# ===========================================================================
# Batch directory pipeline
# ===========================================================================

def process_directory(
    model: DCENet,
    input_dir: Path,
    output_dir: Path,
    reference_dir: Optional[Path],
    device: torch.device,
    max_size: int,
) -> None:
    """Enhance all supported images in input_dir."""
    input_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not input_files:
        logger.error("No supported images found in %s", input_dir)
        sys.exit(1)

    logger.info("Batch mode: %d images to enhance → %s", len(input_files), output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_psnr: List[float] = []
    all_ssim: List[float] = []
    total_time: float = 0.0

    for img_path in input_files:
        out_path = output_dir / img_path.name
        ref_path: Optional[Path] = None
        if reference_dir is not None:
            # Look for matching file (same stem, any extension)
            for ext in SUPPORTED_EXTENSIONS:
                candidate = reference_dir / (img_path.stem + ext)
                if candidate.exists():
                    ref_path = candidate
                    break

        result = process_single_image(model, img_path, out_path, ref_path, device, max_size)
        total_time += result.get("inference_time_ms", 0.0)
        if "psnr" in result:
            all_psnr.append(result["psnr"])
        if "ssim" in result:
            all_ssim.append(result["ssim"])

    # Summary
    print("=" * 52)
    print(f"  Batch complete: {len(input_files)} images")
    print(f"  Avg inference : {total_time / len(input_files):.1f} ms/image")
    if all_psnr:
        print(f"  Avg PSNR      : {np.mean(all_psnr):.2f} dB")
    if all_ssim:
        print(f"  Avg SSIM      : {np.mean(all_ssim):.4f}")
    print("=" * 52)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-DCE Inference CLI — Enhance low-light images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Input image file or directory",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output image file or directory",
    )
    parser.add_argument(
        "--weights", "-w", type=Path, default=Path("checkpoints/zero_dce.pth"),
        help="Path to trained Zero-DCE checkpoint",
    )
    parser.add_argument(
        "--reference", "-r", type=Path, default=None,
        help="Reference (ground-truth) image file or directory for metric computation",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--max_size", type=int, default=512,
        help="Maximum image edge length for inference (larger images are resized)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load model
    model = load_zero_dce(weights_path=args.weights, device=str(device))

    input_path: Path = args.input
    output_path: Path = args.output
    reference_path: Optional[Path] = args.reference

    if input_path.is_dir():
        # Batch mode
        process_directory(
            model, input_path, output_path, reference_path, device, args.max_size
        )
    elif input_path.is_file():
        # Single-image mode
        if output_path.is_dir():
            output_path = output_path / input_path.name
        process_single_image(
            model, input_path, output_path, reference_path, device, args.max_size
        )
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
