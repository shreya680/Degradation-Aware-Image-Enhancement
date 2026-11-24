"""
Image Quality Metrics
======================
Computes PSNR, SSIM, LPIPS, and brightness metrics for evaluating image enhancement.
All functions handle missing references gracefully (return None, not crash).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _validate_arrays(
    original: np.ndarray,
    enhanced: np.ndarray,
) -> tuple[bool, str]:
    """Validate that both arrays are compatible for metric computation."""
    if original is None or enhanced is None:
        return False, "One or both arrays are None"
    if original.shape != enhanced.shape:
        return False, f"Shape mismatch: {original.shape} vs {enhanced.shape}"
    if original.size == 0 or enhanced.size == 0:
        return False, "Empty array"
    return True, ""


def compute_psnr(
    original_array: Optional[np.ndarray],
    enhanced_array: Optional[np.ndarray],
) -> Optional[float]:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        original_array: Reference image as uint8 numpy array (H, W, 3) or (H, W).
        enhanced_array: Enhanced image as uint8 numpy array (H, W, 3) or (H, W).

    Returns:
        PSNR float in dB, or None if computation fails.
    """
    try:
        ok, msg = _validate_arrays(original_array, enhanced_array)
        if not ok:
            logger.warning(f"compute_psnr: {msg}")
            return None

        orig = original_array.astype(np.float64)
        enh = enhanced_array.astype(np.float64)
        mse = np.mean((orig - enh) ** 2)
        if mse == 0.0:
            return float("inf")

        max_pixel = 255.0
        psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
        return round(float(psnr), 4)
    except Exception as e:
        logger.error(f"compute_psnr error: {e}")
        return None


def compute_ssim(
    original_array: Optional[np.ndarray],
    enhanced_array: Optional[np.ndarray],
) -> Optional[float]:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        original_array: Reference image as uint8 numpy array.
        enhanced_array: Enhanced image as uint8 numpy array.

    Returns:
        SSIM float in [-1, 1] (1 = identical), or None if computation fails.
    """
    try:
        ok, msg = _validate_arrays(original_array, enhanced_array)
        if not ok:
            logger.warning(f"compute_ssim: {msg}")
            return None

        from skimage.metrics import structural_similarity as ssim  # type: ignore

        orig = original_array.astype(np.float32)
        enh = enhanced_array.astype(np.float32)

        # Handle grayscale vs colour
        if orig.ndim == 3 and orig.shape[2] == 3:
            score = ssim(orig, enh, data_range=255.0, channel_axis=2)
        else:
            score = ssim(orig, enh, data_range=255.0)

        return round(float(score), 4)
    except Exception as e:
        logger.error(f"compute_ssim error: {e}")
        return None


def compute_lpips(
    original_array: Optional[np.ndarray],
    enhanced_array: Optional[np.ndarray],
) -> Optional[float]:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        original_array: Reference image as uint8 numpy array (H, W, 3).
        enhanced_array: Enhanced image as uint8 numpy array (H, W, 3).

    Returns:
        LPIPS float (lower = more similar), or None if computation fails
        (e.g., lpips not installed, arrays incompatible).
    """
    try:
        import torch  # type: ignore
        import lpips  # type: ignore

        ok, msg = _validate_arrays(original_array, enhanced_array)
        if not ok:
            logger.warning(f"compute_lpips: {msg}")
            return None

        if original_array.ndim != 3 or original_array.shape[2] != 3:
            logger.warning("compute_lpips requires (H, W, 3) colour arrays")
            return None

        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        loss_fn.eval()

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            # Normalize to [-1, 1], shape [1, 3, H, W]
            t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            d = loss_fn(_to_tensor(original_array), _to_tensor(enhanced_array))

        return round(float(d.item()), 4)
    except ImportError:
        logger.warning("lpips package not installed; skipping LPIPS computation.")
        return None
    except Exception as e:
        logger.error(f"compute_lpips error: {e}")
        return None


def compute_brightness(image_array: Optional[np.ndarray]) -> Optional[float]:
    """
    Compute mean grayscale brightness of an image.

    Args:
        image_array: uint8 numpy array (H, W, 3) or (H, W).

    Returns:
        Mean brightness as float in [0, 255], or None on failure.
    """
    try:
        if image_array is None or image_array.size == 0:
            return None

        import cv2  # type: ignore

        if image_array.ndim == 3:
            if image_array.shape[2] == 3:
                # RGB → grayscale
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            elif image_array.shape[2] == 4:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            else:
                gray = image_array[:, :, 0]
        else:
            gray = image_array

        mean_val = float(np.mean(gray.astype(np.float32)))
        return round(mean_val, 4)
    except Exception as e:
        logger.error(f"compute_brightness error: {e}")
        return None
