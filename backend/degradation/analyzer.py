"""
Degradation Analyzer
====================
Real signal-processing-based degradation detection.
Detects: low_light, noise, blur, compression artifacts.
Returns dominant type, severity_score (0-1), severity_level, confidence, all_scores.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_gray(image_path: str | Path) -> np.ndarray:
    """Load an image as uint8 grayscale numpy array."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


def _load_bgr(image_path: str | Path) -> np.ndarray:
    """Load an image as uint8 BGR numpy array."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


# ---------------------------------------------------------------------------
# Individual degradation scorers  (all return 0.0–1.0, higher = worse)
# ---------------------------------------------------------------------------

def _score_low_light(gray: np.ndarray) -> tuple[float, float]:
    """
    Low-light score based on mean pixel brightness.
    Returns (severity_score 0-1, confidence 0-1).
    """
    mean_brightness = float(np.mean(gray))
    # Map brightness to severity score (inverted: dark → high score)
    # mean < 100 → high severity (score > 0.66)
    # 100-130    → medium severity (score 0.33-0.66)
    # 130+       → low severity (score < 0.33)
    score = max(0.0, min(1.0, 1.0 - (mean_brightness / 255.0)))
    # Confidence: strong when image is very dark or very bright
    if mean_brightness < 80 or mean_brightness > 180:
        confidence = 0.90
    elif mean_brightness < 120 or mean_brightness > 150:
        confidence = 0.75
    else:
        confidence = 0.55
    return score, confidence


def _score_noise(gray: np.ndarray) -> tuple[float, float]:
    """
    Noise score via Laplacian standard deviation.
    High std of local Laplacian → high-frequency content → more noise.
    Returns (severity_score 0-1, confidence 0-1).
    """
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    lap_std = float(np.std(laplacian))
    # Empirical thresholds: std > 60 suggests strong noise
    # We normalise against 120 (heavy noise) → score 0–1
    score = min(1.0, lap_std / 120.0)
    confidence = 0.70 if score > 0.3 else 0.50
    return score, confidence


def _score_blur(gray: np.ndarray) -> tuple[float, float]:
    """
    Blur score via Laplacian variance.
    Low variance → blurry image (lacks high-frequency detail).
    Returns (severity_score 0-1, confidence 0-1).
    """
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    lap_var = float(np.var(laplacian))
    # Empirical: var < 100 → very blurry; var > 1000 → sharp
    # Invert and normalise: score = 1 - clamp(var / 500, 0, 1)
    score = max(0.0, 1.0 - min(1.0, lap_var / 500.0))
    confidence = 0.80 if score > 0.5 else 0.55
    return score, confidence


def _score_compression(gray: np.ndarray) -> tuple[float, float]:
    """
    JPEG compression artifact score via DCT block analysis.
    Detects 8×8 block boundary discontinuities.
    Returns (severity_score 0-1, confidence 0-1).
    """
    h, w = gray.shape
    # Compute horizontal and vertical block-boundary differences at 8-pixel intervals
    h_bound = (h // 8) * 8
    w_bound = (w // 8) * 8
    img_clipped = gray[:h_bound, :w_bound].astype(np.float32)

    # Horizontal block boundaries (columns 7, 15, 23, …)
    block_cols = list(range(7, w_bound - 1, 8))
    block_rows = list(range(7, h_bound - 1, 8))

    if not block_cols or not block_rows:
        return 0.0, 0.3

    # Average absolute difference at boundaries vs. within blocks
    boundary_diff_h = float(np.mean(np.abs(
        img_clipped[:, block_cols].astype(float) -
        img_clipped[:, [c + 1 for c in block_cols]].astype(float)
    )))
    boundary_diff_v = float(np.mean(np.abs(
        img_clipped[block_rows, :].astype(float) -
        img_clipped[[r + 1 for r in block_rows], :].astype(float)
    )))

    # Average interior differences (between adjacent pixels inside blocks)
    all_h_diff = float(np.mean(np.abs(np.diff(img_clipped, axis=1))))
    all_v_diff = float(np.mean(np.abs(np.diff(img_clipped, axis=0))))

    # Compression artifacts → boundary diff notably higher than interior diff
    ratio_h = boundary_diff_h / (all_h_diff + 1e-6)
    ratio_v = boundary_diff_v / (all_v_diff + 1e-6)
    ratio = (ratio_h + ratio_v) / 2.0

    # ratio ~1.0 means no blocking, ratio >> 1.5 means blocking
    score = min(1.0, max(0.0, (ratio - 1.0) / 2.0))
    confidence = 0.75 if score > 0.3 else 0.45
    return score, confidence


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

def _severity_level(score: float, low_light: bool = False) -> str:
    """Convert a score to a human-readable severity level."""
    if low_light:
        # For low-light use the specified thresholds
        # mean < 100 → High, 100-130 → Medium, 130+ → Low
        # score = 1 - mean/255 → score > 0.608 → High, 0.49-0.608 → Medium
        if score > 0.608:   # mean < 100
            return "High"
        elif score > 0.490:  # 100-130
            return "Medium"
        else:
            return "Low"
    else:
        if score > 0.66:
            return "High"
        elif score > 0.33:
            return "Medium"
        else:
            return "Low"


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class DegradationAnalyzer:
    """Signal-processing based degradation analyzer."""

    DEGRADATION_TYPES = ["low_light", "noise", "blur", "compression"]

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        """
        Analyze the image for degradation.

        Returns a dict with:
          - degradation_type: dominant type
          - severity_score: 0.0-1.0
          - severity_level: "Low" / "Medium" / "High"
          - confidence: 0.0-1.0
          - all_scores: {type: score, ...}
          - details: descriptive strings per type
        """
        image_path = Path(image_path)
        gray = _load_gray(image_path)

        ll_score, ll_conf = _score_low_light(gray)
        noise_score, noise_conf = _score_noise(gray)
        blur_score, blur_conf = _score_blur(gray)
        comp_score, comp_conf = _score_compression(gray)

        all_scores = {
            "low_light": round(ll_score, 4),
            "noise": round(noise_score, 4),
            "blur": round(blur_score, 4),
            "compression": round(comp_score, 4),
        }

        all_confs = {
            "low_light": ll_conf,
            "noise": noise_conf,
            "blur": blur_conf,
            "compression": comp_conf,
        }

        # Dominant type = highest score
        dominant_type = max(all_scores, key=lambda k: all_scores[k])
        dominant_score = all_scores[dominant_type]
        dominant_conf = all_confs[dominant_type]

        # Severity
        is_ll = dominant_type == "low_light"
        sev_level = _severity_level(dominant_score, low_light=is_ll)

        # Build human-readable details
        mean_brightness = float(np.mean(gray))
        lap_var = float(np.var(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)))
        details = {
            "mean_brightness": round(mean_brightness, 2),
            "laplacian_variance": round(lap_var, 2),
            "low_light_score": all_scores["low_light"],
            "noise_score": all_scores["noise"],
            "blur_score": all_scores["blur"],
            "compression_score": all_scores["compression"],
            "image_size": f"{gray.shape[1]}x{gray.shape[0]}",
        }

        return {
            "degradation_type": dominant_type,
            "severity_score": round(dominant_score, 4),
            "severity_level": sev_level,
            "confidence": round(dominant_conf, 4),
            "all_scores": all_scores,
            "details": details,
        }


# Module-level singleton
_analyzer_instance: DegradationAnalyzer | None = None


def analyze_image(image_path: str | Path) -> dict[str, Any]:
    """Convenience function using a module-level singleton analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DegradationAnalyzer()
    return _analyzer_instance.analyze(image_path)
