"""
Adaptive Enhancement Controller
=================================
Selects and applies the appropriate enhancement strategy based on
the degradation analysis output from the DegradationAnalyzer.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AdaptiveEnhancementController:
    """
    Routes each image to the best enhancement strategy based on its detected
    degradation type and severity.

    Strategies:
      - low_light    → Zero-DCE deep learning model
      - noise        → Gaussian denoising (cv2.GaussianBlur)
      - blur         → Unsharp mask sharpening
      - compression  → Bilateral filtering for artifact removal
    """

    def __init__(self, zero_dce_enhancer=None):
        """
        Args:
            zero_dce_enhancer: An initialised ZeroDCEEnhancer instance.
                               If None, the enhancer will be fetched lazily.
        """
        self._zero_dce = zero_dce_enhancer

    @property
    def zero_dce(self):
        if self._zero_dce is None:
            from backend.enhancement.zero_dce import get_enhancer
            self._zero_dce = get_enhancer()
        return self._zero_dce

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _apply_zero_dce(self, image: Image.Image) -> tuple[Image.Image, str]:
        """Low-light enhancement via Zero-DCE."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            image.save(str(tmp_path))
            enhanced = self.zero_dce.enhance(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return enhanced, "zero_dce"

    @staticmethod
    def _apply_noise_reduction(image: Image.Image, severity_level: str) -> tuple[Image.Image, str]:
        """Gaussian blur denoising – mild kernel, labelled noise_reduction_placeholder."""
        arr = np.array(image.convert("RGB"))
        # Kernel size scales with severity
        ksize = {"Low": 3, "Medium": 5, "High": 7}.get(severity_level, 5)
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # must be odd
        denoised = cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=0)
        return Image.fromarray(denoised, "RGB"), "noise_reduction_placeholder"

    @staticmethod
    def _apply_sharpening(image: Image.Image, severity_level: str) -> tuple[Image.Image, str]:
        """Unsharp mask sharpening for blurry images – labelled sharpening_placeholder."""
        arr = np.array(image.convert("RGB")).astype(np.float32)
        # Unsharp mask: sharp = original + amount * (original - blurred)
        sigma = {"Low": 1.0, "Medium": 1.5, "High": 2.0}.get(severity_level, 1.5)
        amount = {"Low": 0.5, "Medium": 1.0, "High": 1.5}.get(severity_level, 1.0)
        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
        sharpened = arr + amount * (arr - blurred)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return Image.fromarray(sharpened, "RGB"), "sharpening_placeholder"

    @staticmethod
    def _apply_artifact_removal(image: Image.Image, severity_level: str) -> tuple[Image.Image, str]:
        """Bilateral filter to remove JPEG compression artifacts – labelled artifact_removal_placeholder."""
        arr = np.array(image.convert("RGB"))
        # Bilateral filter parameters
        d = {"Low": 5, "Medium": 9, "High": 15}.get(severity_level, 9)
        sigma_color = {"Low": 50, "Medium": 75, "High": 100}.get(severity_level, 75)
        sigma_space = {"Low": 50, "Medium": 75, "High": 100}.get(severity_level, 75)
        filtered = cv2.bilateralFilter(arr, d, sigma_color, sigma_space)
        return Image.fromarray(filtered, "RGB"), "artifact_removal_placeholder"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def enhance(
        self,
        image_path: str | Path,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        degradation_type = analysis.get("degradation_type", "low_light")
        severity_level = analysis.get("severity_level", "Medium")
        detected = analysis.get("detected_degradations", [degradation_type])
        if not isinstance(detected, (list, tuple, set)) or not detected:
            detected = [degradation_type]
        valid_types = {"low_light", "noise", "blur", "compression"}
        detected = [d for d in detected if d in valid_types]
        if not detected:
            detected = [degradation_type]

        enhancement_order = [
            "low_light",
            "noise",
            "compression",
            "blur",
        ]

        t0 = time.perf_counter()
        enhanced_image = pil_image
        applied_strategies = []
        models_used = []
        applied_degradations = set()

        for degradation in enhancement_order:
            if degradation not in detected:
                continue

            if degradation == "low_light":
                enhanced_image, strategy = self._apply_zero_dce(enhanced_image)
                enhanced_array = np.array(enhanced_image.convert("RGB"))
                if float(np.mean(enhanced_array)) < 75:
                    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe = cv2.createCLAHE(
                        clipLimit=1.5,
                        tileGridSize=(8, 8)
                    )
                    enhanced_l = clahe.apply(l_channel)
                    enhanced_array = cv2.cvtColor(
                        cv2.merge((enhanced_l, a_channel, b_channel)),
                        cv2.COLOR_LAB2RGB
                    )
                    enhanced_image = Image.fromarray(enhanced_array, "RGB")
                    strategy = "numpy_curve+clahe_fallback"
                model = "numpy_curve+clahe" if not self.zero_dce.weights_loaded else "zero_dce"
            elif degradation == "noise":
                enhanced_image, strategy = self._apply_noise_reduction(
                    enhanced_image,
                    severity_level,
                )
                model = "cv2.GaussianBlur"
            elif degradation == "compression":
                enhanced_image, strategy = self._apply_artifact_removal(
                    enhanced_image,
                    severity_level,
                )
                model = "cv2.bilateralFilter"
            else:
                blur_severity = "Low" if "noise" in applied_degradations else severity_level
                enhanced_image, strategy = self._apply_sharpening(
                    enhanced_image,
                    blur_severity,
                )
                model = "unsharp_mask"

            applied_strategies.append(strategy)
            models_used.append(model)
            applied_degradations.add(degradation)

        t1 = time.perf_counter()
        processing_time_ms = round((t1 - t0) * 1000, 2)

        return {
            "enhanced_image": enhanced_image,
            "degradation_type": degradation_type,
            "severity_level": severity_level,
            "strategy_chosen": " + ".join(applied_strategies) or "none",
            "model_used": " + ".join(models_used) or "none",
            "processing_time_ms": processing_time_ms,
            "detected_degradations": list(detected),
        }


# Module-level singleton
_controller_instance: Optional[AdaptiveEnhancementController] = None


def get_controller(zero_dce_enhancer=None) -> AdaptiveEnhancementController:
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = AdaptiveEnhancementController(zero_dce_enhancer)
    return _controller_instance
