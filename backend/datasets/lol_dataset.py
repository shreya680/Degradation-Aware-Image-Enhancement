"""
LOL Dataset Loader
===================
Loads the LOL (Low-Light) dataset from data/LOL/.
Structure expected:
  data/LOL/our485/low/   ← 485 low-light training images
  data/LOL/our485/high/  ← 485 normal-light training images
  data/LOL/eval15/low/   ← 15 low-light evaluation images
  data/LOL/eval15/high/  ← 15 normal-light evaluation images
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOL_ROOT = PROJECT_ROOT / "data" / "LOL"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _get_image_files(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _pair_images(low_dir: Path, high_dir: Path) -> list[dict[str, str]]:
    """
    Pair low and high images by filename (stem).
    Returns list of {low, high} dicts with relative path strings.
    """
    low_files = {f.stem: f for f in _get_image_files(low_dir)}
    high_files = {f.stem: f for f in _get_image_files(high_dir)}

    pairs = []
    for stem in sorted(low_files.keys()):
        if stem in high_files:
            pairs.append({
                "low": str(low_files[stem].relative_to(PROJECT_ROOT)),
                "high": str(high_files[stem].relative_to(PROJECT_ROOT)),
                "filename": stem,
            })
    return pairs


class LOLDataset:
    """Loader for the LOL (Low-Light) paired dataset."""

    def __init__(self, lol_root: Path | None = None):
        self.root = Path(lol_root) if lol_root else LOL_ROOT

    def get_stats(self) -> dict[str, Any]:
        """
        Return dataset statistics.

        Returns:
            {
                available: bool,
                train_count: int,
                eval_count: int,
                sample_pairs: list (up to 5 example pairs),
                path: str,
            }
        """
        available = self.root.exists()

        train_pairs: list[dict[str, str]] = []
        eval_pairs: list[dict[str, str]] = []

        if available:
            our485_low = self.root / "our485" / "low"
            our485_high = self.root / "our485" / "high"
            eval15_low = self.root / "eval15" / "low"
            eval15_high = self.root / "eval15" / "high"

            train_pairs = _pair_images(our485_low, our485_high)
            eval_pairs = _pair_images(eval15_low, eval15_high)

            if not train_pairs and not eval_pairs:
                logger.warning(f"LOL dataset found at {self.root} but no paired images detected.")

        return {
            "available": available and (len(train_pairs) > 0 or len(eval_pairs) > 0),
            "train_count": len(train_pairs),
            "eval_count": len(eval_pairs),
            "sample_pairs": (train_pairs + eval_pairs)[:5],
            "path": str(self.root),
        }

    def get_train_pairs(self) -> list[dict[str, str]]:
        """Return all training image pairs."""
        our485_low = self.root / "our485" / "low"
        our485_high = self.root / "our485" / "high"
        return _pair_images(our485_low, our485_high)

    def get_eval_pairs(self) -> list[dict[str, str]]:
        """Return all evaluation image pairs."""
        eval15_low = self.root / "eval15" / "low"
        eval15_high = self.root / "eval15" / "high"
        return _pair_images(eval15_low, eval15_high)
