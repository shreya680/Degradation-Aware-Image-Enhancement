"""
ExDark Dataset Loader
======================
Loads the ExDark (Exclusively Dark) dataset from data/ExDark/.
Categories: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table
Reads annotation files if present (ExDark standard .txt annotation format).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXDARK_ROOT = PROJECT_ROOT / "data" / "ExDark"

CATEGORIES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _count_images_in_dir(directory: Path) -> int:
    """Count image files in a directory recursively."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _get_counts_per_category(root: Path) -> dict[str, int]:
    """
    Count images per ExDark category.
    ExDark directory structure: ExDark/<Category>/ containing images.
    Annotation files may live in ExDark/Annotations/<Category>/ or alongside images.
    """
    counts: dict[str, int] = {}
    for cat in CATEGORIES:
        cat_dir = root / cat
        counts[cat] = _count_images_in_dir(cat_dir)
    return counts


def _read_annotation(annotation_path: Path) -> list[dict[str, Any]]:
    """
    Parse a single ExDark annotation .txt file.
    Format (per line after header):
      <class_name> <x> <y> <w> <h> <tag>
    Returns list of {class_name, x, y, w, h}.
    """
    annotations = []
    if not annotation_path.exists():
        return annotations
    try:
        lines = annotation_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) >= 5:
                class_name = parts[0]
                try:
                    x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    annotations.append({"class_name": class_name, "x": x, "y": y, "w": w, "h": h})
                except ValueError:
                    continue
    except Exception as e:
        logger.warning(f"Could not read annotation {annotation_path}: {e}")
    return annotations


class ExDarkDataset:
    """Loader for the ExDark exclusively-dark image dataset."""

    def __init__(self, exdark_root: Path | None = None):
        self.root = Path(exdark_root) if exdark_root else EXDARK_ROOT

    def get_stats(self) -> dict[str, Any]:
        """
        Return dataset statistics.

        Returns:
            {
                available: bool,
                total_images: int,
                categories: list[str],
                counts_per_category: dict[str, int],
                path: str,
                annotations_available: bool,
            }
        """
        available = self.root.exists()

        if not available:
            return {
                "available": False,
                "total_images": 0,
                "categories": CATEGORIES,
                "counts_per_category": {cat: 0 for cat in CATEGORIES},
                "path": str(self.root),
                "annotations_available": False,
            }

        counts_per_category = _get_counts_per_category(self.root)
        total_images = sum(counts_per_category.values())

        # Check for annotation directory
        annotation_dir = self.root / "Annotations"
        annotations_available = annotation_dir.exists() and any(annotation_dir.rglob("*.txt"))

        if total_images == 0:
            logger.warning(
                f"ExDark dataset found at {self.root} but no images detected in category subdirectories."
            )

        return {
            "available": total_images > 0,
            "total_images": total_images,
            "categories": CATEGORIES,
            "counts_per_category": counts_per_category,
            "path": str(self.root),
            "annotations_available": annotations_available,
        }

    def get_category_images(self, category: str) -> list[Path]:
        """Return all image paths for a given category."""
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Valid: {CATEGORIES}")
        cat_dir = self.root / category
        if not cat_dir.exists():
            return []
        return sorted(
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def get_annotations(self, image_path: Path) -> list[dict[str, Any]]:
        """
        Return annotations for a given image if available.
        Looks in ExDark/Annotations/<category>/<stem>.txt
        """
        # Try standard annotation location
        annotation_dir = self.root / "Annotations"
        for cat in CATEGORIES:
            ann_file = annotation_dir / cat / f"{image_path.stem}.txt"
            if ann_file.exists():
                return _read_annotation(ann_file)
        # Try alongside image
        sibling_ann = image_path.with_suffix(".txt")
        if sibling_ann.exists():
            return _read_annotation(sibling_ann)
        return []
