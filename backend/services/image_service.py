"""
Image Service
==============
Handles upload validation, file saving, and preprocessing for model inference.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}


class ImageServiceError(Exception):
    """Raised for image validation or processing errors."""


class ImageService:
    """Stateless image service (upload, validate, preprocess)."""

    def __init__(
        self,
        upload_dir: Path | None = None,
        max_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ):
        self.upload_dir = Path(upload_dir) if upload_dir else UPLOAD_DIR
        self.max_size_bytes = max_size_bytes
        self.upload_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Standalone functions (used by routes directly)
# ---------------------------------------------------------------------------

async def validate_image(file) -> None:
    """
    Validate that the uploaded file is a valid image.
    Raises ImageServiceError on failure.

    Checks:
      - File extension (jpg/jpeg/png)
      - Content type
      - File size ≤ 10 MB
      - Readable by PIL (actual image data)
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ImageServiceError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    content_type = getattr(file, "content_type", "") or ""
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise ImageServiceError(
            f"Unsupported content type '{content_type}'. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    # Read bytes for size + PIL validation
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise ImageServiceError(
            f"File size {len(contents) / (1024*1024):.1f} MB exceeds maximum 10 MB."
        )
    if len(contents) == 0:
        raise ImageServiceError("Uploaded file is empty.")

    # Validate actual image data with PIL
    try:
        import io
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()
    except Exception as e:
        raise ImageServiceError(f"Invalid or corrupt image file: {e}")

    # Reset so caller can read again
    await file.seek(0)


async def save_upload(file) -> dict[str, Any]:
    """
    Save an uploaded file to the uploads directory.

    Returns:
        {
            filename: str,
            filepath: str,
            dimensions: [width, height],
            file_size: int (bytes),
            preview_url: str,
        }
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_filename = file.filename or "image.jpg"
    suffix = Path(original_filename).suffix.lower() or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    dest_path = UPLOAD_DIR / unique_name

    contents = await file.read()
    dest_path.write_bytes(contents)

    # Get image dimensions
    import io
    with Image.open(io.BytesIO(contents)) as img:
        width, height = img.size

    file_size = len(contents)
    preview_url = f"/images/uploads/{unique_name}"

    logger.info(f"Uploaded image saved: {unique_name} ({file_size} bytes, {width}x{height})")

    return {
        "filename": unique_name,
        "original_filename": original_filename,
        "filepath": str(dest_path),
        "dimensions": [width, height],
        "file_size": file_size,
        "preview_url": preview_url,
    }


def preprocess_for_model(image_path: str | Path):
    """
    Load and preprocess an image for model inference.

    Args:
        image_path: Path to the image file.

    Returns:
        Normalized float32 tensor or numpy array of shape [1, 3, H, W] with values in [0, 1].
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # [0, 1]

    if TORCH_AVAILABLE and torch is not None:
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    # NumPy format (1, 3, H, W)
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)
