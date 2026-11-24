"""Image Preprocessing Utilities for Zero-DCE Training & Inference.

Provides normalisation, denormalisation, resizing, and data-augmentation
helpers used throughout the ml/ package.

All tensor operations assume:
  - Input PIL Images are RGB.
  - Tensors are (C, H, W) float32 in [0, 1] unless otherwise noted.
"""

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

# -------------------------------------------------------------------------------------
# Type alias
# -------------------------------------------------------------------------------------
TensorOrPIL = Union[torch.Tensor, Image.Image]


# -------------------------------------------------------------------------------------
# Core Normalisation
# -------------------------------------------------------------------------------------

def normalize_image(
    img: Union[Image.Image, np.ndarray],
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Convert a PIL Image or uint8 NumPy array to a normalised float32 tensor.

    Args:
        img:  PIL Image (RGB) or uint8 NumPy array (H, W, 3).
        mean: Per-channel mean for normalisation (default: no shift).
        std:  Per-channel std for normalisation (default: no scale).

    Returns:
        Tensor of shape (3, H, W) with values in [0, 1] after /255,
        then normalised by (pixel - mean) / std per channel.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean_t) / (std_t + 1e-8)

    return tensor  # (3, H, W)


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Image.Image:
    """Convert a normalised float tensor back to a PIL RGB Image.

    Args:
        tensor: (3, H, W) or (1, 3, H, W) float32 tensor.
        mean:   The mean used during normalisation.
        std:    The std used during normalisation.

    Returns:
        PIL Image (RGB) with uint8 pixel values.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # (3, H, W)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor * (std_t + 1e-8) + mean_t  # reverse normalisation

    tensor = torch.clamp(tensor, 0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# -------------------------------------------------------------------------------------
# Resizing
# -------------------------------------------------------------------------------------

def resize_for_model(
    img: Union[Image.Image, torch.Tensor],
    max_size: int = 512,
    keep_aspect: bool = True,
    divisible_by: int = 1,
) -> Union[Image.Image, torch.Tensor]:
    """Resize an image so that neither dimension exceeds *max_size*.

    Args:
        img:          PIL Image or (3, H, W) tensor.
        max_size:     Maximum allowed size for the longest edge.
        keep_aspect:  If True, maintain aspect ratio (default). If False,
                      resize to (max_size, max_size).
        divisible_by: Round the resulting dimensions down to a multiple of
                      this value (useful for models requiring even dimensions).

    Returns:
        Resized image in the same type as *img*.
    """
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        img_pil = denormalize_image(img)
    else:
        img_pil = img.convert("RGB") if img.mode != "RGB" else img

    w, h = img_pil.size

    if keep_aspect:
        scale = min(max_size / max(w, h), 1.0)  # never upscale
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = new_h = max_size

    if divisible_by > 1:
        new_w = max(divisible_by, (new_w // divisible_by) * divisible_by)
        new_h = max(divisible_by, (new_h // divisible_by) * divisible_by)

    resized = img_pil.resize((new_w, new_h), Image.LANCZOS)

    if is_tensor:
        return normalize_image(resized)
    return resized


def pad_to_divisible(
    tensor: torch.Tensor,
    divisible_by: int = 32,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad a (3, H, W) tensor so H and W are multiples of *divisible_by*.

    Returns:
        (padded_tensor, (pad_left, pad_right, pad_top, pad_bottom))
    """
    _, h, w = tensor.shape
    pad_h = (divisible_by - h % divisible_by) % divisible_by
    pad_w = (divisible_by - w % divisible_by) % divisible_by
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded = F.pad(tensor.unsqueeze(0), padding, mode="reflect").squeeze(0)
    return padded, padding


def unpad(
    tensor: torch.Tensor,
    padding: Tuple[int, int, int, int],
) -> torch.Tensor:
    """Remove padding from a tensor.

    Args:
        tensor:  (C, H, W) or (B, C, H, W).
        padding: (pad_left, pad_right, pad_top, pad_bottom) as returned by pad_to_divisible.
    """
    pad_left, pad_right, pad_top, pad_bottom = padding
    if tensor.ndim == 3:
        _, h, w = tensor.shape
        return tensor[
            :,
            pad_top: h - pad_bottom if pad_bottom else h,
            pad_left: w - pad_right if pad_right else w,
        ]
    else:
        _, _, h, w = tensor.shape
        return tensor[
            :,
            :,
            pad_top: h - pad_bottom if pad_bottom else h,
            pad_left: w - pad_right if pad_right else w,
        ]


# -------------------------------------------------------------------------------------
# Augmentation Helpers (for training)
# -------------------------------------------------------------------------------------

class RandomHorizontalFlipPair:
    """Randomly flip a (low, high) image pair horizontally with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, low: Image.Image, high: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            low = ImageOps.mirror(low)
            high = ImageOps.mirror(high)
        return low, high


class RandomVerticalFlipPair:
    """Randomly flip a (low, high) image pair vertically with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, low: Image.Image, high: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            low = ImageOps.flip(low)
            high = ImageOps.flip(high)
        return low, high


class RandomCropPair:
    """Randomly crop a (low, high) image pair to *crop_size × crop_size*."""

    def __init__(self, crop_size: int = 256) -> None:
        self.crop_size = crop_size

    def __call__(
        self, low: Image.Image, high: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = low.size
        cs = self.crop_size
        if w < cs or h < cs:
            # Pad if image is smaller than crop size
            pad_w = max(0, cs - w)
            pad_h = max(0, cs - h)
            low = ImageOps.expand(low, (0, 0, pad_w, pad_h))
            high = ImageOps.expand(high, (0, 0, pad_w, pad_h))
            w, h = low.size

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        box = (x, y, x + cs, y + cs)
        return low.crop(box), high.crop(box)


class RandomRotatePair:
    """Randomly rotate a (low, high) image pair by 0, 90, 180, or 270 degrees."""

    def __call__(
        self, low: Image.Image, high: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            low = low.rotate(angle, expand=True)
            high = high.rotate(angle, expand=True)
        return low, high


class ColorJitterSingle:
    """Apply mild colour jitter to a single PIL image (low-light input only)."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        if self.brightness > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img


class TrainingTransformPair:
    """Compose a standard set of augmentations for (low, high) image pairs.

    Pipeline:
        1. RandomCropPair
        2. RandomHorizontalFlipPair
        3. RandomVerticalFlipPair  (optional)
        4. RandomRotatePair        (optional)
        5. Normalise both to tensors in [0, 1]
    """

    def __init__(
        self,
        crop_size: int = 256,
        hflip: bool = True,
        vflip: bool = False,
        rotate: bool = False,
    ) -> None:
        self.crop = RandomCropPair(crop_size)
        self.hflip = RandomHorizontalFlipPair() if hflip else None
        self.vflip = RandomVerticalFlipPair() if vflip else None
        self.rotate = RandomRotatePair() if rotate else None

    def __call__(
        self, low: Image.Image, high: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low, high = self.crop(low, high)
        if self.hflip:
            low, high = self.hflip(low, high)
        if self.vflip:
            low, high = self.vflip(low, high)
        if self.rotate:
            low, high = self.rotate(low, high)
        return normalize_image(low), normalize_image(high)


class InferenceTransform:
    """Minimal transform for inference: resize then normalise.

    Args:
        max_size:     Longest edge limit (default 512).
        divisible_by: Pad so H, W are multiples of this (default 1 = no pad).
    """

    def __init__(self, max_size: int = 512, divisible_by: int = 1) -> None:
        self.max_size = max_size
        self.divisible_by = divisible_by

    def __call__(
        self, img: Image.Image
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Return (tensor, original_size)."""
        orig_size = img.size  # (W, H)
        resized = resize_for_model(img, max_size=self.max_size)
        tensor = normalize_image(resized)
        if self.divisible_by > 1:
            tensor, _ = pad_to_divisible(tensor, self.divisible_by)
        return tensor, orig_size


# -------------------------------------------------------------------------------------
# Convenience I/O
# -------------------------------------------------------------------------------------

def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from disk and convert to RGB PIL Image."""
    return Image.open(Path(path)).convert("RGB")


def save_image(img: Union[Image.Image, torch.Tensor], path: Union[str, Path]) -> None:
    """Save a PIL Image or (3, H, W) tensor to disk.

    Args:
        img:  PIL Image or float tensor.
        path: Destination file path (parent dirs are created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, torch.Tensor):
        img = denormalize_image(img)
    img.save(str(path))


# -------------------------------------------------------------------------------------
# Quick sanity check
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_img = Image.new("RGB", (640, 480), color=(30, 30, 60))
    t = normalize_image(test_img)
    assert t.shape == (3, 480, 640), f"Unexpected shape: {t.shape}"

    restored = denormalize_image(t)
    assert restored.size == (640, 480)

    small = resize_for_model(test_img, max_size=256)
    assert max(small.size) <= 256, f"Resize failed: {small.size}"

    print("image_transforms.py sanity check passed ✓")
