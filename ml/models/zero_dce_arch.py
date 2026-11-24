"""Zero-DCE Architecture - Reference Implementation
   Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
   Based on: Guo et al., CVPR 2020 (https://arxiv.org/abs/2001.06826)

   Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"
   Authors: Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou,
            Sam Kwong, Runmin Cong
   CVPR 2020

Usage:
    from ml.models.zero_dce_arch import DCENet, apply_enhancement_curves, load_zero_dce
    model = load_zero_dce(weights_path='checkpoints/zero_dce.pth', device='cpu')
    # model is ready for inference
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DCENet(nn.Module):
    """Deep Curve Estimation Network (Zero-DCE).

    Architecture:
        - 7 convolutional layers with ReLU activations
        - Symmetric skip connections between conv1↔conv6 and conv2↔conv5
        - Input:  (B, 3, H, W)  — RGB image in [0, 1]
        - Output: (B, 24, H, W) — 8 curve maps × 3 channels (A_1 … A_8)

    Args:
        in_channels:    Number of input channels (default: 3 for RGB).
        num_filters:    Number of feature maps in each conv layer (default: 32).
        num_iterations: Number of enhancement iterations n (default: 8).
                        The model outputs 3*n channels representing n curve maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 32,
        num_iterations: int = 8,
    ) -> None:
        super().__init__()

        self.num_iterations = num_iterations
        out_channels = in_channels * num_iterations  # 3 × 8 = 24

        # ---- Encoder ----------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)

        # ---- Decoder (symmetric skip connections) -----------------------------------
        # conv5 receives: conv4 output concatenated with conv3 output  → 64 → 32
        self.conv5 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        # conv6 receives: conv5 output concatenated with conv2 output  → 64 → 32
        self.conv6 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        # conv7 receives: conv6 output concatenated with conv1 output  → 64 → out_channels
        self.conv7 = nn.Conv2d(num_filters * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self._init_weights()

    # ---------------------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming initialisation for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W) with values in [0, 1].

        Returns:
            curves: Curve parameter tensor of shape (B, 24, H, W) in [-1, 1].
                    Each group of 3 channels is one A map for one iteration.
        """
        # Encoder
        x1 = self.relu(self.conv1(x))   # (B, 32, H, W)
        x2 = self.relu(self.conv2(x1))  # (B, 32, H, W)
        x3 = self.relu(self.conv3(x2))  # (B, 32, H, W)
        x4 = self.relu(self.conv4(x3))  # (B, 32, H, W)

        # Decoder with skip connections
        x5 = self.relu(self.conv5(torch.cat([x4, x3], dim=1)))  # (B, 32, H, W)
        x6 = self.relu(self.conv6(torch.cat([x5, x2], dim=1)))  # (B, 32, H, W)
        # Final layer: Tanh → curve values in [-1, 1]
        curves = self.tanh(self.conv7(torch.cat([x6, x1], dim=1)))  # (B, 24, H, W)
        return curves


# -------------------------------------------------------------------------------------
def apply_enhancement_curves(
    image: torch.Tensor,
    curves: torch.Tensor,
    num_iterations: int = 8,
) -> torch.Tensor:
    """Apply iterative curve enhancement using the Zero-DCE formula.

    Enhancement formula (per-pixel, per-channel):
        LE(x, A) = x + A * x * (1 - x)
    Applied iteratively for n iterations using the n learned curve maps A_i.

    This is equivalent to the quadratic curve:
        LE(x) = x + A*(x − x²)

    Args:
        image:          Input tensor (B, 3, H, W) in [0, 1].
        curves:         Output of DCENet, shape (B, 3*n, H, W).
        num_iterations: Number of enhancement steps n (default: 8).

    Returns:
        Enhanced image tensor (B, 3, H, W) clamped to [0, 1].
    """
    enhanced = image
    for i in range(num_iterations):
        # Extract the i-th curve map (3 channels)
        A_i = curves[:, i * 3:(i + 1) * 3, :, :]
        # LE(x, A) = x + A * x * (1 - x)
        enhanced = enhanced + A_i * (enhanced - enhanced ** 2)

    return torch.clamp(enhanced, 0.0, 1.0)


# -------------------------------------------------------------------------------------
def load_zero_dce(
    weights_path: Optional[str] = None,
    device: str = "cpu",
    num_filters: int = 32,
    num_iterations: int = 8,
) -> DCENet:
    """Instantiate DCENet and optionally load pretrained weights.

    Args:
        weights_path:   Path to a .pth checkpoint file.  If None or the file
                        does not exist, the model is returned with random
                        initialisation and a warning is emitted.
        device:         Torch device string, e.g. ``'cpu'`` or ``'cuda:0'``.
        num_filters:    Number of convolutional filters (must match checkpoint).
        num_iterations: Number of curve iterations (must match checkpoint).

    Returns:
        DCENet model in ``eval()`` mode on the requested device.
    """
    model = DCENet(in_channels=3, num_filters=num_filters, num_iterations=num_iterations)
    model = model.to(device)

    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.is_file():
            try:
                state_dict = torch.load(weights_path, map_location=device)
                # Support checkpoints that wrap weights under a 'state_dict' key
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=True)
                logger.info("Zero-DCE weights loaded from '%s'.", weights_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load Zero-DCE weights from '%s': %s. "
                    "Using untrained model.",
                    weights_path,
                    exc,
                )
        else:
            logger.warning(
                "Zero-DCE weights file not found at '%s'. "
                "Running with untrained model — results will be poor.",
                weights_path,
            )
    else:
        logger.warning(
            "No weights_path provided for Zero-DCE. "
            "Running with randomly initialised model."
        )

    model.eval()
    return model


# -------------------------------------------------------------------------------------
# Quick sanity check when run directly
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = load_zero_dce(weights_path=None, device="cpu")
    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        curves = model(dummy)
        enhanced = apply_enhancement_curves(dummy, curves, num_iterations=model.num_iterations)
    print(f"Input shape:    {dummy.shape}")
    print(f"Curves shape:   {curves.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Enhanced range: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
    print("Architecture check passed ✓")
