"""
IMCFN: Image-based Malware Classification using Fine-tuned Convolutional Neural Network
Vasan et al., Computer Networks 2020.

Pipeline:
  1. Read malware binary as vector of 8-bit unsigned integers.
  2. Reshape into 2D array (width fixed by file-size heuristic, height varies).
  3. Apply colormap (jet) → 3-channel color image.
  4. Resize to 224×224.
  5. Pass through VGG16 pretrained on ImageNet with only Block5 + FC1 + FC2 unfrozen.

Width heuristic from Table 1 of the paper:
  < 10 KB     → 32
  10–30 KB    → 64
  30–60 KB    → 128
  60–100 KB   → 256
  100–200 KB  → 384
  200–500 KB  → 512
  500–1000 KB → 768
  > 1000 KB   → 1024
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Binary → color image conversion
# ---------------------------------------------------------------------------

def _get_width(n_bytes: int) -> int:
    """Return image width for a file of n_bytes bytes (Table 1 of IMCFN paper)."""
    kb = n_bytes / 1024
    if kb < 10:
        return 32
    elif kb < 30:
        return 64
    elif kb < 60:
        return 128
    elif kb < 100:
        return 256
    elif kb < 200:
        return 384
    elif kb < 500:
        return 512
    elif kb < 1000:
        return 768
    else:
        return 1024


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """
    Apply jet colormap to a float array in [0, 1].
    Returns uint8 array of shape (*values.shape, 3).
    Equivalent to matplotlib's 'jet' but without the matplotlib dependency
    during inference.
    """
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def binary_to_image(byte_array: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """
    Convert a 1-D uint8 byte array to a (3, target_size, target_size) float tensor
    normalized to ImageNet statistics.

    Args:
        byte_array: 1-D numpy array of uint8 (raw file bytes).
        target_size: resize both dimensions to this value (paper: 224).

    Returns:
        Tensor of shape (3, target_size, target_size), dtype float32.
    """
    n = len(byte_array)
    width = _get_width(n)
    # Trim so length is a multiple of width
    trimmed = byte_array[: (n // width) * width]
    if len(trimmed) == 0:
        # Edge case: file shorter than one row → pad to one row
        trimmed = np.pad(byte_array, (0, width - len(byte_array) % width))
        if len(trimmed) == 0:
            trimmed = np.zeros(width, dtype=np.uint8)

    gray2d = trimmed.reshape(-1, width).astype(np.float32) / 255.0  # [0, 1]
    color = _jet_colormap(gray2d)  # (H, W, 3) uint8

    pil_img = Image.fromarray(color, mode="RGB")
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)

    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = TF.to_tensor(pil_img)   # (3, H, W) float in [0, 1]
    tensor = (tensor - mean) / std
    return tensor


# ---------------------------------------------------------------------------
# IMCFN model
# ---------------------------------------------------------------------------

class IMCFN(nn.Module):
    """
    Fine-tuned VGG16 for malware image classification (IMCFN).

    Frozen layers  : Block1–Block4 (features[0:24])
    Unfrozen layers: Block5 (features[24:]), FC1, FC2 (classifier[0:6])
    New head       : Dropout + Linear(4096 → num_classes)

    The paper unfreezes FC1, FC2, and Block5 only, and fine-tunes via
    back-propagation with SGD / Adam.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5, image_size: int = 224):
        super().__init__()
        self.image_size = image_size

        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze Block1–Block4 (first 24 feature layers)
        for i, layer in enumerate(vgg.features):
            if i < 24:
                for p in layer.parameters():
                    p.requires_grad = False

        self.features = vgg.features          # Block1–Block5
        self.avgpool  = vgg.avgpool           # AdaptiveAvgPool2d(7×7)

        # VGG16 classifier: 0=Linear(512*7*7,4096), 1=ReLU, 2=Dropout,
        #                    3=Linear(4096,4096), 4=ReLU, 5=Dropout, 6=Linear(4096,1000)
        # We keep FC1+FC2 (layers 0–5) and replace the head.
        self.fc1 = vgg.classifier[0]   # Linear 512*49 → 4096
        self.relu1 = vgg.classifier[1]
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = vgg.classifier[3]   # Linear 4096 → 4096
        self.relu2 = vgg.classifier[4]
        self.drop2 = nn.Dropout(dropout)
        self.head  = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) normalised float."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.head(x)
