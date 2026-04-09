"""
IMCFN: Image-based Malware Classification using Fine-tuned Convolutional Neural Network
Vasan et al., Computer Networks 2020.

Input: pre-generated 224×224 RGB PNG (binary → jet colormap image).
Model: VGG16 pretrained on ImageNet, Block1–4 frozen, Block5 + FC1 + FC2 unfrozen.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


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
