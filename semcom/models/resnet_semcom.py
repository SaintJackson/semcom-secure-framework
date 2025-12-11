"""
ResNet-based semantic communication model.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .base_semcom import BaseSemCom

class ResNetSemCom(BaseSemCom):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        base = resnet18(weights=None)
        layers = list(base.children())

        # Encoder = everything except final classifier
        self.encoder = nn.Sequential(*layers[:-1])
        self.project = nn.Linear(512, bottleneck_dim)

        # Decoder = small MLP classifier
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def encode(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)
        return self.project(feat)

    def decode(self, s):
        return self.decoder(s)
