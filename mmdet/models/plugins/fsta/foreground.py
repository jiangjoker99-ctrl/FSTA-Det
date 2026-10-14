"""Foreground prediction for FSTA feature gates."""

import torch.nn as nn


class ForegroundHead(nn.Module):
    """Predict a one-channel foreground logit map from an FPN feature."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        return self.layers(x)
