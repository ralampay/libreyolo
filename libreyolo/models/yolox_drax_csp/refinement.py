"""Spatial residual refinement for the CSP-Drax backbone."""

import torch
from torch import nn


class RefineSpatialFeatures(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.update = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 2 * channels, 1),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, x):
        scale = self.scale.clamp(min=-0.1, max=0.1)
        return x + scale[None, :, None, None] * self.update(x)
