"""Original low-rank and compressed two-scale convolutional adaptation.

The linear path is an exact LoRA update to a frozen dense 1x1 kernel. The
nonlinear bypass is Convpass-inspired, using depthwise spatial convolutions
and channel mixing instead of a dense 3x3. Drax supplies learned fusion of
local and dilated context. No third-party implementation is copied.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import LoRAConv2d, _hidden


class DraxHybridConv2d(nn.Module):
    """Shape-preserving adaptation of a stride-one dense 1x1 convolution."""

    def __init__(self, base: nn.Conv2d, *, rank: int = 8,
                 reduction: int = 8, alpha: float = 1.0):
        super().__init__()
        if (not isinstance(base, nn.Conv2d) or base.kernel_size != (1, 1)
                or base.groups != 1 or base.stride != (1, 1)
                or base.padding != (0, 0)):
            raise ValueError("Drax hybrid requires a dense stride-one unpadded 1x1 Conv2d")
        if rank < 1 or not math.isfinite(alpha):
            raise ValueError("rank must be positive and alpha must be finite")
        hidden = _hidden(base.in_channels, reduction)
        self.linear = LoRAConv2d(base, rank=rank, alpha=alpha)
        self.down = nn.Conv2d(base.in_channels, hidden, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        self.mix = nn.Conv2d(hidden, hidden, 1)
        self.up = nn.Conv2d(hidden, base.out_channels, 1)
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = F.silu(self.down(x))
        weights = self.fusion_logits.softmax(0)
        spatial = weights[0] * self.local(z) + weights[1] * self.context(z)
        return self.linear(x) + self.alpha * self.up(F.silu(self.mix(spatial)))
