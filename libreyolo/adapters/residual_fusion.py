"""Original residual spatial adapter with content-dependent two-scale fusion.

Implemented from project-owned equations; no third-party code is used.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import _hidden


class DraxResidualFusionConv2d(nn.Module):
    """Frozen convolution plus a normalized, residual spatial bottleneck.

    ``reduction`` controls capacity; there is no LoRA branch. Normalization is
    channel-wise at each pixel, and fusion weights vary by image and position.
    """

    def __init__(self, base: nn.Conv2d, *, reduction: int = 16, alpha: float = 1.0):
        super().__init__()
        if (not isinstance(base, nn.Conv2d) or base.kernel_size != (1, 1)
                or base.groups != 1 or base.stride != (1, 1)
                or base.padding != (0, 0)):
            raise ValueError("Drax residual fusion requires a dense stride-one unpadded 1x1 Conv2d")
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        hidden = _hidden(base.in_channels, reduction)
        self.base = base
        self.down = nn.Conv2d(base.in_channels, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.gate = nn.Conv2d(hidden, 2, 1)
        self.up = nn.Conv2d(hidden, base.out_channels, 1)
        self.alpha = float(alpha)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        if self.norm.normalized_shape[0] == 1:
            # Singleton channel normalization would erase all input variation.
            z = z * self.norm.weight.view(1, 1, 1, 1) + self.norm.bias.view(1, 1, 1, 1)
        else:
            z = self.norm(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        z = F.silu(z)
        gates = self.gate(z).softmax(dim=1)
        spatial = gates[:, :1] * self.local(z) + gates[:, 1:] * self.context(z)
        update = self.up(F.silu(z + spatial))
        return self.base(x) + self.alpha * update
