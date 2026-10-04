"""Small, identity-initialized feature adapters and a convolutional LoRA update.

Implementations are original PyTorch expressions of published architectural
ideas; no third-party implementation code is copied.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _hidden(channels: int, reduction: int) -> int:
    if channels < 1 or reduction < 1:
        raise ValueError("channels and reduction must be positive")
    return max(1, channels // reduction)


class BottleneckAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, alpha: float = 1.0):
        super().__init__()
        hidden = _hidden(channels, reduction)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.activation = nn.SiLU()
        self.up = nn.Conv2d(hidden, channels, 1)
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.alpha * self.up(self.activation(self.down(x)))


class SSFAdapter(nn.Module):
    def __init__(self, channels: int, **_):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta


class ConvpassAdapter(nn.Module):
    """Convolutional bypass inspired by Convpass, adapted to CNN features."""

    def __init__(self, channels: int, reduction: int = 8, alpha: float = 1.0):
        super().__init__()
        hidden = _hidden(channels, reduction)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.up = nn.Conv2d(hidden, channels, 1)
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.alpha * self.up(F.silu(self.spatial(F.silu(self.down(x)))))


class ConvAdapter(nn.Module):
    """Depthwise spatial adaptation with compressed channel mixing."""

    def __init__(self, channels: int, reduction: int = 8, alpha: float = 1.0):
        super().__init__()
        hidden = _hidden(channels, reduction)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.alpha * self.up(F.silu(self.spatial(F.silu(self.down(x)))))


class DraxAdapter(nn.Module):
    """Proposed compressed two-scale adapter inspired by LibreYOLO's DraxBlock.

The in-repo DraxBlock fuses local ConvNeXt and global attention deltas.
Here a dilated depthwise branch approximates wider context cheaply; a learned
gate fuses it with a local depthwise branch before a zero-initialized residual.
"""

    def __init__(self, channels: int, reduction: int = 8, alpha: float = 1.0):
        super().__init__()
        hidden = _hidden(channels, reduction)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        self.up = nn.Conv2d(hidden, channels, 1)
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = F.silu(self.down(x))
        weights = self.fusion_logits.softmax(0)
        z = weights[0] * self.local(z) + weights[1] * self.context(z)
        return x + self.alpha * self.up(F.silu(z))


class LoRAConv2d(nn.Module):
    """Exact low-rank additive update to a frozen 1x1 convolution kernel."""

    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        if base.kernel_size != (1, 1) or base.groups != 1 or rank < 1:
            raise ValueError("LoRA requires a dense 1x1 convolution and positive rank")
        self.base = base
        self.a = nn.Conv2d(base.in_channels, rank, 1, stride=base.stride, bias=False)
        self.b = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** .5)
        nn.init.zeros_(self.b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x))


_REGISTRY = {
    "bottleneck": BottleneckAdapter,
    "ssf": SSFAdapter,
    "convpass": ConvpassAdapter,
    "conv-adapter": ConvAdapter,
    "drax": DraxAdapter,
}


def available_adapters() -> tuple[str, ...]:
    return (*_REGISTRY, "lora", "drax-hybrid")


def create_adapter(name: str, channels: int, *, reduction: int = 8, alpha: float = 1.0) -> nn.Module:
    try:
        adapter = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown feature adapter {name!r}; choose from {tuple(_REGISTRY)}") from exc
    return adapter(channels, reduction=reduction, alpha=alpha)
