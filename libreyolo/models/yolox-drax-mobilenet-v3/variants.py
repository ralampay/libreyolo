"""Independent lightweight backbone ablations; legacy graphs stay unchanged.

Original project implementation using PyTorch primitives and LibreYOLO blocks.
"""

import torch
from torch import nn

from ..yolo9.drax import DraxBlock


ARCHITECTURE_VARIANTS = (
    "legacy",
    "refine-p3p4",
    "spp-p5",
    "balanced-drax",
    "pyramid-drax",
)


def validate_variant(value):
    if value not in ARCHITECTURE_VARIANTS:
        raise ValueError(
            f"Unknown architecture_variant {value!r}; choose {ARCHITECTURE_VARIANTS}"
        )
    return value


def variant_from_state(state, metadata=None):
    prefix = "backbone.backbone."
    markers = {
        "refine-p3p4": prefix + "feature_refiners.",
        "spp-p5": prefix + "p5_pool.",
        "balanced-drax": prefix + "drax_refiner.0.attention_scale",
    }
    found = frozenset(
        name
        for name, marker in markers.items()
        if any(k.startswith(marker) for k in state)
    )
    variants_by_markers = {
        frozenset(): "legacy",
        frozenset(("refine-p3p4",)): "refine-p3p4",
        frozenset(("spp-p5",)): "spp-p5",
        frozenset(("balanced-drax",)): "balanced-drax",
        frozenset(markers): "pyramid-drax",
    }
    if found not in variants_by_markers:
        raise ValueError(
            "Checkpoint contains an unsupported combination of architecture variant tensors"
        )
    inferred = variants_by_markers[found]
    if metadata is not None:
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"version", "preset"}
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
        ):
            raise ValueError(
                "Invalid backbone_variant checkpoint metadata; expected version 1 and preset"
            )
        preset = validate_variant(metadata["preset"])
        if preset != inferred:
            raise ValueError(
                "backbone_variant metadata disagrees with checkpoint tensors"
            )
    return inferred


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


class PoolSpatialContext(nn.Module):
    def __init__(self, channels=960, hidden=160):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.project = nn.Sequential(
            nn.Conv2d(4 * hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.scale = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, x):
        a = self.reduce(x)
        b = self.pool(a)
        c = self.pool(b)
        d = self.pool(c)
        scale = self.scale.clamp(min=-0.1, max=0.1)
        return x + scale[None, :, None, None] * self.project(
            torch.cat((a, b, c, d), dim=1)
        )


class BalancedDraxBlock(DraxBlock):
    def __init__(self, dim=160):
        super().__init__(dim)
        nn.init.constant_(self.convnext.layer_scale, 1e-3)
        self.attention_scale = nn.Parameter(torch.full((dim,), 1e-3))

    def _fuse_deltas(self, conv_delta, attention_delta):
        attention_scale = self.attention_scale.clamp(min=-0.1, max=0.1)
        return super()._fuse_deltas(
            conv_delta, attention_scale[None, :, None, None] * attention_delta
        )
