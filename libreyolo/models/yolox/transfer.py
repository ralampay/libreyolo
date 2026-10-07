"""Explicit classifier replacement for supervised taxonomy transfer.

Original implementation using PyTorch public APIs; no upstream code copied.
"""

import math

import torch
from torch import nn


def reset_classifiers(model, num_classes: int, *, seed: int, prior: float = 0.01):
    """Replace only class predictors, including when class counts are equal.

    Call after strict foundation loading and before freezing/injection. Existing
    objectness, localization and feature tensors are retained. CPU initialization
    makes paired seeds device independent and does not consume caller RNG state.
    """
    if num_classes < 1 or not 0 < prior < 1:
        raise ValueError("num_classes must be positive and prior must be in (0, 1)")
    head = model.head
    replacements = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for old in head.cls_preds:
            new = nn.Conv2d(old.in_channels, num_classes, old.kernel_size,
                            old.stride, old.padding, bias=True)
            nn.init.constant_(new.bias, -math.log((1 - prior) / prior))
            replacements.append(new.to(device=old.weight.device, dtype=old.weight.dtype))
    head.cls_preds = nn.ModuleList(replacements)
    head.num_classes = num_classes
    if hasattr(model, "nb_classes"):
        model.nb_classes = num_classes
    return {name: value.detach().cpu().clone() for name, value in head.cls_preds.state_dict().items()}


def transfer_state_dict(model):
    """Compact trainable state plus the full head, including BatchNorm buffers."""
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name in names or name.startswith("head.")}


def load_transfer_state_dict(model, state):
    """Load into an identically reset/injected/frozen model without silent omissions."""
    expected = transfer_state_dict(model)
    if set(state) != set(expected):
        raise ValueError(f"Transfer state keys differ: missing={set(expected)-set(state)}, unexpected={set(state)-set(expected)}")
    current = model.state_dict()
    for name, value in state.items():
        if value.shape != current[name].shape:
            raise ValueError(f"Transfer state shape mismatch: {name}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value)
