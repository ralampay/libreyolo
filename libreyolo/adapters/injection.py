"""Post-checkpoint adapter injection without changing foundation checkpoint keys."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .layers import LoRAConv2d, available_adapters, create_adapter
from .hybrid import DraxHybridConv2d


class AdaptedFeature(nn.Module):
    def __init__(self, base: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        return self.adapter(self.base(*args, **kwargs))


@dataclass(frozen=True)
class InjectionReport:
    adapter: str
    modules: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int
    total_parameters: int
    trainable_percent: float


def count_parameters(model: nn.Module) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable,
            "trainable_percent": 100 * trainable / total if total else 0.0}


def inject_adapters(model: nn.Module, adapter: str, targets: dict[str, int], *,
                    reduction: int = 8, rank: int = 8, alpha: float = 1.0,
                    train_head: bool = False) -> InjectionReport:
    """Freeze a loaded model, then attach adapters at explicit module paths.

    For LoRA, targets are dense 1x1 Conv2d paths and channel values are ignored.
    Other methods target shape-preserving, tensor-output feature modules.
    """
    if not targets:
        raise ValueError("At least one adapter target is required")
    if reduction < 1 or rank < 1:
        raise ValueError("reduction and rank must be positive")
    if adapter not in available_adapters() or not math.isfinite(alpha):
        raise ValueError("Adapter name must be registered and alpha must be finite")
    resolved = []
    for path, channels in targets.items():
        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        child = getattr(parent, child_name)
        if isinstance(child, (AdaptedFeature, LoRAConv2d, DraxHybridConv2d)):
            raise ValueError(f"Target {path} already has an adapter")
        if adapter in {"lora", "drax-hybrid"}:
            if not isinstance(child, nn.Conv2d) or child.kernel_size != (1, 1) or child.groups != 1:
                raise ValueError(f"LoRA target {path} must be a dense 1x1 Conv2d")
            if adapter == "drax-hybrid" and (child.stride != (1, 1) or child.padding != (0, 0)):
                raise ValueError(f"Drax hybrid target {path} must have stride one and no padding")
        elif channels < 1:
            raise ValueError(f"Target {path} needs a positive channel count")
        resolved.append((path, parent, child_name, child, channels))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for path, parent, child_name, child, channels in resolved:
        weight = next(child.parameters())
        if adapter == "lora":
            replacement = LoRAConv2d(child, rank=rank, alpha=alpha)
        elif adapter == "drax-hybrid":
            replacement = DraxHybridConv2d(child, rank=rank, reduction=reduction, alpha=alpha)
        else:
            replacement = AdaptedFeature(child, create_adapter(adapter, channels, reduction=reduction, alpha=alpha))
        replacement.to(device=weight.device, dtype=weight.dtype)
        setattr(parent, child_name, replacement)
    if train_head:
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    counts = count_parameters(model)
    return InjectionReport(adapter, tuple(targets), counts["trainable"], counts["frozen"],
                           counts["total"], counts["trainable_percent"])


def adapter_state_dict(model: nn.Module) -> dict:
    """Trainable tensors only, for compact adapter-specific checkpoints."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable}


def load_adapter_state_dict(model: nn.Module, state: dict) -> None:
    """Restore a trainable-only state into the matching injected architecture."""
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(f"Adapter state mismatch: missing={sorted(expected - set(state))}, "
                         f"unexpected={sorted(set(state) - expected)}")
    for name, tensor in state.items():
        if parameters[name].shape != tensor.shape:
            raise ValueError(f"Adapter parameter {name} has shape {tuple(tensor.shape)}, "
                             f"expected {tuple(parameters[name].shape)}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor.to(parameters[name].device, parameters[name].dtype))
