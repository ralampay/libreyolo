"""Family-local registry for YOLOX-Drax-MobileNetV3 incremental adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

import torch
from torch import nn


class RegisteredIncrementalAdapter(nn.Module, ABC):
    """Contract implemented by built-in incremental adapter architectures."""

    adapter_type: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        channels: int,
        *,
        reduction: int,
        minimum_channels: int,
        spatial: bool,
        alpha: float,
        hidden_channels: int | None = None,
    ) -> RegisteredIncrementalAdapter:
        """Construct one feature-scale adapter from normalized settings."""

    @classmethod
    @abstractmethod
    def matches_state_dict(
        cls, state_dict: Mapping[str, torch.Tensor], prefix: str, features: tuple[str, ...]
    ) -> bool:
        """Return whether a raw state dict contains this adapter architecture."""

    @classmethod
    @abstractmethod
    def config_from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        features: tuple[str, ...],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recover the settings required before strict state-dict loading."""

    @classmethod
    @abstractmethod
    def config_from_modules(cls, modules: nn.ModuleDict) -> dict[str, Any]:
        """Serialize architecture-specific settings for checkpoint metadata."""

    @abstractmethod
    def set_alpha(self, alpha: float) -> None:
        """Update the residual scale without rebuilding the adapter."""

    @abstractmethod
    def topology_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Return state names and shapes that define structural compatibility."""


class ConvolutionalBottleneckIncrementalAdapter(RegisteredIncrementalAdapter):
    """Residual convolutional PEFT adapter inspired by YOLO-Adapter.

    This is an independent project-specific implementation, not a reproduction
    of that paper's architecture. See ``docs/INCREMENTAL_ADAPTERS.md``.
    """

    adapter_type = "conv_bottleneck"

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 16,
        minimum_channels: int = 8,
        spatial: bool = True,
        alpha: float = 1.0,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        if minimum_channels < 1:
            raise ValueError("minimum_channels must be >= 1")
        hidden = (
            int(hidden_channels)
            if hidden_channels is not None
            else max(channels // reduction, minimum_channels)
        )
        if hidden < 1:
            raise ValueError("hidden_channels must be >= 1")

        self.channels = int(channels)
        self.hidden_channels = hidden
        self.reduction = int(reduction)
        self.minimum_channels = int(minimum_channels)
        self.spatial_enabled = bool(spatial)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1)
        self.activation = nn.SiLU()
        self.spatial = (
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
            )
            if spatial
            else nn.Identity()
        )
        self.up = nn.Conv2d(hidden, channels, kernel_size=1)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        # A fresh enabled adapter is an exact residual identity at construction.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @classmethod
    def from_config(
        cls,
        channels: int,
        *,
        reduction: int,
        minimum_channels: int,
        spatial: bool,
        alpha: float,
        hidden_channels: int | None = None,
    ) -> ConvolutionalBottleneckIncrementalAdapter:
        return cls(
            channels,
            reduction=reduction,
            minimum_channels=minimum_channels,
            spatial=spatial,
            alpha=alpha,
            hidden_channels=hidden_channels,
        )

    @classmethod
    def matches_state_dict(
        cls, state_dict: Mapping[str, torch.Tensor], prefix: str, features: tuple[str, ...]
    ) -> bool:
        return any(f"{prefix}{name}.down.weight" in state_dict for name in features)

    @classmethod
    def config_from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        features: tuple[str, ...],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        hidden_channels = {
            name: int(state_dict[f"{prefix}{name}.down.weight"].shape[0])
            for name in features
        }
        first = features[0]
        channels = int(state_dict[f"{prefix}{first}.down.weight"].shape[1])
        hidden = hidden_channels[first]
        alpha_tensor = state_dict.get(f"{prefix}{first}.alpha")
        alpha = (
            float(alpha_tensor.item())
            if alpha_tensor is not None
            else float(metadata.get("alpha", 1.0))
        )
        spatial = bool(
            metadata.get(
                "spatial",
                any(
                    f"{prefix}{name}.spatial.weight" in state_dict
                    for name in features
                ),
            )
        )
        return {
            "reduction": int(metadata.get("reduction", max(1, channels // hidden))),
            "minimum_channels": int(
                metadata.get("minimum_channels", min(hidden_channels.values()))
            ),
            "spatial": spatial,
            "alpha": alpha,
            "hidden_channels": hidden_channels,
        }

    @classmethod
    def config_from_modules(cls, modules: nn.ModuleDict) -> dict[str, Any]:
        features = tuple(modules)
        first = modules[features[0]]
        if not isinstance(first, cls):
            raise TypeError("Incremental adapter registry/module type mismatch")
        return {
            "reduction": int(first.reduction),
            "minimum_channels": int(first.minimum_channels),
            "spatial": bool(first.spatial_enabled),
            "alpha": float(first.alpha.item()),
            "hidden_channels": {
                name: int(modules[name].hidden_channels) for name in features
            },
        }

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(alpha))

    def topology_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (key, tuple(value.shape))
            for key, value in self.state_dict().items()
            if key != "alpha"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.down(x))
        if self.spatial_enabled:
            z = self.activation(self.spatial(z))
        return x + self.alpha.to(dtype=x.dtype) * self.up(z)


_INCREMENTAL_ADAPTER_REGISTRY = MappingProxyType({
    ConvolutionalBottleneckIncrementalAdapter.adapter_type: (
        ConvolutionalBottleneckIncrementalAdapter
    ),
})


def normalize_incremental_adapter_type(adapter_type: str) -> str:
    normalized = str(adapter_type).strip().lower().replace("-", "_")
    if normalized not in _INCREMENTAL_ADAPTER_REGISTRY:
        available = ", ".join(available_incremental_adapter_types())
        raise ValueError(
            f"Unknown incremental_adapter_type {adapter_type!r}; available: {available}"
        )
    return normalized


def available_incremental_adapter_types() -> tuple[str, ...]:
    return tuple(_INCREMENTAL_ADAPTER_REGISTRY)


def get_incremental_adapter_class(
    adapter_type: str,
) -> type[RegisteredIncrementalAdapter]:
    return _INCREMENTAL_ADAPTER_REGISTRY[
        normalize_incremental_adapter_type(adapter_type)
    ]


def infer_incremental_adapter_type(
    state_dict: Mapping[str, torch.Tensor], prefix: str, features: tuple[str, ...]
) -> str:
    matches = [
        name
        for name, adapter_class in _INCREMENTAL_ADAPTER_REGISTRY.items()
        if adapter_class.matches_state_dict(state_dict, prefix, features)
    ]
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(matches)
        raise RuntimeError(
            "Could not uniquely identify IncrementalAdapter architecture from "
            f"the raw state dict; matches: {detail}. Load a metadata-bearing checkpoint."
        )
    return matches[0]


# Backward-compatible implementation name used by the first release and tests.
IncrementalAdapter = ConvolutionalBottleneckIncrementalAdapter


__all__ = [
    "ConvolutionalBottleneckIncrementalAdapter",
    "IncrementalAdapter",
    "RegisteredIncrementalAdapter",
    "available_incremental_adapter_types",
    "get_incremental_adapter_class",
    "infer_incremental_adapter_type",
    "normalize_incremental_adapter_type",
]
