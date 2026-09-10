"""Detection architecture with MobileNet feature injection."""

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from ..drax_mobilenet_v3.backbone import YOLOXBackbone
from ..yolox.nn import LibreYOLOXModel


class IncrementalAdapter(nn.Module):
    """Residual convolutional PEFT adapter inspired by YOLO-Adapter.

    This is an independent project-specific implementation, not a reproduction
    of that paper's architecture. See ``docs/INCREMENTAL_ADAPTERS.md``.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.down(x))
        if self.spatial_enabled:
            z = self.activation(self.spatial(z))
        return x + self.alpha.to(dtype=x.dtype) * self.up(z)


class IncrementalYOLOXBackbone(YOLOXBackbone):
    """Drax backbone with optional adapters at the YOLOX PAN input boundary."""

    feature_names = ("p3", "p4", "p5")

    def __init__(self, out_channels, *, input_bgr=False):
        super().__init__(out_channels, input_bgr=input_bgr)
        self.incremental_adapters = nn.ModuleDict()
        self.incremental_adapter_enabled = False

    @staticmethod
    def normalize_feature_names(features: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(features, str):
            features = features.split(",")
        normalized = tuple(
            str(name).strip().lower() for name in features if str(name).strip()
        )
        invalid = sorted(
            set(normalized) - set(IncrementalYOLOXBackbone.feature_names)
        )
        if invalid:
            raise ValueError(
                "incremental_adapter_features must contain only p3, p4, p5; "
                f"got {invalid}"
            )
        if not normalized:
            raise ValueError(
                "incremental_adapter_features must select at least one feature"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("incremental_adapter_features must not contain duplicates")
        return tuple(
            name
            for name in IncrementalYOLOXBackbone.feature_names
            if name in normalized
        )

    def attach_incremental_adapters(
        self,
        *,
        reduction: int = 16,
        minimum_channels: int = 8,
        spatial: bool = True,
        alpha: float = 1.0,
        features: str | Iterable[str] = feature_names,
        hidden_channels: Mapping[str, int] | None = None,
    ) -> None:
        selected = self.normalize_feature_names(features)
        desired_hidden = dict(hidden_channels or {})
        desired = {
            name: IncrementalAdapter(
                channels,
                reduction=reduction,
                minimum_channels=minimum_channels,
                spatial=spatial,
                alpha=alpha,
                hidden_channels=desired_hidden.get(name),
            )
            for name, channels in zip(self.feature_names, self.out_channels)
            if name in selected
        }
        reference = self.projections[0][0].weight
        for module in desired.values():
            module.to(device=reference.device, dtype=reference.dtype)
            module.train(self.training)
        current = self.incremental_adapter_config()
        requested_hidden = {
            name: module.hidden_channels for name, module in desired.items()
        }
        if self.incremental_adapters:
            same = (
                tuple(current["features"]) == selected
                and bool(current["spatial"]) == bool(spatial)
                and current["hidden_channels"] == requested_hidden
            )
            if not same:
                raise ValueError(
                    "IncrementalAdapters are already attached with a different "
                    "configuration; construct or reload a fresh model for this "
                    "ablation."
                )
            for module in self.incremental_adapters.values():
                module.alpha.fill_(float(alpha))
            return
        self.incremental_adapters.update(desired)

    def incremental_adapter_config(self) -> dict:
        modules = self.incremental_adapters
        features = tuple(name for name in self.feature_names if name in modules)
        first = modules[features[0]] if features else None
        return {
            "version": 1,
            "enabled": bool(self.incremental_adapter_enabled),
            "features": list(features),
            "reduction": int(first.reduction) if first is not None else 16,
            "minimum_channels": int(first.minimum_channels) if first is not None else 8,
            "spatial": bool(first.spatial_enabled) if first is not None else True,
            "alpha": float(first.alpha.item()) if first is not None else 1.0,
            "hidden_channels": {
                name: int(modules[name].hidden_channels) for name in features
            },
        }

    def enable_incremental_adapters(self) -> None:
        if not self.incremental_adapters:
            raise RuntimeError("Attach IncrementalAdapters before enabling them")
        self.incremental_adapter_enabled = True

    def disable_incremental_adapters(self) -> None:
        self.incremental_adapter_enabled = False

    def forward(self, x):
        features = super().forward(x)
        if not self.incremental_adapter_enabled:
            return features
        return {
            dark_name: self.incremental_adapters[feature_name](feature)
            if feature_name in self.incremental_adapters
            else feature
            for feature_name, (dark_name, feature) in zip(
                self.feature_names, features.items()
            )
        }


class YOLOXDraxMobileNetV3LargeModel(LibreYOLOXModel):
    def __init__(self, config="s", nb_classes=80):
        width = self.CONFIGS[config]["width"]
        backbone = IncrementalYOLOXBackbone(
            tuple(int(c * width) for c in (256, 512, 1024)), input_bgr=True
        )
        super().__init__(config=config, nb_classes=nb_classes, backbone=backbone)

    def _apply_official_bn_hyperparams(self):
        # Keep MobileNet's pretrained epsilon and the adapter's original settings.
        mobile_modules = set(self.backbone.backbone.modules())
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) and module not in mobile_modules:
                module.eps = 1e-3
                module.momentum = 0.03
