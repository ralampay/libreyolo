"""Detection architecture with MobileNet feature injection."""

from collections.abc import Iterable, Mapping

from torch import nn

from ..drax_mobilenet_v3.backbone import YOLOXBackbone
from ..yolox.nn import LibreYOLOXModel
from .incremental_adapters import (
    IncrementalAdapter,
    RegisteredIncrementalAdapter,
    get_incremental_adapter_class,
    normalize_incremental_adapter_type,
)

__all__ = [
    "IncrementalAdapter",
    "IncrementalYOLOXBackbone",
    "YOLOXDraxMobileNetV3LargeModel",
]


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
        adapter_type: str = "conv_bottleneck",
        features: str | Iterable[str] = feature_names,
        hidden_channels: Mapping[str, int] | None = None,
    ) -> None:
        selected = self.normalize_feature_names(features)
        normalized_type = normalize_incremental_adapter_type(adapter_type)
        adapter_class = get_incremental_adapter_class(normalized_type)
        desired_hidden = dict(hidden_channels or {})
        desired = {
            name: adapter_class.from_config(
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
        if self.incremental_adapters:
            current_shapes = {
                name: module.topology_signature()
                for name, module in self.incremental_adapters.items()
                if isinstance(module, RegisteredIncrementalAdapter)
            }
            desired_shapes = {
                name: module.topology_signature()
                for name, module in desired.items()
                if isinstance(module, RegisteredIncrementalAdapter)
            }
            same = (
                current["type"] == normalized_type
                and tuple(current["features"]) == selected
                and current_shapes == desired_shapes
            )
            if not same:
                raise ValueError(
                    "IncrementalAdapters are already attached with a different "
                    "configuration; construct or reload a fresh model for this "
                    "ablation."
                )
            for module in self.incremental_adapters.values():
                if not isinstance(module, RegisteredIncrementalAdapter):
                    raise TypeError("Incremental adapter registry/module type mismatch")
                module.set_alpha(alpha)
            return
        self.incremental_adapters.update(desired)

    def incremental_adapter_config(self) -> dict:
        modules = self.incremental_adapters
        features = tuple(name for name in self.feature_names if name in modules)
        first = modules[features[0]] if features else None
        adapter_type = (
            first.adapter_type
            if isinstance(first, RegisteredIncrementalAdapter)
            else "conv_bottleneck"
        )
        config = {
            "version": 1,
            "type": adapter_type,
            "enabled": bool(self.incremental_adapter_enabled),
            "features": list(features),
        }
        if first is None:
            config.update(
                reduction=16,
                minimum_channels=8,
                spatial=True,
                alpha=1.0,
                hidden_channels={},
            )
            return config
        adapter_class = get_incremental_adapter_class(adapter_type)
        config.update(adapter_class.config_from_modules(modules))
        return config

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
