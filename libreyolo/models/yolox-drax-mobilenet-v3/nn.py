"""Detection architecture with MobileNet feature injection."""

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from ..drax_mobilenet_v3.backbone import YOLOXBackbone
from ..yolox.nn import LibreYOLOXModel
from .variants import (
    BalancedDraxBlock,
    PoolSpatialContext,
    RefineSpatialFeatures,
    validate_variant,
)
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

    def __init__(self, out_channels, *, input_bgr=False, architecture_variant="legacy"):
        super().__init__(out_channels, input_bgr=input_bgr)
        self.architecture_variant = validate_variant(architecture_variant)
        if architecture_variant in ("refine-p3p4", "pyramid-drax"):
            self.feature_refiners = nn.ModuleList(
                [RefineSpatialFeatures(c) for c in self.feature_channels[:2]]
            )
        if architecture_variant in ("spp-p5", "pyramid-drax"):
            self.p5_pool = PoolSpatialContext()
        if architecture_variant in ("balanced-drax", "pyramid-drax"):
            self.drax_refiner = nn.Sequential(BalancedDraxBlock(160))
        if architecture_variant == "pyramid-drax":
            # The classifier's final 160 -> 960 expansion is not needed by a
            # detector. Refine its compact stride-32 map before projection.
            self.features = nn.Sequential(*list(self.features.children())[:-1])
            self.feature_channels = (40, 112, 160)
            self.adapter_down = nn.Identity()
            self.adapter_norm = nn.Identity()
            self.adapter_activation = nn.Identity()
            self.adapter_up = nn.Identity()
            self.adapter_up_norm = nn.Identity()
            self.p5_pool = PoolSpatialContext(channels=160, hidden=80)
            self.projections[2] = nn.Sequential(
                nn.Conv2d(160, self.out_channels[2], 1, bias=False),
                nn.BatchNorm2d(self.out_channels[2], eps=1e-3, momentum=0.03),
                nn.SiLU(),
            )
        self.incremental_adapters = nn.ModuleDict()
        self.incremental_adapter_enabled = False

    def forward_features(self, x):
        if (
            self.architecture_variant == "pyramid-drax"
            and x.device.type == "cuda"
            and torch.is_autocast_enabled("cuda")
        ):
            # ROCm can fault in the MobileNet/Drax mixed-precision backward
            # before YOLOX assignment. Keep the smaller backbone in FP32 while
            # the substantially larger PAN and head retain AMP acceleration.
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward_features(x.float())
        return self._forward_features(x)

    def _forward_features(self, x):
        if self.architecture_variant == "pyramid-drax":
            if self.input_bgr:
                x = x[:, [2, 1, 0]] / 255.0
            x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)
            outputs = []
            for index, layer in enumerate(self.features):
                x = layer(x)
                if index in (6, 12):
                    outputs.append(x)
            p3, p4 = (
                refiner(feature)
                for refiner, feature in zip(self.feature_refiners, outputs)
            )
            return p3, p4, self.p5_pool(self.drax_refiner(x))
        p3, p4, p5 = super().forward_features(x)
        if self.architecture_variant in ("refine-p3p4", "pyramid-drax"):
            p3, p4 = (
                refiner(feature)
                for refiner, feature in zip(self.feature_refiners, (p3, p4))
            )
        if self.architecture_variant == "spp-p5":
            p5 = self.p5_pool(p5)
        return p3, p4, p5

    def load_imagenet_weights(self):
        if self.architecture_variant != "pyramid-drax":
            return super().load_imagenet_weights()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        reference = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features.load_state_dict(reference.features[:-1].state_dict(), strict=True)

    @staticmethod
    def normalize_feature_names(features: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(features, str):
            features = features.split(",")
        normalized = tuple(
            str(name).strip().lower() for name in features if str(name).strip()
        )
        invalid = sorted(set(normalized) - set(IncrementalYOLOXBackbone.feature_names))
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
    def __init__(self, config="s", nb_classes=80, architecture_variant="legacy"):
        width = self.CONFIGS[config]["width"]
        backbone = IncrementalYOLOXBackbone(
            tuple(int(c * width) for c in (256, 512, 1024)),
            input_bgr=True,
            architecture_variant=architecture_variant,
        )
        super().__init__(config=config, nb_classes=nb_classes, backbone=backbone)

    def _apply_official_bn_hyperparams(self):
        # Keep MobileNet's pretrained epsilon and the adapter's original settings.
        mobile_modules = set(self.backbone.backbone.modules())
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) and module not in mobile_modules:
                module.eps = 1e-3
                module.momentum = 0.03
