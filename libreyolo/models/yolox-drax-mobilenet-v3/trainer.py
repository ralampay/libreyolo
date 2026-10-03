"""YOLOX recipe for the Drax MobileNetV3 Large variant."""

import logging
import math
from dataclasses import dataclass

from torch import nn

from ...training.config import YOLOXConfig
from ..yolox.trainer import YOLOXTrainer
from .incremental_adapters import normalize_incremental_adapter_type

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class YOLOXDraxMobileNetV3LargeConfig(YOLOXConfig):
    """Family-scoped stability and incremental-adapter training configuration."""

    clip_max_norm: float = 1.0

    incremental_adapter: bool = False
    incremental_adapter_train_only: bool = False
    incremental_adapter_type: str | None = None
    incremental_adapter_reduction: int = 16
    incremental_adapter_spatial: bool = True
    incremental_adapter_alpha: float = 1.0
    incremental_adapter_train_head: bool = False
    incremental_adapter_features: str = "p3,p4,p5"


class YOLOXDraxMobileNetV3LargeTrainer(YOLOXTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_incremental_adapters()

    @classmethod
    def _config_class(cls):
        return YOLOXDraxMobileNetV3LargeConfig

    def _configure_incremental_adapters(self) -> None:
        enabled = bool(self.config.incremental_adapter)
        train_only = bool(self.config.incremental_adapter_train_only)
        train_head = bool(self.config.incremental_adapter_train_head)
        if train_only and not enabled:
            raise ValueError(
                "incremental_adapter_train_only=True requires incremental_adapter=True"
            )
        if train_only and not getattr(
            self.wrapper_model, "_has_detector_weights", False
        ):
            raise ValueError(
                "incremental_adapter_train_only=True requires a loaded foundation "
                "detector checkpoint"
            )
        if train_head and not train_only:
            raise ValueError(
                "incremental_adapter_train_head=True requires "
                "incremental_adapter_train_only=True"
            )
        if int(self.config.incremental_adapter_reduction) < 1:
            raise ValueError("incremental_adapter_reduction must be >= 1")
        if not math.isfinite(float(self.config.incremental_adapter_alpha)):
            raise ValueError("incremental_adapter_alpha must be finite")

        backbone = self.model.backbone.backbone
        if enabled:
            current = backbone.incremental_adapter_config()
            adapter_type = self.config.incremental_adapter_type or (
                current["type"] if current["features"] else "conv_bottleneck"
            )
            adapter_type = normalize_incremental_adapter_type(adapter_type)
            if backbone.incremental_adapters:
                if current["type"] != adapter_type:
                    raise ValueError(
                        "IncrementalAdapters are already attached with type "
                        f"{current['type']!r}; load a foundation checkpoint to "
                        f"attach {adapter_type!r}."
                    )
            else:
                backbone.attach_incremental_adapters(
                    adapter_type=adapter_type,
                    reduction=int(self.config.incremental_adapter_reduction),
                    spatial=bool(self.config.incremental_adapter_spatial),
                    alpha=float(self.config.incremental_adapter_alpha),
                    features=self.config.incremental_adapter_features,
                )
            backbone.enable_incremental_adapters()
            if self.wrapper_model is not None:
                self.wrapper_model._incremental_adapter_spec = (
                    backbone.incremental_adapter_config()
                )
        elif backbone.incremental_adapters:
            backbone.disable_incremental_adapters()

    def _apply_freeze_config(self) -> None:
        if not self.config.incremental_adapter_train_only:
            super()._apply_freeze_config()
            if not self.config.incremental_adapter:
                for parameter in (
                    self.model.backbone.backbone.incremental_adapters.parameters()
                ):
                    parameter.requires_grad_(False)
            return
        if getattr(self.config, "freeze", None):
            raise ValueError(
                "freeze cannot be combined with incremental_adapter_train_only; "
                "the adapter mode defines the complete trainable parameter set"
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        backbone = self.model.backbone.backbone
        for parameter in backbone.incremental_adapters.parameters():
            parameter.requires_grad_(True)
        if self.config.incremental_adapter_train_head:
            for parameter in self.model.head.parameters():
                parameter.requires_grad_(True)

        adapter_module_ids = {
            id(module) for module in backbone.incremental_adapters.modules()
        }
        self._frozen_bn_modules = tuple(
            module
            for module in self.model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and id(module) not in adapter_module_ids
        )
        self._enforce_frozen_bn_eval()

        report = (
            self.wrapper_model.incremental_adapter_parameter_report()
            if self.wrapper_model is not None
            else {}
        )
        logger.info(
            "IncrementalAdapter parameters: total=%s foundation=%s adapter=%s "
            "trainable=%s frozen=%s trainable_percentage=%.4f%%",
            report.get("total_parameters"),
            report.get("foundation_parameters"),
            report.get("incremental_adapter_parameters"),
            report.get("trainable_parameters"),
            report.get("frozen_parameters"),
            report.get("trainable_percentage", 0.0),
        )

    def _enforce_frozen_bn_eval(self) -> None:
        super()._enforce_frozen_bn_eval()
        if self.config.incremental_adapter_train_only:
            self.model.backbone.backbone.incremental_adapters.train()

    def get_model_family(self):
        return "yolox_drax_mobilenet_v3_large"

    def get_model_tag(self):
        return f"YOLOX-Drax-MobileNetV3-Large-{self.config.size}"

    def get_freeze_groups(self):
        backbone = self.model.backbone.backbone
        groups = [
            (f"backbone.backbone.{name}", module)
            for name, module in backbone.named_children()
        ]
        groups.extend(
            (f"backbone.{name}", module)
            for name, module in self.model.backbone.named_children()
            if name != "backbone"
        )
        groups.append(("head", self.model.head))
        return groups

    def _checkpoint_extra_metadata(self):
        if self.wrapper_model is None:
            return {}
        return self.wrapper_model._checkpoint_extra_metadata()
