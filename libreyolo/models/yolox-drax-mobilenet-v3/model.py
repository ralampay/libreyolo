"""YOLOX neck/head with a fixed Drax MobileNetV3 Large backbone.

train(pretrained=True) initializes ImageNet feature weights only.
The external YOLOX BGR 0-255 input contract is preserved.
"""

from collections.abc import Mapping
from time import perf_counter
from typing import Any, ClassVar

import torch

from ..drax_mobilenet_v3.model import DraxMobileNetVariant
from ..yolox.model import LibreYOLOX
from .incremental_adapters import (
    available_incremental_adapter_types,
    get_incremental_adapter_class,
    infer_incremental_adapter_type,
    normalize_incremental_adapter_type,
)
from .nn import IncrementalYOLOXBackbone, YOLOXDraxMobileNetV3LargeModel
from .trainer import YOLOXDraxMobileNetV3LargeConfig

_TRAIN_DEFAULTS = YOLOXDraxMobileNetV3LargeConfig()


class LibreYOLOXDraxMobileNetV3Large(DraxMobileNetVariant, LibreYOLOX):
    """Detection with a fixed Drax MobileNetV3 Large feature extractor.

    ``train(pretrained=True)`` initializes only ImageNet feature weights;
    ``False`` trains from scratch. Load detector checkpoints via model_path.
    The pretrained argument accepts bools only.
    """

    FAMILY = "yolox_drax_mobilenet_v3_large"
    FILENAME_PREFIX = "LibreYOLOXDraxMobileNetV3Large"
    CLI_ALIASES = ("yolox-drax-mobilenet-v3-large",)
    TASK_INPUT_SIZES: ClassVar = {"detect": LibreYOLOX.INPUT_SIZES}
    TRAIN_CONFIG = YOLOXDraxMobileNetV3LargeConfig

    def __init__(self, model_path=None, size="s", **kwargs):
        if isinstance(model_path, dict):
            spec = self._incremental_adapter_spec_for_state_dict(model_path)
            if spec:
                self._incremental_adapter_spec = spec
        super().__init__(model_path=model_path, size=size, **kwargs)

    def _init_model(self):
        model = YOLOXDraxMobileNetV3LargeModel(self.size, self.nb_classes)
        spec = getattr(self, "_incremental_adapter_spec", None)
        if spec:
            backbone = model.backbone.backbone
            backbone.attach_incremental_adapters(
                adapter_type=spec["type"],
                reduction=spec["reduction"],
                minimum_channels=spec["minimum_channels"],
                spatial=spec["spatial"],
                alpha=spec["alpha"],
                features=spec["features"],
                hidden_channels=spec.get("hidden_channels"),
            )
            if spec.get("enabled", False):
                backbone.enable_incremental_adapters()
        return model

    def _mobile_backbone(self):
        return self.model.backbone.backbone

    def attach_incremental_adapters(
        self,
        *,
        reduction: int = 16,
        minimum_channels: int = 8,
        spatial: bool = True,
        alpha: float = 1.0,
        adapter_type: str = "conv_bottleneck",
        features: str | tuple[str, ...] = ("p3", "p4", "p5"),
    ) -> dict[str, int | float]:
        """Attach zero-initialized P3/P4/P5 adapters without changing outputs."""
        backbone = self._mobile_backbone()
        backbone.attach_incremental_adapters(
            adapter_type=adapter_type,
            reduction=reduction,
            minimum_channels=minimum_channels,
            spatial=spatial,
            alpha=alpha,
            features=features,
        )
        self._incremental_adapter_spec = backbone.incremental_adapter_config()
        self._invalidate_cuda_graphs("attached IncrementalAdapters")
        return self.incremental_adapter_parameter_report()

    def enable_incremental_adapters(self) -> None:
        self._mobile_backbone().enable_incremental_adapters()
        self._incremental_adapter_spec = (
            self._mobile_backbone().incremental_adapter_config()
        )
        self._invalidate_cuda_graphs("enabled IncrementalAdapters")

    def disable_incremental_adapters(self) -> None:
        self._mobile_backbone().disable_incremental_adapters()
        if hasattr(self, "_incremental_adapter_spec"):
            self._incremental_adapter_spec["enabled"] = False
        self._invalidate_cuda_graphs("disabled IncrementalAdapters")

    def incremental_adapter_parameter_report(self) -> dict[str, int | float]:
        adapter_parameters = sum(
            parameter.numel()
            for parameter in self._mobile_backbone().incremental_adapters.parameters()
        )
        total_parameters = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        foundation_parameters = total_parameters - adapter_parameters
        return {
            "total_parameters": total_parameters,
            "foundation_parameters": foundation_parameters,
            "incremental_adapter_parameters": adapter_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": total_parameters - trainable_parameters,
            "trainable_percentage": (
                100.0 * trainable_parameters / total_parameters
                if total_parameters
                else 0.0
            ),
        }

    def _attach_incremental_adapters_from_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], metadata: Mapping | None = None
    ) -> None:
        spec = self._incremental_adapter_spec_for_state_dict(state_dict, metadata)
        if not spec:
            return
        self._incremental_adapter_spec = spec
        backbone = self._mobile_backbone()
        backbone.attach_incremental_adapters(
            adapter_type=spec["type"],
            reduction=spec["reduction"],
            minimum_channels=spec["minimum_channels"],
            spatial=spec["spatial"],
            alpha=spec["alpha"],
            features=spec["features"],
            hidden_channels=spec["hidden_channels"],
        )
        if spec["enabled"]:
            backbone.enable_incremental_adapters()

    @staticmethod
    def _incremental_adapter_spec_for_state_dict(
        state_dict: Mapping[str, torch.Tensor], metadata: Mapping | None = None
    ) -> dict[str, Any] | None:
        prefix = "backbone.backbone.incremental_adapters."
        features = []
        for name in IncrementalYOLOXBackbone.feature_names:
            feature_prefix = f"{prefix}{name}."
            if any(key.startswith(feature_prefix) for key in state_dict):
                features.append(name)
        if not features:
            return None

        config = dict(metadata or {})
        if config and int(config.get("version", 1)) != 1:
            raise RuntimeError(
                "Unsupported IncrementalAdapter checkpoint version: "
                f"{config.get('version')!r}"
            )
        adapter_type = (
            normalize_incremental_adapter_type(config["type"])
            if config.get("type")
            else infer_incremental_adapter_type(
                state_dict, prefix, tuple(features)
            )
        )
        adapter_class = get_incremental_adapter_class(adapter_type)
        adapter_config = adapter_class.config_from_state_dict(
            state_dict,
            prefix,
            tuple(features),
            config,
        )
        return {
            "version": 1,
            "type": adapter_type,
            "enabled": bool(config.get("enabled", True)),
            "features": features,
            **adapter_config,
        }

    @staticmethod
    def available_incremental_adapter_types() -> tuple[str, ...]:
        """Return the adapter architectures built into this model family."""
        return available_incremental_adapter_types()

    def _filter_incoming_state_dict(
        self,
        state_dict: dict,
        *,
        loaded: dict | None = None,
        checkpoint_task: str | None = None,
    ) -> dict:
        metadata = (
            loaded.get("incremental_adapters") if isinstance(loaded, dict) else None
        )
        self._attach_incremental_adapters_from_state_dict(state_dict, metadata)
        return super()._filter_incoming_state_dict(
            state_dict, loaded=loaded, checkpoint_task=checkpoint_task
        )

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        # Raw state dicts do not carry metadata, so infer only adapter structure.
        if not self._mobile_backbone().incremental_adapters:
            self._attach_incremental_adapters_from_state_dict(state_dict)
        super()._prepare_model_for_state_dict(state_dict)

    def _checkpoint_extra_metadata(self) -> dict[str, Any]:
        backbone = self._mobile_backbone()
        if not backbone.incremental_adapters:
            return {}
        return {"incremental_adapters": backbone.incremental_adapter_config()}

    def _prepare_scratch_init(self) -> None:
        super()._prepare_scratch_init()
        self._incremental_adapter_spec = None

    def _model_info_extra(self) -> dict[str, Any]:
        info = super()._model_info_extra()
        backbone = self._mobile_backbone()
        info.update(self.incremental_adapter_parameter_report())
        info["incremental_adapters"] = backbone.incremental_adapter_config()
        return info

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        if getattr(self, "_incremental_adapter_train_only_requested", False):
            raise ValueError(
                "incremental_adapter_train_only=True requires D1 to use the same "
                "class count and class indices as the loaded foundation checkpoint; "
                f"checkpoint nc={self.nb_classes}, dataset nc={new_nb_classes}."
            )
        return super()._rebuild_for_new_classes(new_nb_classes)

    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int = _TRAIN_DEFAULTS.imgsz,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        pretrained: bool = True,
        resume: bool = _TRAIN_DEFAULTS.resume,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        callbacks=None,
        loggers=None,
        incremental_adapter: bool = False,
        incremental_adapter_train_only: bool = False,
        incremental_adapter_type: str | None = None,
        incremental_adapter_reduction: int = 16,
        incremental_adapter_spatial: bool = True,
        incremental_adapter_alpha: float = 1.0,
        incremental_adapter_train_head: bool = False,
        incremental_adapter_features: str = "p3,p4,p5",
        **kwargs,
    ) -> dict:
        """Train, optionally using family-scoped incremental feature adapters."""
        if incremental_adapter_train_only and not incremental_adapter:
            raise ValueError(
                "incremental_adapter_train_only=True requires incremental_adapter=True"
            )
        attached_config = self._mobile_backbone().incremental_adapter_config()
        resolved_adapter_type = normalize_incremental_adapter_type(
            incremental_adapter_type
            or (
                attached_config["type"]
                if attached_config["features"]
                else "conv_bottleneck"
            )
        )
        if not incremental_adapter and incremental_adapter_type is not None:
            raise ValueError(
                "incremental_adapter_type requires incremental_adapter=True"
            )
        if incremental_adapter_train_head and not incremental_adapter_train_only:
            raise ValueError(
                "incremental_adapter_train_head=True requires "
                "incremental_adapter_train_only=True"
            )
        self._incremental_adapter_train_only_requested = incremental_adapter_train_only
        try:
            started = perf_counter()
            results = super().train(
                data=data,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                lr0=lr0,
                optimizer=optimizer,
                device=device,
                workers=workers,
                pretrained=pretrained,
                resume=resume,
                seed=seed,
                project=project,
                name=name,
                exist_ok=exist_ok,
                amp=amp,
                patience=patience,
                allow_download_scripts=allow_download_scripts,
                callbacks=callbacks,
                loggers=loggers,
                incremental_adapter=incremental_adapter,
                incremental_adapter_train_only=incremental_adapter_train_only,
                incremental_adapter_type=resolved_adapter_type,
                incremental_adapter_reduction=incremental_adapter_reduction,
                incremental_adapter_spatial=incremental_adapter_spatial,
                incremental_adapter_alpha=incremental_adapter_alpha,
                incremental_adapter_train_head=incremental_adapter_train_head,
                incremental_adapter_features=incremental_adapter_features,
                **kwargs,
            )
            results["training_time_seconds"] = perf_counter() - started
            if incremental_adapter:
                results["parameter_counts"] = (
                    self.incremental_adapter_parameter_report()
                )
            return results
        finally:
            self._incremental_adapter_train_only_requested = False

    @classmethod
    def can_load(cls, weights_dict):
        return (
            "backbone.backbone.adapter_down.weight" in weights_dict
            and "head.stems.0.conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict):
        key = "backbone.backbone.projections.0.0.weight"
        if key not in weights_dict:
            return None
        channels = weights_dict[key].shape[0]
        return next(
            (
                size
                for size, cfg in YOLOXDraxMobileNetV3LargeModel.CONFIGS.items()
                if int(256 * cfg["width"]) == channels
            ),
            None,
        )

    def _trainer_class(self):
        from .trainer import YOLOXDraxMobileNetV3LargeTrainer

        return YOLOXDraxMobileNetV3LargeTrainer
