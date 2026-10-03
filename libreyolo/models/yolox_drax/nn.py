"""Balanced-width YOLOX network using the existing compact pyramid backbone."""

from importlib import import_module

import torch

from ..yolox.nn import LibreYOLOXModel

_mobile_nn = import_module("libreyolo.models.yolox-drax-mobilenet-v3.nn")


class YOLOXDraxMNetwork(_mobile_nn.YOLOXDraxMobileNetV3LargeModel):
    """Use a wider PAN/head and initialize feature refiners to contribute."""

    CONFIGS = {
        **LibreYOLOXModel.CONFIGS,
        "m": {"depth": 0.67, "width": 0.875, "depthwise": False},
    }

    def __init__(self, config="m", nb_classes=80, architecture_variant="pyramid-drax"):
        if config != "m" or architecture_variant != "pyramid-drax":
            raise ValueError("YOLOXDraxMNetwork requires size m and pyramid-drax")
        super().__init__(
            config=config, nb_classes=nb_classes,
            architecture_variant=architecture_variant,
        )
        backbone = self.backbone.backbone
        with torch.no_grad():
            for refiner in (*backbone.feature_refiners, backbone.p5_pool):
                refiner.scale.fill_(0.05)
            drax = backbone.drax_refiner[0]
            drax.convnext.layer_scale.fill_(0.05)
            drax.attention_scale.fill_(0.05)
