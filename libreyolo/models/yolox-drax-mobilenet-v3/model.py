"""YOLOX neck/head with a fixed Drax MobileNetV3 Large backbone.

train(pretrained=True) initializes ImageNet feature weights only.
The external YOLOX BGR 0-255 input contract is preserved.
"""

from typing import ClassVar

from ..drax_mobilenet_v3.model import DraxMobileNetVariant
from ..yolox.model import LibreYOLOX
from .nn import YOLOXDraxMobileNetV3LargeModel


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

    def _init_model(self):
        return YOLOXDraxMobileNetV3LargeModel(self.size, self.nb_classes)

    def _mobile_backbone(self):
        return self.model.backbone.backbone

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
