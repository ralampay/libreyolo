"""YOLO9 neck/head with a fixed Drax MobileNetV3 Large backbone.

train(pretrained=True) initializes ImageNet feature weights only.
Constructing or reloading a model never downloads backbone weights.
"""

from typing import ClassVar

from ..drax_mobilenet_v3.model import DraxMobileNetVariant
from ..yolo9.model import LibreYOLO9
from ..yolo9.nn import YOLO9_CONFIGS
from .nn import YOLO9DraxMobileNetV3LargeModel


class LibreYOLO9DraxMobileNetV3Large(DraxMobileNetVariant, LibreYOLO9):
    """Detection with a fixed Drax MobileNetV3 Large feature extractor.

    ``train(pretrained=True)`` initializes only ImageNet feature weights;
    ``False`` trains from scratch. Load detector checkpoints via model_path.
    The pretrained argument accepts bools only.
    """

    FAMILY = "yolo9_drax_mobilenet_v3_large"
    FILENAME_PREFIX = "LibreYOLO9DraxMobileNetV3Large"
    CLI_ALIASES = ("yolo9-drax-mobilenet-v3-large",)
    TASK_INPUT_SIZES: ClassVar = {"detect": LibreYOLO9.INPUT_SIZES}

    def __init__(self, model_path=None, size="s", **kwargs):
        super().__init__(model_path, size=size, **kwargs)

    @property
    def uses_drax(self):
        return True

    @property
    def drax_stages(self):
        return ("b5",)

    def _init_model(self):
        return YOLO9DraxMobileNetV3LargeModel(self.size, self.nb_classes, self.reg_max)

    def _mobile_backbone(self):
        return self.model.backbone

    @classmethod
    def can_load(cls, weights_dict):
        return (
            "backbone.adapter_down.weight" in weights_dict
            and "head.cv2.0.0.conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict):
        key = "backbone.projections.0.0.weight"
        if key not in weights_dict:
            return None
        channels = weights_dict[key].shape[0]
        return next(
            (
                size
                for size, cfg in YOLO9_CONFIGS.items()
                if cfg["stages"][0][1] == channels
            ),
            None,
        )

    def _trainer_class(self):
        from .trainer import YOLO9DraxMobileNetV3LargeTrainer

        return YOLO9DraxMobileNetV3LargeTrainer

    def get_backbone_distill_config(self):
        return {
            "tap_points": ["backbone.projections.1"],
            "channels": [self.model.backbone.out_channels[1]],
            "strides": [16],
        }
