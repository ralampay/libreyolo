"""Balanced-capacity YOLOX detector using the existing pyramid Drax backbone."""

from importlib import import_module

from .nn import YOLOXDraxMNetwork
from .trainer import YOLOXDraxMTrainer

_mobile = import_module("libreyolo.models.yolox-drax-mobilenet-v3")


class LibreYOLOXDraxM(_mobile.LibreYOLOXDraxMobileNetV3Large):
    """The fixed 0.67-depth, 0.875-width, pyramid-Drax detector."""

    FAMILY = "yolox_drax"
    FILENAME_PREFIX = "LibreYOLOXDrax"
    CLI_ALIASES = ("yolox-drax",)
    CLI_DEFAULT_SIZE = "m"
    TASK_INPUT_SIZES = {"detect": {"m": 640}}
    NETWORK_CLASS = YOLOXDraxMNetwork

    def __init__(self, model_path=None, size="m", architecture_variant=None, **kwargs):
        if size != "m":
            raise ValueError("yolox-drax supports size 'm' only")
        if architecture_variant not in (None, "pyramid-drax"):
            raise ValueError("yolox-drax-m requires the pyramid-drax backbone")
        super().__init__(
            model_path=model_path, size=size,
            architecture_variant="pyramid-drax", **kwargs,
        )

    def get_distill_config(self):
        width = YOLOXDraxMNetwork.CONFIGS["m"]["width"]
        return {
            "tap_points": ["backbone.C3_p3", "backbone.C3_n3", "backbone.C3_n4"],
            "channels": [int(c * width) for c in (256, 512, 1024)],
            "strides": [8, 16, 32],
        }

    @classmethod
    def can_load(cls, weights_dict):
        projection = weights_dict.get("backbone.backbone.projections.0.0.weight")
        return (
            projection is not None
            and projection.shape[0] == 224
            and "backbone.backbone.feature_refiners.0.scale" in weights_dict
            and "head.stems.0.conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict):
        return "m" if cls.can_load(weights_dict) else None

    def _trainer_class(self):
        return YOLOXDraxMTrainer
