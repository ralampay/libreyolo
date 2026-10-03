"""YOLOX-M CSP backbone with compact Drax refinement and narrower PAN/head."""

from ..yolox.model import LibreYOLOX
from .nn import YOLOXDraxCSPMNetwork
from .trainer import YOLOXDraxCSPMConfig, YOLOXDraxCSPMTrainer


class LibreYOLOXDraxCSPM(LibreYOLOX):
    FAMILY = "yolox_drax_csp"
    FILENAME_PREFIX = "LibreYOLOXDraxCSP"
    CLI_ALIASES = ("yolox-drax-csp",)
    CLI_DEFAULT_SIZE = "m"
    TASK_INPUT_SIZES = {"detect": {"m": 640}}
    TRAIN_CONFIG = YOLOXDraxCSPMConfig

    def __init__(self, model_path=None, size="m", **kwargs):
        if size != "m":
            raise ValueError("yolox-drax-csp supports size 'm' only")
        super().__init__(model_path=model_path, size=size, **kwargs)

    def _init_model(self):
        return YOLOXDraxCSPMNetwork(config=self.size, nb_classes=self.nb_classes)

    def _strict_loading(self):
        return True

    @classmethod
    def can_load(cls, weights_dict):
        return (
            "backbone.backbone.p5_scale" in weights_dict
            and "backbone.backbone.core.stem.conv.conv.weight" in weights_dict
            and "head.stems.0.conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict):
        return "m" if cls.can_load(weights_dict) else None

    def _trainer_class(self):
        return YOLOXDraxCSPMTrainer

    def _get_available_layers(self):
        backbone = self.model.backbone.backbone.core
        return {
            "backbone_stem": backbone.stem,
            "backbone_dark2": backbone.dark2,
            "backbone_dark3": backbone.dark3,
            "backbone_dark4": backbone.dark4,
            "backbone_dark5": backbone.dark5,
        }

    def get_distill_config(self):
        return {
            "tap_points": ["backbone.C3_p3", "backbone.C3_n3", "backbone.C3_n4"],
            "channels": [176, 352, 704],
            "strides": [8, 16, 32],
        }
