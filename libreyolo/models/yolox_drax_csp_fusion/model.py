"""Parameter-efficient YOLOX-M detector with attentive CSP feature fusion."""

from ..yolox.model import LibreYOLOX
from .nn import YOLOXDraxCSPFusionMNetwork
from .trainer import YOLOXDraxCSPFusionMConfig, YOLOXDraxCSPFusionMTrainer


class LibreYOLOXDraxCSPFusionM(LibreYOLOX):
    FAMILY = "yolox_drax_csp_fusion"
    FILENAME_PREFIX = "LibreYOLOXDraxCSPFusion"
    CLI_ALIASES = ("yolox-drax-csp-fusion",)
    CLI_DEFAULT_SIZE = "m"
    TASK_INPUT_SIZES = {"detect": {"m": 640}}
    TRAIN_CONFIG = YOLOXDraxCSPFusionMConfig

    def __init__(self, model_path=None, size="m", **kwargs):
        if size != "m":
            raise ValueError("yolox-drax-csp-fusion supports size 'm' only")
        super().__init__(model_path=model_path, size=size, **kwargs)

    def _init_model(self):
        return YOLOXDraxCSPFusionMNetwork(
            config=self.size, nb_classes=self.nb_classes
        )

    def _strict_loading(self):
        return True

    @classmethod
    def can_load(cls, weights_dict):
        return (
            "backbone.p5_context_scale" in weights_dict
            and "backbone.fuse_p4_top_down.select.weight" in weights_dict
            and "head.stems.0.conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict):
        return "m" if cls.can_load(weights_dict) else None

    def _trainer_class(self):
        return YOLOXDraxCSPFusionMTrainer

    def _get_available_layers(self):
        backbone = self.model.backbone.backbone
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
            "channels": [192, 384, 768],
            "strides": [8, 16, 32],
        }
