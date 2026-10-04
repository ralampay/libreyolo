"""YOLOX training recipe for the CSP-Drax feature-fusion family."""

from dataclasses import dataclass

from ...training.config import YOLOXConfig
from ..yolox.trainer import YOLOXTrainer


@dataclass(kw_only=True)
class YOLOXDraxCSPFusionMConfig(YOLOXConfig):
    """Match the YOLOX control optimization defaults."""


class YOLOXDraxCSPFusionMTrainer(YOLOXTrainer):
    @classmethod
    def _config_class(cls):
        return YOLOXDraxCSPFusionMConfig

    def get_model_family(self):
        return "yolox_drax_csp_fusion"

    def get_model_tag(self):
        return "YOLOX-Drax-CSP-Fusion-M"
