"""Scratch YOLOX training recipe for the CSP–Drax family."""

from dataclasses import dataclass

from ...training.config import YOLOXConfig
from ..yolox.trainer import YOLOXTrainer


@dataclass(kw_only=True)
class YOLOXDraxCSPMConfig(YOLOXConfig):
    """Use the same optimization defaults as the YOLOX control model."""


class YOLOXDraxCSPMTrainer(YOLOXTrainer):
    @classmethod
    def _config_class(cls):
        return YOLOXDraxCSPMConfig

    def get_model_family(self):
        return "yolox_drax_csp"

    def get_model_tag(self):
        return "YOLOX-Drax-CSP-M"
