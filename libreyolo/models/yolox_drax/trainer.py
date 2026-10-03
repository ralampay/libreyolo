"""Training metadata for the balanced YOLOX detector."""

from importlib import import_module

_mobile_trainer = import_module("libreyolo.models.yolox-drax-mobilenet-v3.trainer")


class YOLOXDraxMTrainer(_mobile_trainer.YOLOXDraxMobileNetV3LargeTrainer):
    def get_model_family(self):
        return "yolox_drax"

    def get_model_tag(self):
        return "YOLOX-Drax-M"
