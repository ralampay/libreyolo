"""YOLO9 recipe for the Drax MobileNetV3 Large variant."""

from ..yolo9.trainer import YOLO9Trainer


class YOLO9DraxMobileNetV3LargeTrainer(YOLO9Trainer):
    artifact_model_families = ("yolo9_drax_mobilenet_v3_large",)
    _BACKBONE_FREEZE_MODULES = (
        "features",
        "adapter_down",
        "adapter_norm",
        "drax_refiner",
        "adapter_up",
        "adapter_up_norm",
        "projections",
    )

    def get_model_family(self):
        return "yolo9_drax_mobilenet_v3_large"

    def get_model_tag(self):
        return f"YOLO9-Drax-MobileNetV3-Large-{self.config.size}"
