"""YOLOX recipe for the Drax MobileNetV3 Large variant."""

from ..yolox.trainer import YOLOXTrainer


class YOLOXDraxMobileNetV3LargeTrainer(YOLOXTrainer):
    def get_model_family(self):
        return "yolox_drax_mobilenet_v3_large"

    def get_model_tag(self):
        return f"YOLOX-Drax-MobileNetV3-Large-{self.config.size}"

    def get_freeze_groups(self):
        backbone = self.model.backbone.backbone
        groups = [
            (f"backbone.backbone.{name}", module)
            for name, module in backbone.named_children()
        ]
        groups.extend(
            (f"backbone.{name}", module)
            for name, module in self.model.backbone.named_children()
            if name != "backbone"
        )
        groups.append(("head", self.model.head))
        return groups
