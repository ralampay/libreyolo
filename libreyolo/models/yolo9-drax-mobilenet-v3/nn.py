"""Detection architecture with MobileNet feature injection."""

from ..drax_mobilenet_v3.backbone import DraxMobileNetV3LargeBackbone
from ..yolo9.nn import YOLO9_CONFIGS, LibreYOLO9Model


class YOLO9DraxMobileNetV3LargeModel(LibreYOLO9Model):
    def __init__(self, config="s", nb_classes=80, reg_max=16):
        cfg = YOLO9_CONFIGS[config]
        channels = (cfg["stages"][0][1], cfg["stages"][1][1], cfg["spp_out"])
        backbone = DraxMobileNetV3LargeBackbone(channels)
        super().__init__(
            config=config, nb_classes=nb_classes, reg_max=reg_max, backbone=backbone
        )
