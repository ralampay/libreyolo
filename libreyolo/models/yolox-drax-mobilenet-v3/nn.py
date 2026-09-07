"""Detection architecture with MobileNet feature injection."""

from torch import nn

from ..drax_mobilenet_v3.backbone import YOLOXBackbone
from ..yolox.nn import LibreYOLOXModel


class YOLOXDraxMobileNetV3LargeModel(LibreYOLOXModel):
    def __init__(self, config="s", nb_classes=80):
        width = self.CONFIGS[config]["width"]
        backbone = YOLOXBackbone(
            tuple(int(c * width) for c in (256, 512, 1024)), input_bgr=True
        )
        super().__init__(config=config, nb_classes=nb_classes, backbone=backbone)

    def _apply_official_bn_hyperparams(self):
        # Keep MobileNet's pretrained epsilon and the adapter's original settings.
        mobile_modules = set(self.backbone.backbone.modules())
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) and module not in mobile_modules:
                module.eps = 1e-3
                module.momentum = 0.03
