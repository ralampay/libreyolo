"""CSP feature extractor with compact Drax refinement for YOLOX."""

import torch
from torch import nn

from .refinement import RefineSpatialFeatures
from ..yolo9.drax import DraxBlock
from ..yolox.nn import CSPDarknet, LibreYOLOXModel


class CSPDraxBackbone(nn.Module):
    """Preserve YOLOX-M features and refine spatial and deep context."""

    def __init__(self):
        super().__init__()
        self.core = CSPDarknet(0.67, 0.75)
        self.p3_refiner = RefineSpatialFeatures(192)
        self.p5_reduce = nn.Sequential(
            nn.Conv2d(768, 160, 1, bias=False), nn.BatchNorm2d(160), nn.SiLU(),
        )
        self.p5_drax = DraxBlock(160)
        self.p5_expand = nn.Sequential(
            nn.Conv2d(160, 768, 1, bias=False), nn.BatchNorm2d(768),
        )
        self.p5_scale = nn.Parameter(torch.tensor(0.05))
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(source, target, 1, bias=False),
                nn.BatchNorm2d(target), nn.SiLU(),
            )
            for source, target in ((192, 176), (384, 352), (768, 704))
        )
        with torch.no_grad():
            self.p3_refiner.scale.fill_(0.05)
            self.p5_drax.convnext.layer_scale.fill_(0.05)

    def forward(self, x):
        features = self.core(x)
        p3 = self.p3_refiner(features["dark3"])
        p5 = features["dark5"]
        reduced = self.p5_reduce(p5)
        delta = self.p5_drax(reduced) - reduced
        p5 = p5 + self.p5_scale.clamp(-0.1, 0.1) * self.p5_expand(delta)
        return {
            "dark3": self.projections[0](p3),
            "dark4": self.projections[1](features["dark4"]),
            "dark5": self.projections[2](p5),
        }


class YOLOXDraxCSPMNetwork(LibreYOLOXModel):
    CONFIGS = {
        **LibreYOLOXModel.CONFIGS,
        "m": {"depth": 0.67, "width": 0.6875, "depthwise": False},
    }

    def __init__(self, config="m", nb_classes=80):
        if config != "m":
            raise ValueError("YOLOXDraxCSPMNetwork supports size 'm' only")
        super().__init__(config=config, nb_classes=nb_classes, backbone=CSPDraxBackbone())
