"""MobileNetV3 Large with the MLX late-stage Drax adapter.

Adapter derived from ralampay/mlx at
986408db404f6ccc97a2c5acf6081d3731053bd9 (MIT); see NOTICE.
MobileNet is provided by torchvision (BSD-3-Clause).
"""

import torch
from torch import nn

from ..yolo9.drax import DraxBlock


class DraxMobileNetV3LargeBackbone(nn.Module):
    """Return projected P3/P4/P5 features; accept the detector's native pixels."""

    feature_channels = (40, 112, 960)

    def __init__(self, out_channels, *, input_bgr=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_large

        self.features = mobilenet_v3_large(weights=None).features
        self.input_bgr = input_bgr
        self.out_channels = tuple(out_channels)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.adapter_down = nn.Conv2d(960, 160, 1, bias=False)
        self.adapter_norm = nn.BatchNorm2d(160)
        self.adapter_activation = nn.Hardswish()
        self.drax_refiner = nn.Sequential(DraxBlock(160))
        self.adapter_up = nn.Conv2d(160, 960, 1, bias=False)
        self.adapter_up_norm = nn.BatchNorm2d(960)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(source, target, 1, bias=False),
                    nn.BatchNorm2d(target, eps=1e-3, momentum=0.03),
                    nn.SiLU(),
                )
                for source, target in zip(self.feature_channels, self.out_channels)
            ]
        )

    def load_imagenet_weights(self):
        """Explicitly load feature tensors, excluding the classification head."""
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        reference = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features.load_state_dict(reference.features.state_dict(), strict=True)

    def forward_features(self, x):
        if self.input_bgr:
            x = x[:, [2, 1, 0]] / 255.0
        x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (6, 12):
                outputs.append(x)
        residual = x
        x = self.adapter_activation(self.adapter_norm(self.adapter_down(x)))
        x = self.drax_refiner(x)
        x = residual + self._adapter_update(x)
        return (*outputs, x)

    def _adapter_update(self, x):
        return self.adapter_up_norm(self.adapter_up(x))

    def forward(self, x):
        return tuple(
            layer(feature)
            for layer, feature in zip(self.projections, self.forward_features(x))
        )


class YOLOXBackbone(DraxMobileNetV3LargeBackbone):
    def forward(self, x):
        return dict(zip(("dark3", "dark4", "dark5"), super().forward(x)))
