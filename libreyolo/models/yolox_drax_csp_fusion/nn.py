"""YOLOX-M feature fusion with an independently implemented Drax context path.

The branch-selection vocabulary is informed by Selective Kernel Networks
(arXiv:1903.06586). The multi-scale placement is informed by EfficientDet
(arXiv:1911.09070) and ASFF (arXiv:1911.09516). This implementation was
written for LibreYOLO and does not copy third-party source code.
"""

from __future__ import annotations

import torch
from torch import nn

from ..yolo9.drax import DraxBlock
from ..yolox.nn import (
    BaseConv,
    LibreYOLOXModel,
    YOLOPAFPN,
    YOLOXHead,
)


class BranchAttentionFusion(nn.Module):
    """Reweight two equal-shaped branches without changing initial scale."""

    def __init__(self, channels: int, reduction: int = 16, min_channels: int = 32):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden_channels = max(min_channels, channels // reduction)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.select = nn.Conv2d(hidden_channels, 2 * channels, kernel_size=1)
        nn.init.zeros_(self.select.weight)
        nn.init.zeros_(self.select.bias)

    def forward(self, first: torch.Tensor, second: torch.Tensor):
        if first.shape != second.shape:
            raise ValueError(
                "fusion branches must have identical shapes, got "
                f"{tuple(first.shape)} and {tuple(second.shape)}"
            )
        batch_size = first.shape[0]
        descriptor = self.pool(first + second)
        logits = self.select(self.activation(self.reduce(descriptor)))
        weights = logits.reshape(batch_size, 2, self.channels, 1, 1)
        # Multiplying the two-way softmax by two makes the zero-logit state an
        # exact pass-through for both branches before their normal concat.
        weights = 2.0 * weights.softmax(dim=1)
        return weights[:, 0] * first, weights[:, 1] * second


class DraxFusionPAFPN(YOLOPAFPN):
    """YOLOX-M PAN with P5 context and attention at every branch merge."""

    def __init__(self, act: str = "silu"):
        super().__init__(depth=0.67, width=0.75, depthwise=False, act=act)
        c3, c4, c5 = 192, 384, 768
        self.p5_reduce = BaseConv(c5, 160, 1, 1, act=act)
        self.p5_drax = DraxBlock(160, efficient=True, fusion_mode="sknet")
        self.p5_expand = nn.Sequential(
            nn.Conv2d(160, c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(c5),
        )
        self.p5_context_scale = nn.Parameter(torch.tensor(1e-3))
        if self.p5_drax.fusion_gate is not None:
            nn.init.zeros_(self.p5_drax.fusion_gate[-1].weight)
            nn.init.zeros_(self.p5_drax.fusion_gate[-1].bias)

        self.fuse_p4_top_down = BranchAttentionFusion(c4)
        self.fuse_p3_top_down = BranchAttentionFusion(c3)
        self.fuse_p4_bottom_up = BranchAttentionFusion(c3)
        self.fuse_p5_bottom_up = BranchAttentionFusion(c4)

    def forward(self, input_tensor: torch.Tensor):
        out_features = self.backbone(input_tensor)
        x2, x1, x0 = [out_features[name] for name in self.in_features]

        reduced = self.p5_reduce(x0)
        context_delta = self.p5_drax(reduced) - reduced
        x0 = x0 + self.p5_context_scale * self.p5_expand(context_delta)

        fpn_out0 = self.lateral_conv0(x0)
        top_p4, skip_p4 = self.fuse_p4_top_down(self.upsample(fpn_out0), x1)
        f_out0 = self.C3_p4(torch.cat((top_p4, skip_p4), dim=1))

        fpn_out1 = self.reduce_conv1(f_out0)
        top_p3, skip_p3 = self.fuse_p3_top_down(self.upsample(fpn_out1), x2)
        pan_out2 = self.C3_p3(torch.cat((top_p3, skip_p3), dim=1))

        bottom_p4, top_skip_p4 = self.fuse_p4_bottom_up(
            self.bu_conv2(pan_out2), fpn_out1
        )
        pan_out1 = self.C3_n3(torch.cat((bottom_p4, top_skip_p4), dim=1))

        bottom_p5, top_skip_p5 = self.fuse_p5_bottom_up(
            self.bu_conv1(pan_out1), fpn_out0
        )
        pan_out0 = self.C3_n4(torch.cat((bottom_p5, top_skip_p5), dim=1))
        return pan_out2, pan_out1, pan_out0


class YOLOXDraxCSPFusionMNetwork(LibreYOLOXModel):
    """Full-width YOLOX-M body with a depthwise decoupled head."""

    def __init__(self, config: str = "m", nb_classes: int = 80, act: str = "silu"):
        if config != "m":
            raise ValueError("YOLOXDraxCSPFusionMNetwork supports size 'm' only")
        nn.Module.__init__(self)
        self.config = config
        self.nb_classes = nb_classes
        self.backbone = DraxFusionPAFPN(act=act)
        self.head = YOLOXHead(
            num_classes=nb_classes,
            width=0.75,
            in_channels=[256, 512, 1024],
            depthwise=True,
            act=act,
        )
        self._apply_official_bn_hyperparams()
