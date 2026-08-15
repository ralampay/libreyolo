"""Original LibreYOLO Drax feature-refinement modules.

The implementation was written for LibreYOLO without copying or adapting
third-party source code. Its architectural vocabulary is informed by the
ConvNeXt paper (arXiv:2201.03545) and Selective Kernel Networks paper
(arXiv:1903.06586).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


DRAX_FUSION_MODES = ("average", "sknet")
DRAX_CONFIG_VERSION = 1


def resolve_drax_fusion_mode(fusion_mode: str) -> str:
    if not isinstance(fusion_mode, str):
        raise TypeError("fusion_mode must be a string")
    normalized = fusion_mode.strip().lower()

    if normalized not in DRAX_FUSION_MODES:
        supported = ", ".join(DRAX_FUSION_MODES)
        raise ValueError(
            f"Unsupported Drax fusion mode "
            f"'{fusion_mode}'. "
            f"Choose one of: {supported}."
        )

    return normalized

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        if not 0.0 <= self.drop_prob < 1.0:
            raise ValueError("drop_prob must satisfy 0.0 <= drop_prob < 1.0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = (
            keep_prob
            + torch.rand(
                shape,
                dtype=x.dtype,
                device=x.device,
            )
        )

        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor

class LayerNorm2D(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(
            num_channels,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2)

class SelfAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.GroupNorm(
            1,
            dim,
        )

        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=1,
            bias=qkv_bias,
        )

        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
        )

        self.scale = self.head_dim**-0.5

        self.attn_dropout = nn.Dropout(
            attn_dropout
        )

        self.proj_dropout = nn.Dropout(
            proj_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        residual = x

        qkv = self.qkv(
            self.norm(x)
        )

        q, k, v = torch.chunk(
            qkv,
            3,
            dim=1,
        )

        def reshape_heads(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            return tensor.reshape(
                batch_size,
                self.num_heads,
                self.head_dim,
                height * width,
            )

        q = reshape_heads(q).transpose(-2, -1)
        k = reshape_heads(k)
        v = reshape_heads(v).transpose(-2, -1)

        attn = torch.matmul(
            q,
            k,
        ) * self.scale

        attn = attn.softmax(
            dim=-1
        )

        attn = self.attn_dropout(
            attn
        )

        out = torch.matmul(
            attn,
            v,
        )

        out = (
            out.transpose(-2, -1)
            .contiguous()
            .reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        out = self.proj(out)
        out = self.proj_dropout(out)

        return residual + out

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if expansion < 1:
            raise ValueError(
                "expansion must be at least 1"
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd "
                "to preserve spatial size"
            )

        hidden_dim = dim * expansion

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        self.norm = LayerNorm2D(dim)

        self.pwconv1 = nn.Conv2d(
            dim,
            hidden_dim,
            kernel_size=1,
        )

        self.activation = nn.GELU()

        self.pwconv2 = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim)
            )
        else:
            self.layer_scale = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)

        if self.layer_scale is not None:
            x = x * self.layer_scale.view(
                1,
                -1,
                1,
                1,
            )

        x = self.dropout(x)

        return residual + x

def _resolve_num_heads(
    dim: int,
    max_heads: int = 8,
) -> int:
    for num_heads in range(
        min(max_heads, dim),
        0,
        -1,
    ):
        if dim % num_heads == 0:
            return num_heads

    return 1

class DraxBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        *,
        use_attention: bool = True,
        efficient: bool = True,
        drop_path: float = 0.0,
        fusion_mode: str = "average",
    ) -> None:
        super().__init__()

        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")

        self.use_attention = use_attention
        self.efficient = efficient

        self.fusion_mode = (
            resolve_drax_fusion_mode(
                fusion_mode
            )
        )

        # Local convolutional branch
        self.convnext = ConvNeXtBlock(
            dim
        )

        self.drop_path = DropPath(
            drop_path
        )

        # Optional learned branch weighting.
        if (
            use_attention
            and self.fusion_mode == "sknet"
        ):
            fusion_dim = max(
                32,
                dim // 16,
            )

            self.fusion_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),

                nn.Conv2d(
                    dim,
                    fusion_dim,
                    kernel_size=1,
                ),

                nn.ReLU(inplace=True),

                nn.Conv2d(
                    fusion_dim,
                    2 * dim,
                    kernel_size=1,
                ),
            )
        else:
            self.fusion_gate = None

        if not use_attention:
            self.attention = None
            self.attn_down = None
            self.attn_up = None
            return

        attention_dim = (
            _resolve_efficient_dim(dim)
            if efficient
            else dim
        )

        self.attention = SelfAttention2D(
            attention_dim,
            num_heads=_resolve_num_heads(
                attention_dim
            ),
        )

        if efficient:
            self.attn_down = nn.Conv2d(
                dim,
                attention_dim,
                kernel_size=1,
            )

            self.attn_up = nn.Conv2d(
                attention_dim,
                dim,
                kernel_size=1,
            )
        else:
            self.attn_down = None
            self.attn_up = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ConvNeXt already returns:
        #
        # x + conv_update
        #
        # so subtract x to isolate the update.
        conv_delta = (
            self.convnext(x) - x
        )

        if (
            not self.use_attention
            or self.attention is None
        ):
            return x + self.drop_path(
                conv_delta
            )

        if self.efficient:
            reduced = self.attn_down(x)

            attention_delta = self.attn_up(
                self.attention(reduced)
                - reduced
            )
        else:
            attention_delta = (
                self.attention(x) - x
            )

        fused_delta = self._fuse_deltas(
            conv_delta,
            attention_delta,
        )

        return x + self.drop_path(
            fused_delta
        )

    def _fuse_deltas(
        self,
        conv_delta: torch.Tensor,
        attention_delta: torch.Tensor,
    ) -> torch.Tensor:

        if self.fusion_mode == "average":
            return 0.5 * (
                conv_delta
                + attention_delta
            )

        if self.fusion_gate is None:
            raise RuntimeError(
                "SKNet fusion requires "
                "an initialized fusion gate."
            )

        batch_size, channels, _, _ = (
            conv_delta.shape
        )

        logits = self.fusion_gate(
            conv_delta
            + attention_delta
        )

        weights = logits.reshape(
            batch_size,
            2,
            channels,
            1,
            1,
        ).softmax(dim=1)

        return (
            weights[:, 0] * conv_delta
            + weights[:, 1]
            * attention_delta
        )

@dataclass(frozen=True)
class DraxConfig:
    enabled: bool = False

    stages: tuple[str, ...] = ("b5",)

    use_attention: bool = True
    efficient: bool = True

    fusion_mode: str = "average"

    drop_path: float = 0.0

    def __post_init__(self) -> None:
        for name in ("enabled", "use_attention", "efficient"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        if not isinstance(self.stages, tuple) or not all(
            isinstance(stage, str) for stage in self.stages
        ):
            raise TypeError("stages must be a tuple of strings")

        valid_stages = {"b3", "b4", "b5"}

        invalid = set(self.stages) - valid_stages

        if invalid:
            raise ValueError(
                f"Invalid Drax stages: {sorted(invalid)}. "
                f"Expected only: {sorted(valid_stages)}."
            )

        object.__setattr__(
            self,
            "fusion_mode",
            resolve_drax_fusion_mode(self.fusion_mode),
        )

        if isinstance(self.drop_path, bool) or not isinstance(
            self.drop_path, (int, float)
        ):
            raise TypeError("drop_path must be a number")
        drop_path = float(self.drop_path)
        if not 0.0 <= drop_path < 1.0:
            raise ValueError(
                "drop_path must satisfy "
                "0.0 <= drop_path < 1.0"
            )
        object.__setattr__(self, "drop_path", drop_path)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned primitive checkpoint representation."""
        return {
            "version": DRAX_CONFIG_VERSION,
            "enabled": self.enabled,
            "stages": list(self.stages),
            "use_attention": self.use_attention,
            "efficient": self.efficient,
            "fusion_mode": self.fusion_mode,
            "drop_path": self.drop_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DraxConfig":
        """Build a Drax configuration from checkpoint-safe metadata."""
        if not isinstance(data, dict):
            raise TypeError("Drax configuration metadata must be a dictionary")

        expected = {
            "version",
            "enabled",
            "stages",
            "use_attention",
            "efficient",
            "fusion_mode",
            "drop_path",
        }
        missing = sorted(expected - set(data))
        unexpected = sorted(set(data) - expected)
        if missing or unexpected:
            raise ValueError(
                "Invalid Drax configuration fields: "
                f"missing={missing}, unexpected={unexpected}"
            )

        version = data["version"]
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != DRAX_CONFIG_VERSION
        ):
            raise ValueError(
                f"Unsupported Drax configuration version {version!r}; "
                f"expected {DRAX_CONFIG_VERSION}"
            )

        stages = data["stages"]
        if not isinstance(stages, (list, tuple)):
            raise TypeError("Drax configuration stages must be a list of strings")

        return cls(
            enabled=data["enabled"],
            stages=tuple(stages),
            use_attention=data["use_attention"],
            efficient=data["efficient"],
            fusion_mode=data["fusion_mode"],
            drop_path=data["drop_path"],
        )


def _drax_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[DraxConfig, bool]:
    """Return checkpoint Drax configuration and whether it was inferred.

    State-dict inference is a compatibility fallback for early development
    checkpoints. It can recover enabled stages only, so all other options use
    the first supported experiment defaults.
    """
    if "drax" in checkpoint:
        return DraxConfig.from_dict(checkpoint["drax"]), False

    state_dict = checkpoint.get("train_model", checkpoint.get("model", checkpoint))
    if not isinstance(state_dict, dict):
        return DraxConfig(), False

    stage_prefixes = (
        ("b3", "backbone.drax2."),
        ("b4", "backbone.drax3."),
        ("b5", "backbone.drax4."),
    )
    stages = tuple(
        stage
        for stage, prefix in stage_prefixes
        if any(key.startswith(prefix) for key in state_dict)
    )
    if not stages:
        return DraxConfig(), False
    return DraxConfig(enabled=True, stages=stages), True

def _resolve_efficient_dim(
    dim: int,
) -> int:
    reduced_dim = max(
        32,
        dim // 2,
    )

    while (
        reduced_dim > 1
        and dim % reduced_dim != 0
    ):
        reduced_dim -= 1

    return reduced_dim
