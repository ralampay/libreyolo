"""Legacy compatibility adapters; active research development is owned by MLX.

This package remains standalone for existing callers and checkpoints. New
experiments use mlx.modes.object_detection.feature_adapters instead.
"""

from .layers import create_adapter, available_adapters
from .injection import inject_adapters, count_parameters, adapter_state_dict, load_adapter_state_dict
from .yolox import yolox_targets
from .residual_fusion import DraxResidualFusionConv2d
from .hybrid import DraxHybridConv2d, DraxSpatialConv2d

__all__ = ["create_adapter", "available_adapters", "inject_adapters", "count_parameters", "adapter_state_dict", "load_adapter_state_dict", "yolox_targets", "DraxHybridConv2d", "DraxSpatialConv2d", "DraxResidualFusionConv2d"]
