"""YOLOX feature-placement policy, separate from generic adapter layers."""

from torch import nn


def yolox_targets(model: nn.Module, placement: str = "backbone+neck", adapter: str = "drax") -> dict[str, int]:
    if placement not in {"backbone", "neck", "backbone+neck"}:
        raise ValueError("placement must be backbone, neck, or backbone+neck")
    if not hasattr(model, "backbone") or not hasattr(model, "head"):
        raise ValueError("Expected a LibreYOLOXModel")
    if adapter == "lora":
        roots = ("backbone.backbone.",) if placement == "backbone" else (("backbone.",) if placement == "backbone+neck" else ("backbone.lateral_conv0.", "backbone.reduce_conv1.", "backbone.C3_", "backbone.bu_conv"))
        return {name: module.out_channels for name, module in model.named_modules()
                if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)
                and module.groups == 1 and name.startswith(roots) and not name.startswith("head.")}
    width = model.CONFIGS[model.config]["width"]
    selected = {}
    if placement in {"backbone", "backbone+neck"}:
        selected.update({f"backbone.backbone.dark{stage}": int(ch * width)
                         for stage, ch in ((3, 256), (4, 512), (5, 1024))})
    if placement in {"neck", "backbone+neck"}:
        selected.update({f"backbone.{stage}": int(ch * width)
                         for stage, ch in (("C3_p3", 256), ("C3_n3", 512), ("C3_n4", 1024))})
    return selected
