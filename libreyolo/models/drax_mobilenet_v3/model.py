"""Lifecycle shared by the two fixed-architecture detector variants."""


class DraxMobileNetVariant:
    CLI_DEFAULT_SIZE = "s"
    PRETRAINED_BACKBONE_ONLY = True
    SUPPORTED_TASKS = ("detect",)
    SUPPORTS_CUDA_GRAPH = False

    def __init__(self, model_path=None, size="s", **kwargs):
        from pathlib import Path

        if kwargs.get("drax_config") is not None:
            raise ValueError(
                "This variant uses a fixed Drax adapter; drax_config applies to LibreYOLO9."
            )
        self._has_detector_weights = model_path is not None
        if isinstance(model_path, Path):
            model_path = str(model_path)
        super().__init__(model_path, size=size, **kwargs)

    def _load_weights(self, model_path):
        super()._load_weights(model_path)
        self._has_detector_weights = True

    @classmethod
    def get_download_url(cls, filename):
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict):
        return None

    def _strict_loading(self):
        return True

    def _prepare_scratch_init(self):
        super()._prepare_scratch_init()
        self._imagenet_initialized = False
        self._has_detector_weights = False

    def _initialize_training_weights(self, pretrained, resume):
        if pretrained is not None and not isinstance(pretrained, bool):
            raise ValueError(
                "pretrained must be True or False; load detector checkpoints with model_path."
            )
        if (
            resume
            or self._has_detector_weights
            or getattr(self, "_imagenet_initialized", False)
        ):
            return
        if pretrained:
            self._mobile_backbone().load_imagenet_weights()
            self._imagenet_initialized = True

    def _get_available_layers(self):
        backbone = self._mobile_backbone()
        return {
            "backbone_p3": backbone.projections[0],
            "backbone_p4": backbone.projections[1],
            "backbone_p5": backbone.projections[2],
            "backbone_drax": backbone.drax_refiner,
        }

    def _model_info_extra(self):
        backbone = self._mobile_backbone()
        return {
            "backbone": "drax_mobilenet_v3_large",
            "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
            "drax_parameters": sum(
                p.numel() for p in backbone.drax_refiner.parameters()
            ),
        }
