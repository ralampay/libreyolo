"""Focused unit coverage for optional Drax refinement in YOLOv9."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from libreyolo.models.yolo9.drax import DraxBlock, DraxConfig
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import Backbone9, LibreYOLO9Model
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from libreyolo.utils.serialization import (
    load_trusted_torch_file,
    wrap_libreyolo_checkpoint,
)

pytestmark = pytest.mark.unit


def _b5_config(**overrides) -> DraxConfig:
    values = {
        "enabled": True,
        "stages": ("b5",),
        "use_attention": True,
        "efficient": True,
        "fusion_mode": "average",
        "drop_path": 0.0,
    }
    values.update(overrides)
    return DraxConfig(**values)


def _wrapped_checkpoint(model: LibreYOLO9, **extra):
    return wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="yolo9",
        size=model.size,
        task="detect",
        nc=model.nb_classes,
        names=model.names,
        imgsz=640,
        **extra,
    )


def test_drax_config_serialization_round_trip():
    config = _b5_config(stages=("b4", "b5"), fusion_mode="SKNET", drop_path=0.2)
    data = config.to_dict()

    assert data == {
        "version": 1,
        "enabled": True,
        "stages": ["b4", "b5"],
        "use_attention": True,
        "efficient": True,
        "fusion_mode": "sknet",
        "drop_path": 0.2,
    }
    assert DraxConfig.from_dict(data) == config


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"stages": ("b2",)}, "Invalid Drax stages"),
        ({"stages": ["b5"]}, "tuple of strings"),
        ({"fusion_mode": "sum"}, "Unsupported Drax fusion mode"),
        ({"drop_path": -0.1}, "drop_path"),
        ({"drop_path": 1.0}, "drop_path"),
    ],
)
def test_drax_config_validation(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        DraxConfig(enabled=True, **kwargs)


def test_drax_config_rejects_incomplete_or_unknown_versions():
    with pytest.raises(ValueError, match="missing"):
        DraxConfig.from_dict({"version": 1})
    data = _b5_config().to_dict()
    data["version"] = 2
    with pytest.raises(ValueError, match="Unsupported"):
        DraxConfig.from_dict(data)


@pytest.mark.parametrize("fusion_mode", ["average", "sknet"])
def test_drax_block_preserves_shape(fusion_mode):
    block = DraxBlock(dim=32, fusion_mode=fusion_mode)
    output = block(torch.randn(2, 32, 8, 8))
    assert output.shape == (2, 32, 8, 8)


def test_drax_block_backward_reaches_both_branches():
    block = DraxBlock(dim=32, use_attention=True, efficient=True)
    output = block(torch.randn(1, 32, 8, 8, requires_grad=True))
    output.square().mean().backward()

    representatives = (
        block.convnext.pwconv2.weight,
        block.attention.qkv.weight,
        block.attn_down.weight,
        block.attn_up.weight,
    )
    for parameter in representatives:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("stages", "types"),
    [
        (("b5",), (nn.Identity, nn.Identity, DraxBlock)),
        (("b4", "b5"), (nn.Identity, DraxBlock, DraxBlock)),
    ],
)
def test_drax_stage_construction(stages, types):
    backbone = Backbone9(
        config="s",
        drax_config=_b5_config(stages=stages, use_attention=False),
    )
    assert isinstance(backbone.drax2, types[0])
    assert isinstance(backbone.drax3, types[1])
    assert isinstance(backbone.drax4, types[2])


@pytest.mark.parametrize(
    "config",
    [
        DraxConfig(),
        _b5_config(),
        _b5_config(stages=("b4", "b5"), use_attention=False),
    ],
)
def test_backbone_channel_contract_at_640(config):
    backbone = Backbone9(config="s", drax_config=config).eval()
    with torch.no_grad():
        p3, p4, p5 = backbone(torch.zeros(1, 3, 640, 640))
    assert p3.shape == (1, 128, 80, 80)
    assert p4.shape == (1, 192, 40, 40)
    assert p5.shape == (1, 256, 20, 20)


def test_explicitly_disabled_model_matches_vanilla():
    torch.manual_seed(11)
    vanilla = LibreYOLO9Model(config="s", nb_classes=3).eval()
    explicit = LibreYOLO9Model(
        config="s", nb_classes=3, drax_config=DraxConfig(enabled=False)
    ).eval()
    explicit.load_state_dict(vanilla.state_dict(), strict=True)

    sample = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        vanilla_output = vanilla(sample)["predictions"]
        explicit_output = explicit(sample)["predictions"]
    torch.testing.assert_close(vanilla_output, explicit_output)


def test_full_yolo9_b5_drax_forward():
    model = LibreYOLO9Model(config="s", nb_classes=3, drax_config=_b5_config()).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output["predictions"].shape == (1, 7, 84)


def test_layer_registry_and_freeze_groups_include_drax():
    wrapper = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    layers = wrapper._get_available_layers()
    assert layers["backbone_drax4"] is wrapper.model.backbone.drax4

    trainer = YOLO9Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data="unused.yaml",
        device="cpu",
    )
    names = [name for name, _module in trainer.get_freeze_groups()]
    assert "backbone.drax2" in names
    assert "backbone.drax3" in names
    assert "backbone.drax4" in names


def test_optimizer_contains_all_trainable_drax_parameters():
    wrapper = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    trainer = YOLO9Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data="unused.yaml",
        device="cpu",
    )
    optimizer = trainer._setup_optimizer()
    drax_params = {
        id(parameter)
        for parameter in wrapper.model.backbone.drax4.parameters()
        if parameter.requires_grad
    }
    optimizer_params = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert drax_params
    assert drax_params <= optimizer_params


def test_drax_checkpoint_round_trip_reconstructs_and_restores_weight(tmp_path):
    config = _b5_config()
    original = LibreYOLO9(None, size="s", device="cpu", drax_config=config)
    key = "backbone.drax4.convnext.dwconv.weight"
    with torch.no_grad():
        original.model.state_dict()[key].fill_(0.125)

    checkpoint = tmp_path / "drax.pt"
    original.save(str(checkpoint))
    raw = load_trusted_torch_file(checkpoint)
    assert raw["drax"] == config.to_dict()

    loaded = LibreYOLO9(str(checkpoint), size="s", device="cpu")
    assert loaded.drax_config == config
    assert isinstance(loaded.model.backbone.drax4, DraxBlock)
    assert torch.equal(original.model.state_dict()[key], loaded.model.state_dict()[key])


def test_explicit_compatible_config_precedes_checkpoint_metadata(tmp_path):
    original = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    checkpoint = tmp_path / "drax.pt"
    original.save(str(checkpoint))

    explicit = _b5_config(drop_path=0.25)
    loaded = LibreYOLO9(
        str(checkpoint), size="s", device="cpu", drax_config=explicit
    )
    assert loaded.drax_config == explicit


def test_model_info_exposes_drax_architecture_and_parameter_count():
    model = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    info = model.info(verbose=False)
    assert info["drax"] == model.drax_config.to_dict()
    assert info["drax_parameters"] > 0
    assert info["parameters"] > info["drax_parameters"]


def test_legacy_vanilla_checkpoint_loads_with_drax_disabled(tmp_path):
    vanilla = LibreYOLO9(None, size="s", device="cpu")
    checkpoint = tmp_path / "vanilla.pt"
    torch.save(_wrapped_checkpoint(vanilla), checkpoint)

    loaded = LibreYOLO9(str(checkpoint), size="s", device="cpu")
    assert loaded.drax_config == DraxConfig()
    assert isinstance(loaded.model.backbone.drax2, nn.Identity)
    assert isinstance(loaded.model.backbone.drax3, nn.Identity)
    assert isinstance(loaded.model.backbone.drax4, nn.Identity)


def test_missing_trained_drax_tensor_is_rejected(tmp_path):
    model = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    checkpoint = _wrapped_checkpoint(model, drax=model.drax_config.to_dict())
    checkpoint["model"].pop("backbone.drax4.convnext.dwconv.weight")
    path = tmp_path / "corrupt.pt"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="Drax checkpoint tensors do not match"):
        LibreYOLO9(str(path), size="s", device="cpu")


def test_legacy_drax_checkpoint_infers_stages(tmp_path, caplog):
    model = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    path = tmp_path / "legacy-drax.pt"
    torch.save(_wrapped_checkpoint(model), path)

    with caplog.at_level(logging.WARNING):
        loaded = LibreYOLO9(str(path), size="s", device="cpu")
    assert loaded.drax_config == _b5_config()
    assert "inferred stages B5" in caplog.text


def test_vanilla_transfer_into_drax_preserves_fresh_drax_parameters(tmp_path):
    vanilla = LibreYOLO9(None, size="s", device="cpu")
    checkpoint = tmp_path / "vanilla.pt"
    vanilla.save(str(checkpoint))

    drax = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    key = "backbone.drax4.convnext.dwconv.weight"
    before = drax.model.state_dict()[key].clone()
    stats = drax._load_transfer_weights(checkpoint)

    assert torch.equal(
        drax.model.state_dict()["backbone.conv0.conv.weight"],
        vanilla.model.state_dict()["backbone.conv0.conv.weight"],
    )
    assert torch.equal(drax.model.state_dict()[key], before)
    assert stats["drax_new_tensors"] > 0
    assert stats["drax_new_parameters"] > 0
    assert stats["skipped_shape_mismatch"] == 0


def test_trainer_metadata_and_matching_resume(tmp_path):
    config = _b5_config()
    wrapper = LibreYOLO9(None, size="s", device="cpu", drax_config=config)
    trainer = YOLO9Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data="unused.yaml",
        device="cpu",
    )
    trainer.optimizer = trainer._setup_optimizer()
    first = next(iter(trainer.optimizer.param_groups[0]["params"]))
    first.grad = torch.ones_like(first)
    trainer.optimizer.step()

    checkpoint = _wrapped_checkpoint(
        wrapper,
        epoch=2,
        optimizer=trainer.optimizer.state_dict(),
        drax=config.to_dict(),
    )
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)

    resumed_wrapper = LibreYOLO9(str(path), size="s", device="cpu")
    resumed = YOLO9Trainer(
        model=resumed_wrapper.model,
        wrapper_model=resumed_wrapper,
        size="s",
        num_classes=80,
        data="unused.yaml",
        device="cpu",
    )
    resumed.optimizer = resumed._setup_optimizer()
    resumed.resume(str(path))

    assert resumed.start_epoch == 3
    assert resumed_wrapper.drax_config == config
    assert resumed.optimizer.state


def test_resume_rejects_mismatching_drax_architecture(tmp_path):
    source = LibreYOLO9(None, size="s", device="cpu", drax_config=_b5_config())
    path = tmp_path / "last.pt"
    torch.save(
        _wrapped_checkpoint(source, epoch=0, drax=source.drax_config.to_dict()),
        path,
    )

    current = LibreYOLO9(None, size="s", device="cpu", drax_config=DraxConfig())
    trainer = YOLO9Trainer(
        model=current.model,
        wrapper_model=current,
        size="s",
        num_classes=80,
        data="unused.yaml",
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="Drax architecture differs"):
        trainer.resume(str(path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_forward_backward_and_step():
    wrapper = LibreYOLO9(
        None,
        size="s",
        nb_classes=3,
        device="cuda",
        drax_config=_b5_config(),
    )
    wrapper.model.train()
    optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=1e-3)
    targets = torch.zeros(1, 10, 5, device="cuda")
    targets[0, 0] = torch.tensor([0.0, 32.0, 32.0, 16.0, 16.0], device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = wrapper.model(
            torch.randn(1, 3, 64, 64, device="cuda"), targets=targets
        )["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
