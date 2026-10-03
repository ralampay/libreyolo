"""Offline architecture, reconstruction and gradient contracts."""

import importlib

import pytest
import torch
from libreyolo import LibreYOLO, LibreYOLOX, LibreYOLOXDraxMobileNetV3Large
from libreyolo.models.drax_mobilenet_v3.backbone import YOLOXBackbone

variants = importlib.import_module("libreyolo.models.yolox-drax-mobilenet-v3.variants")
network = importlib.import_module("libreyolo.models.yolox-drax-mobilenet-v3.nn")
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("preset", variants.ARCHITECTURE_VARIANTS)
def test_variant_shapes_gradients_and_rebuild(preset):
    wrapper = LibreYOLOXDraxMobileNetV3Large(
        None, size="n", nb_classes=6, device="cpu", architecture_variant=preset
    )
    backbone = wrapper.model.backbone.backbone
    features = backbone.forward_features(torch.rand(2, 3, 64, 96) * 255)
    assert [tuple(t.shape) for t in features] == [
        (2, 40, 8, 12),
        (2, 112, 4, 6),
        (2, 160 if preset == "pyramid-drax" else 960, 2, 3),
    ]
    sum(t.square().mean() for t in features).backward()
    for n, p in backbone.named_parameters():
        if any(
            s in n
            for s in (
                "feature_refiners",
                "p5_pool",
                "attention_scale",
            )
        ):
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
    wrapper._rebuild_for_new_classes(3)
    assert wrapper.model.backbone.backbone.architecture_variant == preset
    wrapper._reset_for_scratch(seed=42)
    assert wrapper.model.backbone.backbone.architecture_variant == preset


@pytest.mark.parametrize("preset", variants.ARCHITECTURE_VARIANTS)
def test_checkpoint_reload_and_worker_reconstruction(preset, tmp_path):
    from libreyolo.training.ddp_spawn import _build_init_kw

    wrapper = LibreYOLOXDraxMobileNetV3Large(
        None, size="n", nb_classes=6, device="cpu", architecture_variant=preset
    )
    wrapper.model.eval()
    sample = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        before = wrapper.model(sample)[0]
    path = tmp_path / "model.pt"
    wrapper.save(str(path))
    loaded = LibreYOLO(str(path), device="cpu")
    assert loaded.architecture_variant == preset
    with torch.no_grad():
        after = loaded.model(sample)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    kwargs = _build_init_kw(wrapper)
    assert kwargs["architecture_variant"] == preset
    raw = LibreYOLOXDraxMobileNetV3Large(
        wrapper.model.state_dict(), size="n", nb_classes=6, device="cpu"
    )
    assert raw.architecture_variant == preset
    with pytest.raises((ValueError, RuntimeError), match="differs from checkpoint"):
        LibreYOLOXDraxMobileNetV3Large(
            str(path),
            size="n",
            device="cpu",
            architecture_variant="spp-p5" if preset != "spp-p5" else "legacy",
        )


def test_legacy_backbone_outputs_are_unchanged():
    torch.manual_seed(42)
    original = YOLOXBackbone((64, 128, 256), input_bgr=True).eval()
    torch.manual_seed(42)
    current = network.IncrementalYOLOXBackbone((64, 128, 256), input_bgr=True).eval()
    assert set(original.state_dict()) == set(current.state_dict())
    image = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        a, b = original(image), current(image)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_residual_and_balanced_initialization():
    for block, c in [
        (variants.RefineSpatialFeatures(40), 40),
        (variants.PoolSpatialContext(), 960),
    ]:
        assert torch.all(block.scale == 1e-3)
        with torch.no_grad():
            block.scale.zero_()
        x = torch.rand(2, c, 4, 4)
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)
    balanced = variants.BalancedDraxBlock()
    assert torch.all(balanced.convnext.layer_scale == 1e-3)
    assert torch.all(balanced.attention_scale == 1e-3)
    pyramid = network.IncrementalYOLOXBackbone(
        (64, 128, 256), architecture_variant="pyramid-drax"
    )
    assert pyramid.feature_channels == (40, 112, 160)
    assert len(pyramid.features) == 16
    assert pyramid.projections[2][0].in_channels == 160


def test_invalid_or_inconsistent_preset():
    with pytest.raises(ValueError):
        variants.validate_variant("typo")
    for metadata in (
        {"version": 2, "preset": "legacy"},
        {"version": 1, "preset": "spp-p5"},
    ):
        with pytest.raises(ValueError):
            variants.variant_from_state({}, metadata)


def test_pyramid_drax_combines_repairs_below_yolox_m_parameter_budget():
    candidate = LibreYOLOXDraxMobileNetV3Large(
        None,
        size="m",
        nb_classes=6,
        device="cpu",
        architecture_variant="pyramid-drax",
    )
    baseline = LibreYOLOX(None, size="m", nb_classes=6, device="cpu")
    backbone = candidate.model.backbone.backbone

    assert hasattr(backbone, "feature_refiners")
    assert hasattr(backbone, "p5_pool")
    assert isinstance(backbone.drax_refiner[0], variants.BalancedDraxBlock)
    assert sum(p.numel() for p in candidate.model.parameters()) < sum(
        p.numel() for p in baseline.model.parameters()
    )


def test_partial_variant_marker_combinations_are_rejected():
    state = {
        "backbone.backbone.feature_refiners.0.scale": torch.zeros(1),
        "backbone.backbone.p5_pool.scale": torch.zeros(1),
    }
    with pytest.raises(ValueError, match="unsupported combination"):
        variants.variant_from_state(state)


def test_mobile_drax_training_recipe_clips_gradients_without_changing_yolox():
    from libreyolo.models.yolox.trainer import YOLOXTrainer

    trainer_module = importlib.import_module(
        "libreyolo.models.yolox-drax-mobilenet-v3.trainer"
    )

    drax = trainer_module.YOLOXDraxMobileNetV3LargeTrainer._config_class()()
    vanilla = YOLOXTrainer._config_class()()

    assert drax.clip_max_norm == 1.0
    assert not hasattr(vanilla, "clip_max_norm")
