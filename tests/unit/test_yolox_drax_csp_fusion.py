"""Contracts for the parameter-efficient YOLOX CSP-Drax fusion family."""

import pytest
import torch

from libreyolo import LibreYOLO, LibreYOLOX, LibreYOLOXDraxCSPFusionM
from libreyolo.models.yolox_drax_csp_fusion.nn import BranchAttentionFusion
from libreyolo.models.yolox_drax_csp_fusion.transfer import (
    InitializeYOLOXSharedTransfer,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_branch_attention_starts_as_exact_scale_preserving_pass_through():
    fusion = BranchAttentionFusion(16)
    first = torch.randn(2, 16, 5, 5)
    second = torch.randn(2, 16, 5, 5)
    actual_first, actual_second = fusion(first, second)
    torch.testing.assert_close(actual_first, first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, second, rtol=0, atol=0)


def test_fusion_model_is_smaller_than_yolox_m_and_has_finite_gradients():
    candidate = LibreYOLOXDraxCSPFusionM(None, nb_classes=6, device="cpu")
    control = LibreYOLOX(None, size="m", nb_classes=6, device="cpu")
    assert sum(p.numel() for p in candidate.model.parameters()) < sum(
        p.numel() for p in control.model.parameters()
    )

    images = torch.rand(2, 3, 64, 64) * 255
    features = candidate.model.backbone(images)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 192, 8, 8),
        (2, 384, 4, 4),
        (2, 768, 2, 2),
    ]
    sum(feature.square().mean() for feature in features).backward()
    parameters = dict(candidate.model.backbone.named_parameters())
    for name in (
        "p5_context_scale",
        "p5_drax.attention.qkv.weight",
        "fuse_p4_top_down.select.weight",
        "fuse_p3_top_down.select.weight",
        "fuse_p4_bottom_up.select.weight",
        "fuse_p5_bottom_up.select.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_fusion_model_computes_a_finite_detection_loss_and_backward_pass():
    detector = LibreYOLOXDraxCSPFusionM(None, nb_classes=2, device="cpu")
    detector.model.train()
    images = torch.rand(2, 3, 64, 64) * 255
    targets = torch.zeros(2, 4, 5)
    targets[:, 0] = torch.tensor([0.0, 32.0, 32.0, 16.0, 16.0])
    losses = detector.model(images, targets)
    assert torch.isfinite(losses["total_loss"])
    assert losses["total_loss"] > 0
    losses["total_loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in detector.model.parameters()
    )


def test_shared_transfer_copies_the_same_body_and_rejects_wrong_size():
    source = LibreYOLOX(None, size="m", nb_classes=80, device="cpu")
    control = LibreYOLOX(None, size="m", nb_classes=4, device="cpu")
    candidate = LibreYOLOXDraxCSPFusionM(None, nb_classes=4, device="cpu")
    report = InitializeYOLOXSharedTransfer(
        source.model.state_dict(), control, candidate, reset_seed=17
    ).execute()
    assert report.transferred_tensors > 0
    source_state = source.model.state_dict()
    for key, value in control.model.state_dict().items():
        if key.startswith("backbone."):
            torch.testing.assert_close(value, source_state[key], rtol=0, atol=0)
    for key, value in candidate.model.state_dict().items():
        if key in source_state and key.startswith("backbone."):
            torch.testing.assert_close(value, source_state[key], rtol=0, atol=0)

    wrong_size = LibreYOLOX(None, size="s", nb_classes=4, device="cpu")
    with pytest.raises(ValueError, match="not YOLOX-M body-compatible"):
        InitializeYOLOXSharedTransfer(
            source.model.state_dict(), wrong_size, candidate
        ).execute()


def test_fusion_checkpoint_round_trip_and_registry_discrimination(tmp_path):
    detector = LibreYOLOXDraxCSPFusionM(None, nb_classes=6, device="cpu")
    checkpoint = tmp_path / "LibreYOLOXDraxCSPFusionm.pt"
    detector.save(str(checkpoint))
    loaded = LibreYOLO(str(checkpoint), device="cpu")
    assert isinstance(loaded, LibreYOLOXDraxCSPFusionM)
    assert loaded.nb_classes == 6
    assert not LibreYOLOX.can_load(loaded.model.state_dict())
    detector._rebuild_for_new_classes(3)
    assert detector.model.head.num_classes == 3


def test_shared_transfer_rejects_incomplete_body_before_mutation():
    control = LibreYOLOX(None, size="m", nb_classes=2, device="cpu")
    candidate = LibreYOLOXDraxCSPFusionM(None, size="m", nb_classes=2, device="cpu")
    state = control.model.state_dict()
    key = next(key for key in state if key.startswith("backbone."))
    before = state[key].clone()
    with pytest.raises(ValueError, match="missing required YOLOX body tensors"):
        InitializeYOLOXSharedTransfer({key: before + 1}, control, candidate).execute()
    torch.testing.assert_close(control.model.state_dict()[key], before, rtol=0, atol=0)
