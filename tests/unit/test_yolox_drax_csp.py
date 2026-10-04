"""Architecture and checkpoint contracts for the scratch CSP–Drax detector."""

import pytest
import torch

from libreyolo import LibreYOLO, LibreYOLOX, LibreYOLOXDraxCSPM
from libreyolo.models.yolox_drax_csp.trainer import YOLOXDraxCSPMConfig

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_csp_drax_budget_features_and_gradients():
    detector = LibreYOLOXDraxCSPM(None, nb_classes=6, device="cpu")
    baseline = LibreYOLOX(None, size="m", nb_classes=6, device="cpu")
    assert sum(p.numel() for p in detector.model.parameters()) == 24_596_306
    assert sum(p.numel() for p in detector.model.parameters()) < sum(
        p.numel() for p in baseline.model.parameters()
    )
    x = torch.rand(2, 3, 64, 64) * 255
    features = detector.model.backbone.backbone(x)
    assert [tuple(t.shape) for t in features.values()] == [
        (2, 176, 8, 8), (2, 352, 4, 4), (2, 704, 2, 2),
    ]
    sum(feature.square().mean() for feature in features.values()).backward()
    backbone = detector.model.backbone.backbone
    for name in (
        "p3_refiner.scale",
        "p5_drax.convnext.layer_scale",
        "p5_drax.attention.qkv.weight",
        "p5_scale",
    ):
        gradient = dict(backbone.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_csp_drax_checkpoint_reloads_without_changing_other_families(tmp_path):
    detector = LibreYOLOXDraxCSPM(None, nb_classes=6, device="cpu")
    detector.model.eval()
    x = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        expected = detector.model(x)
    checkpoint = tmp_path / "LibreYOLOXDraxCSPm.pt"
    detector.save(str(checkpoint))
    loaded = LibreYOLO(str(checkpoint), device="cpu")
    assert isinstance(loaded, LibreYOLOXDraxCSPM)
    assert loaded.nb_classes == 6
    with torch.no_grad():
        for actual, original in zip(loaded.model(x), expected):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not LibreYOLOX.can_load(loaded.model.state_dict())
    detector._rebuild_for_new_classes(3)
    assert detector.model.head.num_classes == 3


def test_csp_drax_does_not_apply_family_specific_gradient_clipping():
    config = YOLOXDraxCSPMConfig()
    assert not hasattr(config, "clip_max_norm")
