import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libreyolo.adapters import (
    DraxHybridConv2d, adapter_state_dict, inject_adapters,
    load_adapter_state_dict, yolox_targets,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_identity_and_exact_low_rank_weight_algebra():
    base = nn.Conv2d(12, 16, 1)
    layer = DraxHybridConv2d(base, rank=3, reduction=4, alpha=2)
    x = torch.randn(2, 12, 7, 9)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    nn.init.normal_(layer.linear.b.weight)
    delta = layer.linear.b.weight.flatten(1) @ layer.linear.a.weight.flatten(1)
    expected = F.conv2d(x, base.weight + (2 / 3) * delta[:, :, None, None], base.bias)
    torch.testing.assert_close(layer(x), expected)


def test_both_paths_learn_and_frozen_weights_stay_exact():
    layer = DraxHybridConv2d(nn.Conv2d(8, 8, 1), rank=2, reduction=2)
    frozen = {k: p.detach().clone() for k, p in layer.named_parameters() if not p.requires_grad}
    initial = {k: p.detach().clone() for k, p in layer.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=0.01)
    x = torch.randn(2, 8, 7, 9)
    for _ in range(3):
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
        optimizer.step()
    parameters = dict(layer.named_parameters())
    for name, before in frozen.items():
        assert parameters[name].grad is None
        assert torch.equal(before, parameters[name])
    for name in ("linear.a.weight", "linear.b.weight", "down.weight", "local.weight",
                 "context.weight", "mix.weight", "up.weight", "fusion_logits"):
        assert not torch.equal(initial[name], parameters[name]), name


@pytest.mark.parametrize("placement", ["backbone", "neck", "backbone+neck"])
def test_yolox_injection_identity_and_checkpoint_roundtrip(placement):
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    model = LibreYOLOXModel(config="l", nb_classes=6).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(x)
    targets = yolox_targets(model, placement, "drax-hybrid")
    assert targets == yolox_targets(model, placement, "lora")
    report = inject_adapters(model, "drax-hybrid", targets)
    assert report.frozen_parameters == 54_151_841
    with torch.no_grad():
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    if placement == "neck":
        assert len(targets) == 26
        assert report.trainable_parameters == 2_234_900
    with pytest.raises(ValueError, match="already has an adapter"):
        inject_adapters(model, "drax-hybrid", targets)
    state = adapter_state_dict(model)
    load_adapter_state_dict(model, state)
    malformed = dict(state)
    malformed[next(iter(state))] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        load_adapter_state_dict(model, malformed)


@pytest.mark.parametrize("base", [nn.Conv2d(4, 4, 3), nn.Conv2d(4, 4, 1, stride=2)])
def test_unsupported_convolutions_rejected(base):
    with pytest.raises(ValueError, match="stride-one"):
        DraxHybridConv2d(base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
@pytest.mark.e2e
def test_cuda_injection_keeps_all_tensors_on_device():
    model = nn.Sequential(nn.Conv2d(8, 8, 1)).cuda()
    inject_adapters(model, "drax-hybrid", {"0": 8})
    assert all(p.is_cuda for p in model.parameters())
    with torch.autocast("cuda"):
        loss = model(torch.randn(2, 8, 8, 8, device="cuda")).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
