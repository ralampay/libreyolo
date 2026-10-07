import copy

import pytest
import torch
from torch import nn

from libreyolo.adapters import (
    DraxResidualFusionConv2d, adapter_state_dict, inject_adapters,
    load_adapter_state_dict, yolox_targets,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('channels,shape', [((32, 16), (2, 32, 5, 7)), ((8, 12), (1, 8, 1, 3))])
def test_identity_shapes_and_frozen_base(channels, shape):
    base = nn.Conv2d(*channels, 1)
    adapter = DraxResidualFusionConv2d(base)
    x = torch.randn(shape)
    assert torch.equal(adapter(x), base(x))
    adapter(x).square().mean().backward()
    assert base.weight.grad is None
    assert adapter.up.weight.grad is not None


def test_spatial_gate_and_gradient_flow():
    torch.manual_seed(19)
    adapter = DraxResidualFusionConv2d(nn.Conv2d(32, 16, 1), reduction=4)
    x = torch.randn(2, 32, 5, 7)
    z = torch.randn(2, 8, 5, 7)
    assert torch.equal(adapter.gate(z).softmax(1), torch.full((2, 2, 5, 7), .5))
    with torch.no_grad():
        adapter.gate.weight[0, 0] = 1
    weights = adapter.gate(z).softmax(1)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 5, 7))
    assert weights[:, 0].std() > 0
    optimizer = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=.1)
    adapter(x).square().mean().backward()
    assert adapter.down.weight.grad.count_nonzero() == 0
    optimizer.step()
    optimizer.zero_grad()
    adapter(x).square().mean().backward()
    for name, parameter in adapter.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.count_nonzero() > 0, name


def test_internal_skip_preserves_bottleneck_features():
    adapter = DraxResidualFusionConv2d(nn.Conv2d(32, 16, 1), reduction=4)
    with torch.no_grad():
        for conv in (adapter.local, adapter.context):
            conv.weight.zero_()
            conv.bias.zero_()
        adapter.up.weight.normal_()
    x = torch.randn(2, 32, 5, 7)
    z = adapter.down(x)
    z = torch.nn.functional.silu(adapter.norm(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    expected = adapter.base(x) + adapter.up(torch.nn.functional.silu(z))
    torch.testing.assert_close(adapter(x), expected)
    assert not torch.equal(adapter(x), adapter.base(x))


@pytest.mark.parametrize('base', [nn.Identity(), nn.Conv2d(8, 8, 3), nn.Conv2d(8, 8, 1, stride=2), nn.Conv2d(8, 8, 1, padding=1), nn.Conv2d(8, 8, 1, groups=2)])
def test_invalid_base(base):
    with pytest.raises(ValueError, match='dense stride-one'):
        DraxResidualFusionConv2d(base)


@pytest.mark.parametrize('kwargs', [{'reduction': 0}, {'alpha': float('nan')}, {'alpha': float('inf')}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DraxResidualFusionConv2d(nn.Conv2d(8, 8, 1), **kwargs)


def test_injection_checkpoint_roundtrip_and_duplicate_rejection():
    model = nn.Sequential(nn.Conv2d(32, 16, 1))
    restored = copy.deepcopy(model)
    for target in (model, restored):
        inject_adapters(target, 'drax-residual-fusion', {'0': 16}, reduction=16)
    with torch.no_grad():
        model[0].up.weight.normal_()
    state = adapter_state_dict(model)
    load_adapter_state_dict(restored, state)
    x = torch.randn(2, 32, 5, 7)
    torch.testing.assert_close(model(x), restored(x))
    with pytest.raises(ValueError, match='state mismatch'):
        load_adapter_state_dict(restored, {})
    with pytest.raises(ValueError, match='already has an adapter'):
        inject_adapters(model, 'drax-residual-fusion', {'0': 16})


@pytest.mark.parametrize('method,rank,reduction,expected', [
    ('lora', 8, 8, 178176), ('lora', 100, 8, 2227200),
    ('drax-spatial', 8, 8, 2056724), ('drax-residual-fusion', 8, 16, 961724),
])
def test_yolox_l_parameter_budget(method, rank, reduction, expected):
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    model = LibreYOLOXModel(config='l', nb_classes=6)
    targets = yolox_targets(model, 'neck', method)
    assert len(targets) == 26
    assert targets == yolox_targets(model, 'neck', 'lora')
    report = inject_adapters(model, method, targets, rank=rank, reduction=reduction, train_head=True)
    head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    assert head == 7552801
    assert report.trainable_parameters == expected + head


def test_singleton_bottleneck_remains_input_dependent():
    adapter = DraxResidualFusionConv2d(nn.Conv2d(8, 4, 1), reduction=16)
    with torch.no_grad():
        adapter.up.weight.fill_(1)
    x = torch.randn(2, 8, 3, 5)
    adapter(x).square().mean().backward()
    assert adapter.down.weight.grad.count_nonzero() > 0


def test_full_model_initial_identity():
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    model = LibreYOLOXModel(config='n', nb_classes=6).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(x)
        inject_adapters(model, 'drax-residual-fusion', yolox_targets(model, 'neck', 'drax-residual-fusion'), reduction=16)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
