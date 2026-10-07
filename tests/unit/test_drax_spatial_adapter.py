import copy

import pytest
import torch
from torch import nn

from libreyolo.adapters import DraxHybridConv2d, DraxSpatialConv2d, inject_adapters, yolox_targets
from libreyolo.adapters.layers import LoRAConv2d

pytestmark = pytest.mark.unit


def test_spatial_matches_reference_initialization_without_lora():
    base = nn.Conv2d(16,32,1)
    torch.manual_seed(7)
    hybrid = DraxHybridConv2d(copy.deepcopy(base))
    rng = torch.random.get_rng_state()
    torch.manual_seed(7)
    spatial = DraxSpatialConv2d(copy.deepcopy(base))
    assert torch.equal(rng,torch.random.get_rng_state())
    for name,value in spatial.state_dict().items():
        reference = "linear."+name if name.startswith("base.") else name
        assert torch.equal(value,hybrid.state_dict()[reference])
    assert not any(isinstance(module,LoRAConv2d) for module in spatial.modules())
    x = torch.randn(2,16,8,8)
    assert torch.equal(spatial(x),base(x))
    spatial(x).square().mean().backward()
    assert spatial.up.weight.grad is not None
    assert spatial.base.weight.grad is None


@pytest.mark.parametrize("method,rank,alpha,expected",[("lora",100,12.5,2227200),("drax-spatial",8,1.,2056724)])
def test_yolox_l_ablation_counts_and_targets(method,rank,alpha,expected):
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    model = LibreYOLOXModel(config="l",nb_classes=6)
    targets = yolox_targets(model,"neck",method)
    assert len(targets) == 26
    assert targets == yolox_targets(model,"neck","drax-hybrid")
    report = inject_adapters(model,method,targets,rank=rank,alpha=alpha,train_head=True)
    assert report.trainable_parameters == expected+7552801
    if method == "lora":
        assert all(module.scale == .125 for module in model.modules() if isinstance(module,LoRAConv2d))
