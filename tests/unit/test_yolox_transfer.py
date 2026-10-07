import pytest
import torch
from torch import nn

from libreyolo.models.yolox.transfer import reset_classifiers, transfer_state_dict, load_transfer_state_dict


def model():
    result = nn.Module()
    result.body = nn.Conv2d(3,4,1)
    result.head = nn.Module()
    result.head.cls_preds = nn.ModuleList([nn.Conv2d(4,6,1) for _ in range(3)])
    result.head.bn = nn.BatchNorm2d(4)
    result.head.obj_preds = nn.ModuleList([nn.Conv2d(4,1,1)])
    return result


@pytest.mark.parametrize("classes",[6,3])
def test_reset_same_or_different_taxonomy_preserves_body_and_objectness(classes):
    network = model()
    before = {k:v.clone() for k,v in network.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    initial = reset_classifiers(network,classes,seed=7)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert network.head.num_classes == classes
    for key,value in network.state_dict().items():
        if not key.startswith("head.cls_preds"):
            assert torch.equal(before[key],value)
    if classes == 6:
        assert not torch.equal(before["head.cls_preds.0.weight"],initial["0.weight"])
    other = model()
    assert all(torch.equal(v,reset_classifiers(other,classes,seed=7)[k]) for k,v in initial.items())


def test_transfer_roundtrip_includes_bn_and_rejects_missing_buffers():
    source,target = model(),model()
    source.body.requires_grad_(False)
    target.body.requires_grad_(False)
    source.head.bn.running_mean.fill_(3)
    state = transfer_state_dict(source)
    assert "body.weight" not in state
    assert "head.bn.num_batches_tracked" in state
    load_transfer_state_dict(target,state)
    assert all(torch.equal(v,target.state_dict()[k]) for k,v in state.items())
    del state["head.bn.running_mean"]
    with pytest.raises(ValueError,match="keys differ"):
        load_transfer_state_dict(target,state)
