import pytest
import torch
from torch import nn

from libreyolo.adapters import create_adapter, available_adapters, inject_adapters, count_parameters, yolox_targets, adapter_state_dict, load_adapter_state_dict
from libreyolo.adapters.layers import LoRAConv2d


@pytest.mark.unit
@pytest.mark.parametrize("name", ["bottleneck", "ssf", "convpass", "conv-adapter", "drax"])
def test_feature_adapter_identity_and_shape(name):
    adapter = create_adapter(name, channels=16, reduction=4)
    x = torch.randn(2, 16, 9, 11)
    assert adapter(x).shape == x.shape
    torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)


@pytest.mark.unit
def test_lora_is_a_low_rank_weight_update():
    base = nn.Conv2d(4, 6, 1)
    x = torch.randn(2, 4, 7, 7)
    expected = base(x).clone()
    layer = LoRAConv2d(base, rank=2)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert not base.weight.requires_grad
    assert layer.a.weight.requires_grad and layer.b.weight.requires_grad


@pytest.mark.unit
def test_registry_and_freezing():
    assert set(available_adapters()) == {"bottleneck", "ssf", "lora", "convpass", "conv-adapter", "drax"}
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Conv2d(4, 4, 1)
            self.head = nn.Conv2d(4, 2, 1)

    model = Toy()
    x = torch.randn(1, 4, 8, 8)
    expected = model.feature(x)
    report = inject_adapters(model, "drax", {"feature": 4}, reduction=2)
    torch.testing.assert_close(model.feature(x), expected, rtol=0, atol=0)
    assert report.trainable_parameters == count_parameters(model)["trainable"]
    assert all(not p.requires_grad for p in model.feature.base.parameters())
    assert all(p.requires_grad for p in model.feature.adapter.parameters())
    state = adapter_state_dict(model)
    load_adapter_state_dict(model, state)
    with pytest.raises(ValueError, match="missing"):
        load_adapter_state_dict(model, {})


@pytest.mark.unit
def test_yolox_l_neck_targets():
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    model = LibreYOLOXModel(config="l", nb_classes=6)
    assert yolox_targets(model, "neck") == {"backbone.C3_p3": 256, "backbone.C3_n3": 512,
                                             "backbone.C3_n4": 1024}
    assert yolox_targets(model) == yolox_targets(model, "neck")
    lora_targets = yolox_targets(model, "neck", "lora")
    assert lora_targets
    assert all(name.startswith("backbone.") and not name.startswith("backbone.backbone.")
               for name in lora_targets)
    report = inject_adapters(model, "lora", lora_targets, rank=4)
    assert report.trainable_parameters > 0
    assert all(not p.requires_grad for name, p in model.named_parameters() if ".base." in name)


@pytest.mark.unit
def test_parameter_matched_neck_adapters():
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    counts = {}
    for name in ("bottleneck", "conv-adapter", "drax"):
        model = LibreYOLOXModel(config="l", nb_classes=6)
        report = inject_adapters(model, name, yolox_targets(model, "neck", name), reduction=8)
        counts[name] = report.trainable_parameters
    assert max(counts.values()) / min(counts.values()) < 1.02


@pytest.mark.external_data
def test_foundational_checkpoint_strict_load_and_identity():
    from pathlib import Path
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    from libreyolo.utils.serialization import load_untrusted_torch_file
    path = Path.home() / "Desktop/object-detection-models/foundational-yolox-l.pt"
    if not path.is_file():
        pytest.skip("External foundation checkpoint unavailable")
    checkpoint = load_untrusted_torch_file(str(path), map_location="cpu")
    model = LibreYOLOXModel(config="l", nb_classes=checkpoint["nc"]).eval()
    result = model.load_state_dict(checkpoint["model"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        expected = model(x)
        for name in available_adapters():
            fresh = LibreYOLOXModel(config="l", nb_classes=checkpoint["nc"]).eval()
            fresh.load_state_dict(checkpoint["model"], strict=True)
            inject_adapters(fresh, name, yolox_targets(fresh, "neck", name))
            actual = fresh(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
