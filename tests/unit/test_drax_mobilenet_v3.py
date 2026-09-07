"""Offline contracts for the Drax MobileNetV3 Large detector variants."""

import importlib
import pickle

import pytest
import torch
from torchvision.models import mobilenet_v3_large

from libreyolo import (
    LibreYOLO,
    LibreYOLO9DraxMobileNetV3Large,
    LibreYOLOXDraxMobileNetV3Large,
)
from libreyolo.models.drax_mobilenet_v3.backbone import DraxMobileNetV3LargeBackbone
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolox.model import LibreYOLOX

pytestmark = pytest.mark.unit
VARIANTS = (LibreYOLO9DraxMobileNetV3Large, LibreYOLOXDraxMobileNetV3Large)


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _prediction(model, sample):
    output = model(sample)
    return output["predictions"] if isinstance(output, dict) else output[0]


@pytest.mark.parametrize("cls", VARIANTS)
def test_package_and_ddp_class_resolution(cls):
    from libreyolo.training.ddp_spawn import _build_init_kw

    wrapper = cls(None, size="s", nb_classes=2, device="cpu")
    package = cls.CLI_ALIASES[0].removesuffix("-large")
    assert cls.__module__ == f"libreyolo.models.{package}.model"
    kwargs = _build_init_kw(wrapper)
    module = importlib.import_module(kwargs.pop("_module"))
    worker_class = getattr(module, kwargs.pop("_class"))
    assert worker_class is cls
    assert pickle.loads(pickle.dumps(cls)) is cls
    assert kwargs["size"] == "s"
    assert kwargs["nb_classes"] == 2


def test_backbone_features_and_pixel_contract():
    reference = mobilenet_v3_large(weights=None).eval()
    rgb = DraxMobileNetV3LargeBackbone((40, 112, 960)).eval()
    rgb.features.load_state_dict(reference.features.state_dict())
    bgr = DraxMobileNetV3LargeBackbone((40, 112, 960), input_bgr=True).eval()
    bgr.load_state_dict(rgb.state_dict())
    image = torch.rand(2, 3, 64, 96)
    with torch.no_grad():
        features = rgb.forward_features(image)
        bgr_features = bgr.forward_features(image[:, [2, 1, 0]] * 255)
        expected = (image - rgb.mean) / rgb.std
        for index, layer in enumerate(reference.features):
            expected = layer(expected)
            if index in (6, 12):
                torch.testing.assert_close(features[(6, 12).index(index)], expected)
    assert [tuple(x.shape) for x in features] == [
        (2, 40, 8, 12),
        (2, 112, 4, 6),
        (2, 960, 2, 3),
    ]
    for a, b in zip(features, bgr_features):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "cls,size", [(cls, size) for cls in VARIANTS for size in cls.INPUT_SIZES]
)
def test_all_sizes_forward_and_routing(cls, size):
    wrapper = cls(None, size=size, nb_classes=3, device="cpu")
    wrapper.model.eval()
    state = wrapper.model.state_dict()
    assert cls.detect_size(state) == size
    assert cls.can_load(state)
    assert not LibreYOLO9.can_load(state)
    assert not LibreYOLOX.can_load(state)
    assert not next(other for other in VARIANTS if other is not cls).can_load(state)
    with torch.no_grad():
        prediction = _prediction(wrapper.model, torch.rand(1, 3, 64, 64))
    assert torch.isfinite(prediction).all()
    assert cls.convert_upstream_state_dict(state) is None


@pytest.mark.parametrize("cls", VARIANTS)
def test_loss_backward_reaches_mobile_and_drax(cls):
    wrapper = cls(None, size="s", nb_classes=2, device="cpu")
    sample = torch.rand(2, 3, 64, 64)
    targets = torch.zeros(2, 2, 5)
    if cls is VARIANTS[0]:
        targets[:, 0] = torch.tensor([1, 0.25, 0.25, 0.75, 0.75])
    else:
        sample *= 255
        targets[:, 0] = torch.tensor([1, 32, 32, 32, 32])
    losses = wrapper.model(sample, targets=targets)
    loss = losses.get("total_loss", losses.get("loss"))
    assert torch.isfinite(loss)
    loss.backward()
    backbone = wrapper._mobile_backbone()
    for parameter in (
        backbone.features[0][0].weight,
        backbone.drax_refiner[0].attention.qkv.weight,
        backbone.adapter_up.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    before = backbone.adapter_up.weight.detach().clone()
    torch.optim.SGD(wrapper.model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, backbone.adapter_up.weight)


@pytest.mark.parametrize("cls", VARIANTS)
def test_checkpoint_reload_class_rebuild_and_rejection(cls, tmp_path, monkeypatch):
    wrapper = cls(None, size="s", nb_classes=3, device="cpu")
    monkeypatch.setattr(
        DraxMobileNetV3LargeBackbone,
        "load_imagenet_weights",
        lambda _: pytest.fail("Unexpected download"),
    )
    before = wrapper._mobile_backbone().features[0][0].weight.detach().clone()
    wrapper._rebuild_for_new_classes(2)
    wrapper.names = {0: "a", 1: "b"}
    torch.testing.assert_close(before, wrapper._mobile_backbone().features[0][0].weight)
    wrapper.model.eval()
    sample = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = _prediction(wrapper.model, sample)
    path = wrapper.save(str(tmp_path / "custom.pt"))
    loaded = LibreYOLO(path, device="cpu")
    assert type(loaded) is cls
    assert loaded.nb_classes == 2
    with torch.no_grad():
        torch.testing.assert_close(
            _prediction(loaded.model, sample), expected, rtol=0, atol=0
        )
    loaded._initialize_training_weights(True, False)
    loaded._initialize_training_weights(True, True)
    assert len(loaded._get_available_layers()) == 4
    bad = dict(wrapper.model.state_dict())
    bad.pop(next(key for key in bad if "adapter_down.weight" in key))
    with pytest.raises(RuntimeError):
        cls(bad, size="s", nb_classes=2, device="cpu")
    parent = LibreYOLO9 if cls is VARIANTS[0] else LibreYOLOX
    with pytest.raises(RuntimeError, match="model_family"):
        parent(path, size="s", device="cpu")


@pytest.mark.parametrize("cls", VARIANTS)
def test_explicit_imagenet_transfer_and_no_download_on_construction(cls, monkeypatch):
    from torchvision import models

    reference = mobilenet_v3_large(weights=None)
    with torch.no_grad():
        reference.features[0][0].weight.fill_(0.125)
    real = models.mobilenet_v3_large
    requests = []

    def factory(*, weights):
        requests.append(weights)
        return reference if weights is not None else real(weights=None)

    monkeypatch.setattr(models, "mobilenet_v3_large", factory)
    wrapper = cls(None, size="s", device="cpu")
    assert all(weight is None for weight in requests)
    adapter = wrapper._mobile_backbone().adapter_down.weight.detach().clone()
    wrapper._initialize_training_weights(True, False)
    torch.testing.assert_close(
        wrapper._mobile_backbone().features[0][0].weight,
        reference.features[0][0].weight,
    )
    torch.testing.assert_close(adapter, wrapper._mobile_backbone().adapter_down.weight)
    wrapper._initialize_training_weights(True, False)
    assert sum(weight is not None for weight in requests) == 1
    assert wrapper._mobile_backbone().features[0][1].eps == reference.features[0][1].eps
    assert wrapper._mobile_backbone().adapter_norm.eps == 1e-5
    with pytest.raises(ValueError, match="pretrained"):
        wrapper._initialize_training_weights("some.pt", False)


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("pretrained", [True, False])
def test_cli_variant_construction(cls, pretrained):
    from libreyolo.cli.commands.train import _create_explicit_task_train_model
    from libreyolo.cli.config import detect_family_from_name, resolve_model_name

    alias = cls.CLI_ALIASES[0]
    filename = resolve_model_name(alias)
    assert detect_family_from_name(alias) == cls.FAMILY
    assert filename == f"{cls.FILENAME_PREFIX}s.pt"
    assert resolve_model_name(f"{alias}-m") == f"{cls.FILENAME_PREFIX}m.pt"
    model = _create_explicit_task_train_model(
        family=cls.FAMILY,
        model_path=filename,
        task=None,
        resume=False,
        device="cpu",
        pretrained=pretrained,
    )
    assert type(model) is cls
    assert model.size == "s"


@pytest.mark.parametrize("cls", VARIANTS)
def test_freeze_groups_cover_backbone(cls):
    wrapper = cls(None, size="s", device="cpu")
    trainer = wrapper._trainer_class()(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data="unused.yaml",
        device="cpu",
    )
    assert trainer.get_model_family() == cls.FAMILY
    parameters = {
        id(p) for _, module in trainer.get_freeze_groups() for p in module.parameters()
    }
    assert {id(p) for p in wrapper.model.parameters()} <= parameters


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("flags", [False, True])
def test_cli_train_both_grammars(cls, flags):
    import json

    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.commands.train import train_cmd
    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()
    app.command(cls=KeyValueCommand)(train_cmd)
    values = {"model": cls.CLI_ALIASES[0], "data": "unused.yaml"}
    args = (
        [part for k, v in values.items() for part in (f"--{k}", v)]
        + ["--no-pretrained"]
        if flags
        else [f"{k}={v}" for k, v in values.items()] + ["pretrained=false"]
    )
    result = CliRunner().invoke(app, [*args, "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    assert cls.FAMILY in json.dumps(json.loads(result.stdout))


@pytest.mark.parametrize("cls", VARIANTS)
def test_public_train_initializes_backbone_only(cls, monkeypatch):
    import libreyolo.data

    monkeypatch.setattr(
        libreyolo.data,
        "load_data_config",
        lambda *args, **kwargs: {"nc": 2, "names": ["a", "b"]},
    )
    wrapper = cls(None, size="s", device="cpu")
    calls = []
    monkeypatch.setattr(
        DraxMobileNetV3LargeBackbone,
        "load_imagenet_weights",
        lambda self: calls.append(self),
    )
    trainer_cls = wrapper._trainer_class()
    monkeypatch.setattr(trainer_cls, "train", lambda self: {})
    wrapper.train(data="unused.yaml", pretrained=True, device="cpu")
    assert calls == [wrapper._mobile_backbone()]
    wrapper.train(data="unused.yaml", pretrained=False, device="cpu")
    assert len(calls) == 1


@pytest.mark.parametrize("cls", VARIANTS)
def test_missing_explicit_checkpoint_is_not_imagenet_fallback(cls):
    from libreyolo.cli.commands.train import _create_explicit_task_train_model

    result = _create_explicit_task_train_model(
        family=cls.FAMILY,
        model_path=f"./{cls.FILENAME_PREFIX}s.pt",
        task=None,
        resume=False,
        device="cpu",
        pretrained=True,
    )
    assert result is None


@pytest.mark.parametrize("cls", VARIANTS)
def test_fixed_variant_rejects_standard_yolo9_drax_config(cls):
    from libreyolo.models.yolo9.drax import DraxConfig

    with pytest.raises(ValueError, match="fixed Drax adapter"):
        cls(None, size="s", device="cpu", drax_config=DraxConfig(enabled=False))
