"""IncrementalAdapter contracts for YOLOX-Drax-MobileNetV3 only."""

import importlib

import pytest
import torch

from libreyolo import LibreYOLO, LibreYOLOXDraxMobileNetV3Large

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    from libreyolo.cli.config import _CLI_NAME_TO_WEIGHTS

    _CLI_NAME_TO_WEIGHTS.clear()


def _adapter_class():
    module = importlib.import_module(
        "libreyolo.models.yolox-drax-mobilenet-v3.nn"
    )
    return module.IncrementalAdapter


def _wrapper(nb_classes=2):
    wrapper = LibreYOLOXDraxMobileNetV3Large(
        None, size="s", nb_classes=nb_classes, device="cpu"
    )
    # Adapter-only training requires an already-trained detector. Unit tests use
    # this in-memory state as their hermetic foundation checkpoint surrogate.
    wrapper._has_detector_weights = True
    return wrapper


def _trainer(wrapper, **kwargs):
    return wrapper._trainer_class()(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=wrapper.nb_classes,
        data="unused.yaml",
        device="cpu",
        **kwargs,
    )


def _prediction(model, sample):
    output = model(sample)
    return output["predictions"] if isinstance(output, dict) else output[0]


def test_incremental_adapter_preserves_shape_dtype_and_zero_init_identity():
    adapter = _adapter_class()(32, reduction=16, spatial=True, alpha=0.75)
    sample = torch.randn(2, 32, 7, 11, dtype=torch.float32)
    output = adapter(sample)
    assert output.shape == sample.shape
    assert output.dtype == sample.dtype
    assert output.device == sample.device
    torch.testing.assert_close(output, sample, rtol=0, atol=0)
    assert torch.count_nonzero(adapter.up.weight) == 0
    assert torch.count_nonzero(adapter.up.bias) == 0


def test_attached_adapters_inherit_foundation_device_and_dtype():
    wrapper = _wrapper()
    wrapper.model.to(dtype=torch.float64)
    wrapper.attach_incremental_adapters()
    for parameter in wrapper._mobile_backbone().incremental_adapters.parameters():
        assert parameter.device == wrapper.device
        assert parameter.dtype == torch.float64


def test_adapters_are_absent_disabled_by_default_and_hard_bypassed():
    wrapper = _wrapper()
    backbone = wrapper._mobile_backbone()
    assert not backbone.incremental_adapters
    assert backbone.incremental_adapter_enabled is False
    assert not any("incremental_adapters." in key for key in wrapper.model.state_dict())

    wrapper.model.eval()
    sample = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        original = _prediction(wrapper.model, sample)
    wrapper.attach_incremental_adapters()
    with torch.no_grad():
        disabled = _prediction(wrapper.model, sample)
    torch.testing.assert_close(disabled, original, rtol=0, atol=0)

    with torch.no_grad():
        backbone.incremental_adapters["p3"].up.bias.fill_(0.5)
    wrapper.enable_incremental_adapters()
    with torch.no_grad():
        enabled = _prediction(wrapper.model, sample)
    assert not torch.equal(enabled, original)
    wrapper.disable_incremental_adapters()
    with torch.no_grad():
        bypassed = _prediction(wrapper.model, sample)
    torch.testing.assert_close(bypassed, original, rtol=0, atol=0)


def test_fresh_enabled_adapters_preserve_foundation_predictions():
    wrapper = _wrapper()
    wrapper.model.eval()
    sample = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        expected = _prediction(wrapper.model, sample)
    wrapper.attach_incremental_adapters()
    wrapper.enable_incremental_adapters()
    with torch.no_grad():
        actual = _prediction(wrapper.model, sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_foundation_checkpoint_loads_strictly_and_matches(tmp_path):
    wrapper = _wrapper()
    wrapper.model.eval()
    sample = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        expected = _prediction(wrapper.model, sample)
    path = wrapper.save(str(tmp_path / "foundation.pt"))

    loaded = LibreYOLO(path, device="cpu")
    assert type(loaded) is LibreYOLOXDraxMobileNetV3Large
    assert not loaded._mobile_backbone().incremental_adapters
    with torch.no_grad():
        actual = _prediction(loaded.model, sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_checkpoint_save_restore_preserves_structure_state_and_output(
    tmp_path,
):
    wrapper = _wrapper()
    wrapper.attach_incremental_adapters(
        reduction=8, spatial=False, alpha=0.25, features=("p3", "p5")
    )
    wrapper.enable_incremental_adapters()
    with torch.no_grad():
        wrapper._mobile_backbone().incremental_adapters["p3"].up.bias.fill_(0.125)
    wrapper.model.eval()
    sample = torch.rand(1, 3, 64, 64) * 255
    with torch.no_grad():
        expected = _prediction(wrapper.model, sample)
    path = wrapper.save(str(tmp_path / "incremental.pt"))

    loaded = LibreYOLO(path, device="cpu")
    config = loaded._mobile_backbone().incremental_adapter_config()
    assert config["enabled"] is True
    assert config["features"] == ["p3", "p5"]
    assert config["spatial"] is False
    assert config["alpha"] == pytest.approx(0.25)
    with torch.no_grad():
        actual = _prediction(loaded.model, sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_enabled_raw_state_dict_restores_strictly():
    wrapper = _wrapper()
    wrapper.attach_incremental_adapters(reduction=4, features=("p4",))
    wrapper.enable_incremental_adapters()
    state = wrapper.model.state_dict()
    loaded = LibreYOLOXDraxMobileNetV3Large(
        state, size="s", nb_classes=2, device="cpu"
    )
    config = loaded._mobile_backbone().incremental_adapter_config()
    assert config["features"] == ["p4"]
    assert config["hidden_channels"] == {"p4": 64}
    assert config["enabled"] is True


def test_adapter_only_freezes_foundation_and_optimizer_membership():
    wrapper = _wrapper()
    trainer = _trainer(
        wrapper,
        incremental_adapter=True,
        incremental_adapter_train_only=True,
    )
    trainer._apply_freeze_config()
    named = dict(wrapper.model.named_parameters())
    trainable = {name for name, parameter in named.items() if parameter.requires_grad}
    assert trainable
    assert all("incremental_adapters." in name for name in trainable)
    assert not named["backbone.backbone.adapter_down.weight"].requires_grad
    assert not named["backbone.backbone.adapter_up.weight"].requires_grad
    assert not any(
        parameter.requires_grad for parameter in wrapper.model.head.parameters()
    )

    optimizer = trainer._setup_optimizer()
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {
        id(parameter)
        for name, parameter in named.items()
        if "incremental_adapters." in name
    }
    report = wrapper.incremental_adapter_parameter_report()
    assert report["incremental_adapter_parameters"] == report["trainable_parameters"]
    assert report["foundation_parameters"] == report["frozen_parameters"]


def test_adapter_only_gradients_and_frozen_batchnorm_statistics():
    wrapper = _wrapper()
    trainer = _trainer(
        wrapper,
        incremental_adapter=True,
        incremental_adapter_train_only=True,
    )
    trainer._apply_freeze_config()
    backbone = wrapper._mobile_backbone()
    bn = backbone.features[0][1]
    before_mean = bn.running_mean.detach().clone()
    before_var = bn.running_var.detach().clone()

    wrapper.model.train()
    trainer._enforce_frozen_bn_eval()
    sample = torch.rand(2, 3, 64, 64) * 255
    targets = torch.zeros(2, 2, 5)
    targets[:, 0] = torch.tensor([1, 32, 32, 32, 32])
    loss = wrapper.model(sample, targets=targets)["total_loss"]
    loss.backward()

    torch.testing.assert_close(bn.running_mean, before_mean, rtol=0, atol=0)
    torch.testing.assert_close(bn.running_var, before_var, rtol=0, atol=0)
    for name, parameter in wrapper.model.named_parameters():
        if "incremental_adapters." in name:
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name


def test_train_head_ablation_unfreezes_only_adapters_and_head():
    wrapper = _wrapper()
    trainer = _trainer(
        wrapper,
        incremental_adapter=True,
        incremental_adapter_train_only=True,
        incremental_adapter_train_head=True,
    )
    trainer._apply_freeze_config()
    for name, parameter in wrapper.model.named_parameters():
        expected = "incremental_adapters." in name or name.startswith("head.")
        assert parameter.requires_grad is expected, name


def test_normal_training_mode_has_no_adapter_parameters_and_stays_trainable():
    wrapper = _wrapper()
    trainer = _trainer(wrapper)
    trainer._apply_freeze_config()
    assert not wrapper._mobile_backbone().incremental_adapters
    assert all(parameter.requires_grad for parameter in wrapper.model.parameters())


def test_disabled_loaded_adapters_are_excluded_from_normal_optimizer():
    wrapper = _wrapper()
    wrapper.attach_incremental_adapters()
    trainer = _trainer(wrapper, incremental_adapter=False)
    trainer._apply_freeze_config()
    optimizer = trainer._setup_optimizer()
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in wrapper.model.named_parameters():
        if "incremental_adapters." in name:
            assert not parameter.requires_grad
            assert id(parameter) not in optimized
        else:
            assert parameter.requires_grad


def test_adapter_only_requires_a_loaded_foundation_checkpoint():
    wrapper = LibreYOLOXDraxMobileNetV3Large(
        None, size="s", nb_classes=2, device="cpu"
    )
    with pytest.raises(ValueError, match="loaded foundation detector checkpoint"):
        _trainer(
            wrapper,
            incremental_adapter=True,
            incremental_adapter_train_only=True,
        )


@pytest.mark.parametrize("key_value", [True, False])
def test_cli_adapter_options_support_both_grammars(key_value):
    import json

    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.commands.train import train_cmd
    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()
    app.command(cls=KeyValueCommand)(train_cmd)
    base = [
        "model=yolox-drax-mobilenet-v3-large",
        "data=unused.yaml",
        "incremental_adapter=true",
        "incremental_adapter_train_only=true",
        "incremental_adapter_spatial=false",
    ]
    args = (
        base
        if key_value
        else [
            "--model",
            "yolox-drax-mobilenet-v3-large",
            "--data",
            "unused.yaml",
            "--incremental-adapter",
            "--incremental-adapter-train-only",
            "--no-incremental-adapter-spatial",
        ]
    )
    result = CliRunner().invoke(app, [*args, "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    config = json.loads(result.stdout)["resolved_config"]
    assert config["incremental_adapter"] is True
    assert config["incremental_adapter_train_only"] is True
    assert config["incremental_adapter_spatial"] is False


def test_family_cli_builder_forwards_incremental_options():
    from libreyolo.cli.config import build_family_train_kwargs

    params = {
        "epochs": 1,
        "incremental_adapter": True,
        "incremental_adapter_train_only": True,
        "incremental_adapter_reduction": 8,
        "incremental_adapter_spatial": False,
        "incremental_adapter_alpha": 0.5,
        "incremental_adapter_train_head": False,
        "incremental_adapter_features": "p3,p5",
    }
    kwargs = build_family_train_kwargs(
        params, "yolox_drax_mobilenet_v3_large"
    )
    assert kwargs["incremental_adapter"] is True
    assert kwargs["incremental_adapter_reduction"] == 8
    assert kwargs["incremental_adapter_features"] == "p3,p5"


def test_python_train_api_reaches_family_trainer(monkeypatch):
    import libreyolo.data

    monkeypatch.setattr(
        libreyolo.data,
        "load_data_config",
        lambda *args, **kwargs: {"nc": 2, "names": ["a", "b"]},
    )
    wrapper = _wrapper()
    captured = {}

    def fake_train(trainer):
        captured["config"] = trainer.config
        trainer._apply_freeze_config()
        return {}

    monkeypatch.setattr(wrapper._trainer_class(), "train", fake_train)
    results = wrapper.train(
        data="unused.yaml",
        incremental_adapter=True,
        incremental_adapter_train_only=True,
        incremental_adapter_reduction=8,
        incremental_adapter_features="p3,p5",
        device="cpu",
    )
    assert captured["config"].incremental_adapter_reduction == 8
    assert captured["config"].incremental_adapter_features == "p3,p5"
    assert results["training_time_seconds"] >= 0
    assert results["parameter_counts"]["trainable_parameters"] == results[
        "parameter_counts"
    ]["incremental_adapter_parameters"]


def test_adapter_only_rejects_class_count_change_without_rebuilding_head(monkeypatch):
    import libreyolo.data

    monkeypatch.setattr(
        libreyolo.data,
        "load_data_config",
        lambda *args, **kwargs: {"nc": 3, "names": ["a", "b", "c"]},
    )
    wrapper = _wrapper(nb_classes=2)
    original_head = wrapper.model.head
    with pytest.raises(ValueError, match="same class count and class indices"):
        wrapper.train(
            data="unused.yaml",
            incremental_adapter=True,
            incremental_adapter_train_only=True,
            device="cpu",
        )
    assert wrapper.model.head is original_head
    assert wrapper.nb_classes == 2


def test_cli_rejects_adapter_options_for_other_families():
    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.commands.train import train_cmd
    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()
    app.command(cls=KeyValueCommand)(train_cmd)
    result = CliRunner().invoke(
        app,
        [
            "model=yolox-s",
            "data=unused.yaml",
            "incremental_adapter=true",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "only for yolox-drax-mobilenet-v3-large" in result.stdout


def test_cli_help_json_exposes_incremental_adapter_options():
    import json

    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.commands.train import train_cmd
    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer()
    app.command(cls=KeyValueCommand)(train_cmd)
    result = CliRunner().invoke(app, ["--help-json"])
    assert result.exit_code == 0, result.output
    names = {item["name"] for item in json.loads(result.stdout)["parameters"]}
    assert {
        "incremental_adapter",
        "incremental_adapter_train_only",
        "incremental_adapter_reduction",
        "incremental_adapter_spatial",
        "incremental_adapter_alpha",
        "incremental_adapter_train_head",
        "incremental_adapter_features",
    } <= names
