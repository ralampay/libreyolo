"""Offline train/validate/resume smoke checks for both MobileNet variants."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import (
    LibreYOLO,
    LibreYOLO9DraxMobileNetV3Large,
    LibreYOLOXDraxMobileNetV3Large,
)

pytestmark = pytest.mark.smoke


def _tiny_detection_dataset(root: Path) -> Path:
    rng = np.random.default_rng(17)
    for split, count in (("train", 4), ("val", 2)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(count):
            image = rng.integers(40, 110, size=(64, 64, 3), dtype=np.uint8)
            image[16:48, 16:48] = 220
            Image.fromarray(image).save(image_dir / f"{index}.jpg")
            (label_dir / f"{index}.txt").write_text(
                f"{index % 2} 0.5 0.5 0.5 0.5\n",
                encoding="utf-8",
            )

    config = root / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": ["square_a", "square_b"],
            }
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.parametrize(
    "cls", [LibreYOLO9DraxMobileNetV3Large, LibreYOLOXDraxMobileNetV3Large]
)
def test_train_reload_resume(cls, tmp_path):
    torch.set_num_threads(2)
    data = _tiny_detection_dataset(tmp_path / "data")
    wrapper = cls(None, size="s", device="cpu")
    common = {
        "data": str(data),
        "batch": 2,
        "imgsz": 64,
        "lr0": 0.001,
        "optimizer": "SGD",
        "device": "cpu",
        "workers": 0,
        "project": str(tmp_path / "runs"),
        "amp": False,
        "patience": 0,
        "exist_ok": True,
        "warmup_epochs": 0,
    }
    result = wrapper.train(epochs=1, name="initial", pretrained=False, **common)
    assert math.isfinite(result["final_loss"])
    path = result["last_checkpoint"]
    loaded = LibreYOLO(path, device="cpu")
    assert type(loaded) is cls
    assert loaded.nb_classes == 2
    metrics = loaded.val(data=str(data), imgsz=64, batch=2, device="cpu", workers=0)
    assert math.isfinite(float(metrics["metrics/mAP50-95"]))
    resumed = loaded.train(epochs=2, name="resumed", resume=True, **common)
    assert math.isfinite(resumed["final_loss"])
    assert Path(resumed["last_checkpoint"]).exists()
