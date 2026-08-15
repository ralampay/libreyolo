"""Offline YOLOv9-S + Drax(B5) training and reload smoke test."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.models.yolo9 import DraxConfig
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.utils.serialization import load_trusted_torch_file

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


def test_yolo9_s_drax_b5_train_validate_save_reload(tmp_path):
    data = _tiny_detection_dataset(tmp_path / "data")
    vanilla_path = tmp_path / "vanilla-s.pt"
    LibreYOLO9(None, size="s", device="cpu").save(str(vanilla_path))

    drax_config = DraxConfig(enabled=True, stages=("b5",))
    model = LibreYOLO9(
        None,
        size="s",
        device="cpu",
        drax_config=drax_config,
    )
    results = model.train(
        data=str(data),
        epochs=1,
        batch=2,
        imgsz=64,
        lr0=0.001,
        optimizer="SGD",
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name="drax-smoke",
        amp=False,
        patience=0,
        pretrained=str(vanilla_path),
        exist_ok=True,
    )

    assert math.isfinite(results["final_loss"])
    checkpoint_path = Path(results["last_checkpoint"])
    checkpoint = load_trusted_torch_file(
        checkpoint_path,
        map_location="cpu",
        context="Drax smoke checkpoint",
    )
    assert checkpoint["drax"] == drax_config.to_dict()
    assert checkpoint["optimizer"]

    loaded = LibreYOLO9(str(checkpoint_path), size="s", device="cpu")
    assert loaded.drax_config == drax_config
    metrics = loaded.val(data=str(data), imgsz=64, batch=2, device="cpu", workers=0)
    assert math.isfinite(float(metrics["metrics/mAP50-95"]))

    loaded.model.eval()
    with torch.no_grad():
        output = loaded.model(torch.randn(1, 3, 64, 64))
    assert torch.isfinite(output["predictions"]).all()
