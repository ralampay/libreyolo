from types import SimpleNamespace

import numpy as np
import pytest

from libreyolo.models.yolox.trainer import YOLOXTrainer
from libreyolo.training.config import YOLOXConfig

pytestmark = pytest.mark.unit


def test_yolox_dense_label_capacity():
    config = YOLOXConfig(max_labels=596, flip_prob=0, hsv_prob=0)
    train, _ = YOLOXTrainer.create_transforms(SimpleNamespace(config=config))
    targets = np.tile([10, 10, 30, 30, 0], (468, 1)).astype(np.float32)
    _, labels = train(np.zeros((320, 320, 3), dtype=np.uint8), targets, (320, 320))
    assert labels.shape == (596, 5)
    assert (labels[:, 3] > 0).sum() == 468
    assert YOLOXConfig().max_labels == 50


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_yolox_label_capacity_rejects_invalid_values(capacity):
    with pytest.raises(ValueError, match="max_labels"):
        YOLOXConfig(max_labels=capacity)
