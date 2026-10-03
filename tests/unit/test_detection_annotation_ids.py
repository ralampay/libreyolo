import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from libreyolo.data.yolo_coco_api import YOLOCocoAPI

pytestmark = pytest.mark.unit


def test_duplicate_detection_matching_matches_canonical_coco(tmp_path):
    image = tmp_path / "a.jpg"
    Image.new("RGB", (100, 100)).save(image)
    label = tmp_path / "a.txt"
    label.write_text("0 .05 .05 .1 .1\n0 .25 .25 .1 .1\n")
    gt = YOLOCocoAPI(
        None, None, ["object"], image_files=[str(image)], label_files=[str(label)]
    )
    rows = [
        dict(image_id=0, category_id=0, bbox=b, score=s)
        for b, s in [
            ([0, 0, 10, 10], 0.99),
            ([0, 0, 10, 10], 0.98),
            ([70, 70, 10, 10], 0.97),
            ([20, 20, 10, 10], 0.96),
        ]
    ]
    native = gt.loadRes(rows)
    assert list(native.anns) == [1, 2, 3, 4]
    canonical = COCO()
    canonical.dataset = {
        "info": {},
        "images": list(gt.imgs.values()),
        "categories": list(gt.cats.values()),
        "annotations": list(gt.anns.values()),
    }
    canonical.createIndex()

    def precision(g, d):
        e = COCOeval(g, d, "bbox")
        e.evaluate()
        e.accumulate()
        return e.eval["precision"]

    np.testing.assert_array_equal(
        precision(gt, native), precision(canonical, canonical.loadRes(rows))
    )
