"""CPU ONNX parity for the two MobileNet detector variants."""

import numpy as np
import pytest
import torch

from libreyolo import (
    LibreYOLO,
    LibreYOLO9DraxMobileNetV3Large,
    LibreYOLOXDraxMobileNetV3Large,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "cls", [LibreYOLO9DraxMobileNetV3Large, LibreYOLOXDraxMobileNetV3Large]
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_onnx_raw_and_preprocess_parity(cls, dynamic, tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = cls(None, size="s", nb_classes=2, device="cpu")
        model.model.eval()
        output = model.export(format="onnx", imgsz=64, dynamic=dynamic, simplify=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            output, options, providers=["CPUExecutionProvider"]
        )
        backend = LibreYOLO(output)
        assert backend.model_family == cls.FAMILY
        model.model.eval()
        model.model.head.export = True
        for batch in [1, 2] if dynamic else [1]:
            size = 64
            image = np.random.default_rng(12).integers(
                0, 256, (size, size, 3), dtype=np.uint8
            )
            tensor = model._preprocess(image, input_size=size)[0]
            backend_tensor = backend._preprocess(image, size, "auto")[0]
            np.testing.assert_allclose(tensor.numpy(), backend_tensor, rtol=0, atol=0)
            tensor = tensor.repeat(batch, 1, 1, 1)
            with torch.no_grad():
                expected = model.model(tensor)
            actual = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[
                0
            ]
            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
    finally:
        torch.set_num_threads(previous)
