# Drax MobileNetV3 Large detection backbones

These detect-only variants replace the native YOLO9 or YOLOX backbone with
MobileNetV3 Large and a final-stage Drax adapter. Each retains its parent
neck, head, loss, and augmentation recipe.

```python
from libreyolo import LibreYOLO9DraxMobileNetV3Large, LibreYOLOXDraxMobileNetV3Large

model = LibreYOLO9DraxMobileNetV3Large(size="s", device="cpu")
# Or: model = LibreYOLOXDraxMobileNetV3Large(size="s", device="cpu")
model.train(data="path/to/data.yaml", pretrained=True, epochs=100)
model.save("detector.pt")
```

```bash
libreyolo train model=yolo9-drax-mobilenet-v3-large data=path/to/data.yaml pretrained=true
libreyolo train --model yolox-drax-mobilenet-v3-large --data path/to/data.yaml --no-pretrained
```

The aliases default to `s`. Append a size, such as
`yolox-drax-mobilenet-v3-large-l`, to change the neck/head size.
YOLO9 supports `t/s/m/c`; YOLOX supports `n/t/s/m/l/x`.

Construction never downloads weights. `train(pretrained=True)` loads only
MobileNet's ImageNet V2 feature weights through torchvision, leaving Drax,
projections, neck, and head newly initialized. `pretrained=False` follows
the library's seeded scratch-training contract. Load a trained detector
with `LibreYOLO("detector.pt")`; training or resuming that checkpoint never
replaces its backbone with ImageNet weights. Resume requires a training
checkpoint such as `last.pt`, not the inference-only output of `save()`.

## Architecture

Stride-8/16/32 features have 40/112/960 channels. The stride-32 feature is
refined by a 960-to-160 projection, BatchNorm, Hardswish, one Drax block,
and a 160-to-960 projection with BatchNorm and an outer residual connection.
The Drax block uses efficient attention, average fusion, and no drop-path.
Three 1x1 convolution/BatchNorm/SiLU projections match the detector neck.

YOLO9 accepts RGB 0-1 tensors; YOLOX accepts BGR 0-255 tensors. Conversion
to ImageNet-normalized RGB happens inside the backbone in training,
inference, and export. MobileNet's BatchNorm settings are preserved.

With 80 classes, the size-s variants have 8,226,304 parameters (YOLO9)
and 8,829,839 parameters (YOLOX). Parameter count does not imply latency
or accuracy; neither has been benchmarked on a full dataset.

The source adapter is adapted from MIT-declared `ralampay/mlx`, commit
`986408db404f6ccc97a2c5acf6081d3731053bd9`. Attribution is included in
module NOTICE files. MLX classification checkpoints are not imported.

## Validation

Local tests cover all supported sizes, gradients through the backbone and
Drax, checkpoint round trips, class-count changes, CLI construction, and
train/validate/resume on a synthetic fixture. Feature parity against the
MLX classifier's pre-pooling path was checked with identical random weights.
Full-dataset accuracy, RF1 convergence, and GPU performance are not established.

ImageNet V2 initialization was also checked against the real torchvision
checkpoint: every transferred feature tensor matched exactly. ONNX CPU FP32
raw tensors and preprocessing pass parity checks at size `s`, 64x64, for
static and dynamic batch exports. Other export runtimes were not tested.

A 100-update scratch run on four synthetic 64x64 images reduced training
loss but did not pass the overfit gate (validation mAP remained zero).
This is a training-evidence limit, not an accuracy-qualified checkpoint.

Implementation packages live in `libreyolo/models/yolo9-drax-mobilenet-v3/`
and `libreyolo/models/yolox-drax-mobilenet-v3/`. Their shared backbone lives
in `libreyolo/models/drax_mobilenet_v3/`. Use the top-level Python classes
shown above; the registry loads the hyphenated packages with `importlib`.
