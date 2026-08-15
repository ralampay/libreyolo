# YOLOv9 Drax experiments

Drax is an optional residual dual-branch feature refinement module for the
standard YOLOv9 family. Vanilla YOLOv9 remains the default. The first supported
configuration enables Drax only after the B5 ELAN stage, where a 640-pixel
input has a 20-by-20 feature map.

The implementation is original LibreYOLO code informed by *A ConvNet for the
2020s* (arXiv:2201.03545) and *Selective Kernel Networks* (arXiv:1903.06586).
No third-party source code was copied or adapted for the Drax module.

## Controlled first experiment

Keep the dataset, seed, image size, optimizer, batch size, augmentation, and
epoch count identical between the two runs.

```python
from libreyolo import LibreYOLO9


baseline = LibreYOLO9(model_path=None, size="s")
baseline.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=0,
    pretrained=True,
    project="runs/drax",
    name="yolo9s-baseline",
)
```

```python
from libreyolo import LibreYOLO9
from libreyolo.models.yolo9 import DraxConfig


drax = LibreYOLO9(
    model_path=None,
    size="s",
    drax_config=DraxConfig(
        enabled=True,
        stages=("b5",),
        use_attention=True,
        efficient=True,
        fusion_mode="average",
        drop_path=0.0,
    ),
)
drax.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=0,
    pretrained=True,
    project="runs/drax",
    name="yolo9s-drax-b5",
)
```

`pretrained=True` uses the existing YOLOv9-S transfer-loading path. Matching
YOLOv9 tensors load by key and shape, class-dependent incompatible head
tensors follow the existing transfer behavior, and Drax tensors remain newly
initialized. The transfer summary reports each category and the Drax-specific
tensor and parameter counts.

Compare at least mAP50, mAP50-95, precision, recall, total and Drax parameter
counts, peak GPU memory, inference throughput, and training time per epoch.
Use the metrics and timing artifacts already written by the training loop;
GFLOPs should only be reported when measured by an existing supported profiler.

| Measurement | Experiment A: YOLOv9-S | Experiment B: YOLOv9-S + Drax(B5) |
| --- | ---: | ---: |
| Parameters | 7,226,192 | 7,897,168 |
| Drax parameters | 0 | 670,976 |
| mAP50 | record | record |
| mAP50-95 | record | record |
| Precision | record | record |
| Recall | record | record |
| Peak GPU memory | record | record |
| Inference speed / FPS | record | record |
| Training time per epoch | record | record |
| GFLOPs, when supported | record | record |

## Reload for inference or resume

Training checkpoints carry a primitive, versioned `drax` architecture
dictionary. Loading reconstructs the architecture before applying the state
dict, so inference needs no Drax-specific path:

```python
from libreyolo import LibreYOLO9


model = LibreYOLO9(
    "runs/drax/yolo9s-drax-b5/weights/best.pt",
    size="s",
)
results = model.predict("path/to/image.jpg")
```

Resume uses the same loading path and additionally requires the checkpoint and
current Drax configurations to match exactly:

```python
model = LibreYOLO9(
    "runs/drax/yolo9s-drax-b5/weights/last.pt",
    size="s",
)
model.train(
    data="path/to/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    lr0=0.01,
    optimizer="SGD",
    resume=True,
)
```

Re-specify the same training settings used by the original run; resume restores
model, optimizer, EMA, scaler, epoch, and RNG state while the current training
configuration defines the continued schedule.

`model.info()` reports total, trainable, and Drax parameter counts along with
the enabled stages and fusion settings.

## Current limits

B3 and B4 full spatial attention are substantially more expensive than B5 at
640 pixels. Spatial-reduction and windowed attention are future work. Export
parity, exporter support claims, and FLOPs accounting are also not established
by this first training integration.
