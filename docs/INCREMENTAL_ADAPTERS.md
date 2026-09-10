# IncrementalAdapters for YOLOX-Drax-MobileNetV3

This experimental feature is scoped only to
`yolox-drax-mobilenet-v3-large`. It is not available on other LibreYOLO model
families.

## Motivation

Assume a YOLOX-Drax-MobileNetV3 foundation detector has already been trained
on a large dataset, D0. Later, a small dataset, D1, becomes available with
exactly the same class definitions and class indices. Replaying all historical
data and retraining the complete detector may be expensive.

IncrementalAdapters test whether compact trainable modules can adapt the
foundation features using D1 while the foundation detector remains immutable.
This is parameter-efficient fine-tuning (PEFT) for continual object detection.
When D1 represents a demonstrated distribution or environment change, such as
a new camera, viewpoint, city, weather condition, or illumination, the setting
is more specifically domain-incremental object detection (DIOD). If D1 is only
more data from approximately the same distribution, describe it conservatively
as incremental or continual object detection with new instances.

The experimental hypothesis is that an adapter trained on D1 may approach a
full-retraining reference while updating substantially fewer parameters and
without complete D0 replay. It is not assumed to outperform full retraining.

## Literature basis

[YOLO-Adapter](https://doi.org/10.1016/j.neucom.2026.133138) introduces
convolutional bottleneck adapters for parameter-efficient few-shot adaptation
of frozen YOLO detectors and motivates adapters as a way to preserve pretrained
generalization when data is scarce. This implementation is inspired by the
paper's convolutional adapter principle. It does not reproduce the paper's
specific parallel placement or variance-consistent initialization.

[Learning Domain Bias (LDB)](https://doi.org/10.1609/aaai.v38i13.29427)
formulates non-exemplar DIOD by first training a base detector, freezing it,
and learning compact domain-specific parameters. LDB motivates this project's
freeze-the-foundation training strategy. IncrementalAdapters do not implement
LDB's domain-bias modules or domain selector.

[Orthogonal Knowledge Refreshing (OKR)](https://arxiv.org/abs/2607.17340)
treats PEFT as a promising DIOD approach and adds independent domain-specific
low-rank branches plus orthogonal gradient projection. OKR motivates the
parameter-efficient continual-learning direction and future extension points.
This first version implements neither low-rank branches nor orthogonal
training.

### What is borrowed and what is project-specific

Borrowed from the literature:

- parameter-efficient adaptation of a frozen foundation model
- residual feature correction
- a convolutional bottleneck concept
- continual and domain-incremental adaptation motivation

Project-specific experimental design:

- application to YOLOX-Drax-MobileNetV3 Large
- placement at its projected P3/P4/P5 boundary
- a hard enable/disable mechanism
- strict compatibility with existing LibreYOLO checkpoints
- comparison among Models A, B, and C

No novelty is claimed for the established adapter concepts.

## Architecture

The MobileNetV3 stages produce stride-8, stride-16, and stride-32 features
with 40, 112, and 960 channels. The existing Drax components
`adapter_down`, `adapter_norm`, `drax_refiner`, and `adapter_up` refine only
the raw stride-32 feature. They are foundation architecture and are unrelated
to IncrementalAdapters.

After the existing feature projections, YOLOX receives `dark3`, `dark4`, and
`dark5`. Their channels are `256 * width`, `512 * width`, and `1024 * width`.
IncrementalAdapters operate on those projected tensors immediately before the
YOLOX PAN/FPN:

```text
Drax-MobileNetV3
       |
  projections
       |
   +---+---+
   |   |   |
  P3  P4  P5       (dark3, dark4, dark5)
   |   |   |
  IA  IA  IA
   |   |   |
   +---+---+
       |
   YOLOX PAN
       |
      Head
```

For `x` with `C` channels, the default adapter is:

```text
z     = Conv1x1(C, h)(x)
z     = SiLU(z)
z     = DepthwiseConv3x3(h)(z)
z     = SiLU(z)
delta = Conv1x1(h, C)(z)
IA(x) = x + alpha * delta

h = max(C // reduction, 8)
reduction = 16
alpha = 1.0
```

The final 1x1 convolution's weight and bias are initialized to zero, so a
fresh enabled adapter is exactly an identity at construction. There is no
BatchNorm inside the adapter. Set `incremental_adapter_spatial=false` for the
1x1-only ablation, or select a subset with
`incremental_adapter_features=p3,p5`.

For size `s` with 80 classes, the foundation has 8,829,839 parameters. The
default three adapters add 44,520 parameters: 2,264 at P3, 8,624 at P4, and
33,632 at P5. Adapter-only training therefore updates 44,520 of 8,874,359
parameters, approximately 0.502%. Counts vary with model size, class count,
reduction, spatial stage, and selected scales. The training result includes a
`training_time_seconds` value and, when adapters are enabled, a
`parameter_counts` dictionary. `model.incremental_adapter_parameter_report()`
returns the parameter accounting for an attached model.

## Training modes

Normal foundation training leaves the feature absent and preserves historical
behavior:

```bash
libreyolo train model=yolox-drax-mobilenet-v3-large data=D0.yaml pretrained=false
```

Model A loads the foundation checkpoint and performs ordinary full-model
fine-tuning on D1. Adapter options are omitted, so adapters remain disabled:

```bash
libreyolo train model=foundation.pt data=D1.yaml name=model-a
```

Model B loads the same checkpoint, attaches fresh zero-initialized adapters,
freezes every foundation parameter and frozen BatchNorm running statistic, and
optimizes only `incremental_adapters.*`:

```bash
libreyolo train model=foundation.pt data=D1.yaml name=model-b \
  incremental_adapter=true incremental_adapter_train_only=true
```

The optional head ablation adds
`incremental_adapter_train_head=true`. The head remains frozen by default.
Reduction ratios and scales can be selected, for example:

```bash
libreyolo train model=foundation.pt data=D1.yaml \
  incremental_adapter=true incremental_adapter_train_only=true \
  incremental_adapter_reduction=8 incremental_adapter_features=p3,p5 \
  incremental_adapter_spatial=false
```

Model C is the full-retraining reference on a dataset YAML representing D0+D1:

```bash
libreyolo train model=yolox-drax-mobilenet-v3-large \
  data=D0-plus-D1.yaml pretrained=false name=model-c
```

The same controls are Python `train()` keyword arguments. To attach adapters
outside training, call `attach_incremental_adapters()`, then
`enable_incremental_adapters()` or `disable_incremental_adapters()`.

## Checkpoint compatibility

Adapters are not constructed by default. Consequently, an old
`foundation.pt` has the same expected module graph and loads with the family's
existing strict state-dict rules. Strictness is not disabled globally or for
this family.

The lifecycle for Model B is:

```text
load foundation.pt strictly
        |
attach zero-initialized IncrementalAdapters
        |
freeze every foundation parameter and BatchNorm statistic
        |
construct optimizer from trainable parameters only
```

An adapter-enabled checkpoint stores `incremental_adapters.*` tensors and a
small optional `incremental_adapters` metadata dictionary. On load, only the
presence of that namespace permits LibreYOLO to reconstruct the adapter graph;
all missing or unexpected foundation tensors still fail strict loading. Raw
adapter state dictionaries can reconstruct their feature selection, hidden
channels, spatial stage, and alpha from tensor keys and shapes.

These invariants are unit-tested:

```text
old foundation checkpoint + adapters OFF = original detector output

old foundation checkpoint + fresh adapters ON + foundation frozen
    = valid adapter-only optimizer and training graph
```

## Experimental protocol

Use identical D0/D1 class definitions and indices. Evaluate the foundation,
Models A/B/C, and useful ablations on T0 and T1, plus Tcombined when available.
Record mAP50, mAP50-95, precision, recall, wall-clock training time, total and
trainable parameter counts, and peak GPU memory when the training environment
already reports it.

The primary comparison table is:

| Model | T0 mAP50-95 | T1 mAP50-95 | Params trained | Training time |
|---|---:|---:|---:|---:|
| Foundation F | | | | |
| A: normal fine-tuning on D1 | | | | |
| B: adapter-only training on D1 | | | | |
| C: full retraining on D0+D1 | | | | |

Evaluate each checkpoint separately against both test YAMLs, for example:

```bash
libreyolo val model=runs/train/model-b/weights/best.pt data=T0.yaml
libreyolo val model=runs/train/model-b/weights/best.pt data=T1.yaml
```

## Limitations

- Adapter training can still overfit a small D1.
- Adapters do not guarantee prevention of catastrophic forgetting.
- T1 performance can improve while T0 foundation performance decreases.
- P3/P4/P5 placement, reduction ratio, spatial convolution, and alpha require
  experimental validation.
- The same-label-space assumption is mandatory; the adapter-only mode rejects
  a class-count change and does not rebuild the detection head.
- Model C remains the reference for full retraining. Model B is not assumed to
  outperform it.
- No full-dataset accuracy, convergence, training-time, or peak-memory claim is
  established by the unit tests.

## Future extensions

- Multiple independent domain adapters, `F+A1`, `F+A2`, and `F+A3`, following
  the broad domain-specific adaptation motivation of LDB.
- Cumulative adapter training with explicit measurement of forgetting inside
  the adapter.
- A small replay memory mixed with new samples.
- Distillation from the frozen foundation or previous adapted model.
- An optional OKR-inspired orthogonal constraint that limits interference with
  previously learned adaptation subspaces.
- Low-rank adapter branches compared with the convolutional bottleneck.

None of these extensions is implemented in this version.

## References

```bibtex
@article{chiniforoushan2026yoloadapter,
  author = {Mohammadamin Chiniforoushan and Mohammad Reza Mohammadi},
  title = {{YOLO-Adapter}: Beyond Full Fine-Tuning for Accurate Few-Shot Object Detection},
  journal = {Neurocomputing},
  volume = {678},
  pages = {133138},
  year = {2026},
  doi = {10.1016/j.neucom.2026.133138}
}

@article{song2024learningdomainbias,
  author = {Xiang Song and Yuhang He and Songlin Dong and Yihong Gong},
  title = {Non-exemplar Domain Incremental Object Detection via Learning Domain Bias},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {38},
  number = {13},
  pages = {15056--15065},
  year = {2024},
  doi = {10.1609/aaai.v38i13.29427}
}

@article{zhang2026okr,
  author = {Aoting Zhang and Dongbao Yang and Chang Liu and Xiaopeng Hong and Can Ma and Yu Zhou},
  title = {Orthogonal Knowledge Refreshing for Domain-Incremental Object Detection},
  journal = {arXiv preprint arXiv:2607.17340},
  year = {2026},
  eprint = {2607.17340},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
