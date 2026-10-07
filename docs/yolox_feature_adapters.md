# YOLOX feature adapters

`libreyolo.adapters` provides checkpoint-independent PyTorch modules. Construct
standard `LibreYOLOXModel`, load its ordinary checkpoint strictly, freeze its
parameters, then call `inject_adapters`. The generic adapter classes accept
`[B,C,H,W]` tensors and have no dependency on YOLOX. `yolox_targets` owns the
YOLOX-specific placement policy. Injecting before loading a foundation
checkpoint is intentionally unsupported because wrappers change parameter keys.

The default research placement is the three neck outputs `backbone.C3_p3`,
`backbone.C3_n3`, and `backbone.C3_n4` (strides 8, 16 and 32; channels 256,
512 and 1024 for YOLOX-L). This exposes all detector scales while keeping
the dark backbone and detection head frozen. Backbone placement wraps dark3,
dark4 and dark5. `backbone+neck` wraps all six. LoRA instead targets dense 1x1
convolutions within the selected region; it updates the convolution kernel by
`(alpha/rank) * B @ A` with pretrained `W` frozen. The runtime path uses two
1x1 convolutions, which is algebraically the same low-rank kernel update.

| Name | Classification | Feature operation |
| --- | --- | --- |
| Bottleneck | adaptation of adapter bottleneck to CNN maps | 1x1 down, SiLU, 1x1 up, residual |
| SSF | direct feature-map implementation | per-channel scale and shift |
| LoRA | adaptation of low-rank weight updates to 1x1 CNN kernels | frozen convolution plus rank-r update |
| Convpass | adaptation from transformer bypass to CNN maps | 1x1 down, 3x3 spatial, 1x1 up, residual |
| Conv-Adapter | adaptation of published ConvNet concept | 1x1 down, depthwise 3x3, 1x1 up, residual |
| DraxAdapter | new proposed method | 1x1 down, local and dilated depthwise branches, softmax fusion, 1x1 up, residual |

All feature adapters start at exact identity. The output projection is zero
initialized except for SSF (`gamma=1`, `beta=0`). The LoRA `B` factor is zero
initialized. Frozen batch-norm running statistics must remain fixed during
training. MLX's experimental trainer enforces this.

LibreYOLO's `DraxBlock` combines a local ConvNeXt branch with a wider-context
attention branch, then fuses residual deltas. DraxAdapter retains the local
versus wider-context parallel paths and learned fusion, but uses compressed
depthwise 3x3 and dilated 3x3 convolutions instead of ConvNeXt and self
attention. This is a new architecture inspired by Drax, not a reproduction.
At width `C` and hidden width `r=floor(C/reduction)`, its trainable parameter
count is `2Cr + 21r + C + 2` with biases enabled. The two spatial paths
have nominal 3x3 and 5x5 receptive fields. Bottleneck and Conv-Adapter use
the same reduction for approximate parameter matching; Drax adds only one
extra depthwise path and two fusion logits at each target.

`inject_adapters` returns exact frozen/trainable/total counts and module names.
`adapter_state_dict` emits trainable parameter tensors only. Reconstruction
requires the original foundation checkpoint plus the recorded method, target,
reduction/rank, and `train_head` settings.

## Literature and provenance

These are independently written implementations of published ideas, not
ported third-party code. No external implementation code was copied.

- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751), 2019: residual bottleneck adapter pattern.
- Lian et al., [Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823), NeurIPS 2022: SSF.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022: low-rank weight update.
- Jie and Deng, [Convolutional Bypasses Are Better Vision Transformer Adapters](https://arxiv.org/abs/2207.07039), 2022: Convpass. Their original architecture targets vision transformers; this implementation operates directly on CNN maps.
- Chen et al., [Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets](https://arxiv.org/abs/2208.07463), CVPRW 2024: ConvNet adaptation. This implementation applies the spatial compressed branch at YOLOX feature boundaries, not inside every original convolution block.
- LibreYOLO `libreyolo/models/yolo9/drax.py`, `DraxBlock`, in-repository reference: proposed DraxAdapter inspiration.

For the research experiment, split preparation, training, evaluation and
reports are owned by MLX. See its object-detection adapter guide.
# Drax hybrid

`drax-hybrid` combines a genuine rank-configurable LoRA weight update with a
nonlinear compressed spatial bypass. The bypass uses local and dilation-2
depthwise convolutions, a learned two-way softmax, and a compressed 1x1 mixer.
This is an original Convpass-inspired depthwise-separable adaptation, not an
exact reproduction of Convpass. It retains the original Drax adapter's
local/context fusion idea. No third-party source code is copied.

The generic `DraxHybridConv2d` accepts a frozen dense, unpadded, stride-one 1x1
convolution. Both branch output projections start at zero. Its linear path is
exactly `W + (alpha/rank) BA`; the spatial path is nonlinear and cannot be
merged into that weight update. `inject_adapters` accepts the registered name
with existing rank, reduction, and alpha controls. The YOLOX policy shares the
exact target selector with LoRA for neck, backbone, and combined placement.
Neck placement wraps 26 dense 1x1 convolutions before their existing normalization
and activation. Standard foundation checkpoints load before injection.

At rank 8 and reduction 8, YOLOX-L neck injection trains 2,234,900 parameters.
Earlier three-projection checkpoints require their recorded injection paths;
they are not compatible with the new default placement.
Memory efficiency and detection accuracy must be measured for each workload.

# Compatibility status

This package remains available for existing callers and checkpoints. Active
experimental adapter implementations now live in MLX. New research adapters
must be added there; LibreYOLO retains this standalone compatibility surface
without an MLX dependency. Detector models and ordinary training are unchanged.
