# Drax spatial-only ablation

`drax-spatial` uses the same dense, stride-one 1x1 convolution injection locations
as `drax-hybrid`. It retains the frozen convolution, compressed local/dilated
depthwise paths, learned two-way fusion, channel mixing and zero-initialized
output projection. It contains no LoRA module or low-rank trainable tensors.

Construction uses a temporary rank-eight hybrid to reproduce spatial parameter
initialization and RNG advancement. The temporary low-rank layers are discarded;
rank does not control the spatial adapter's size. Reduction controls its width.
Existing hybrid state keys and behavior are unchanged. This is an ablation of
the convolution-wrapper hybrid, not the earlier feature-level `drax` adapter.

For six-class YOLOX-L with all 26 neck convolutions, reduction eight adds 2,056,724
trainable spatial parameters. Rank-100 LoRA on the same locations adds 2,227,200;
alpha 12.5 preserves the rank-eight, alpha-one multiplier of 1/8.
