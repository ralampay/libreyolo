# YOLOX taxonomy transfer

Load the original checkpoint strictly before calling
`libreyolo.models.yolox.transfer.reset_classifiers(model, num_classes, seed=seed)`.
This explicitly replaces all class prediction convolutions, even when class
counts match. Objectness, box regression and feature weights are preserved.
Initialization uses an isolated CPU random generator and a 0.01 class prior.
Set target names on the wrapper separately. Freeze and inject adapters afterwards.

`transfer_state_dict` exports trainable tensors and the entire detection head,
including BatchNorm running statistics. `load_transfer_state_dict` requires the
same classifier reset, injection and trainability configuration, validates keys
and shapes before copying, and does not change frozen foundation tensors.
This compact tensor mapping is not a standalone standard model checkpoint.
The orchestrator must record foundation hash, target classes, adapter settings
and initialization seed. Legacy adapter checkpoints retain their existing format.

For dense annotations, `YOLOXConfig.max_labels` controls the training transform's
padded label capacity (default 50, positive integer). Set it to at least the
maximum object count when geometric augmentation does not combine images.
Mosaic/mixup can require a larger capacity. This does not disable the transform's
existing resized-box filter. `eval_max_det` controls validation detection limits
separately; it does not change COCO metric definitions.
