# YOLOX CSP-Drax feature fusion

`yolox_drax_csp_fusion` is the versioned successor to the experimental
`yolox_drax_csp` graph. The legacy family remains loadable. Its family-specific
gradient clipping was removed so it now inherits the same optimization
defaults as YOLOX.

The fusion model keeps the YOLOX-M CSPDarknet and PAN widths. A 160-channel
efficient Drax block supplies residual P5 context with a learnable scale
initialized to `1e-3`. Each of the two top-down and two bottom-up PAN joins has
a channel-wise, two-branch softmax gate. Gate logits start at zero and the
weights are multiplied by two, so both inputs are exact pass-throughs at
initialization. A depthwise-separable YOLOX head keeps the total parameter
count below YOLOX-M.

## Shared-transfer comparison

`InitializeYOLOXSharedTransfer` accepts an official or converted YOLOX-M state
dict (or checkpoint path), a YOLOX-M control, and a fusion candidate. Its
`execute()` method strictly copies every YOLOX backbone/PAN tensor into both
models, excludes both heads, resets both heads from a recorded seed, and
returns a structured report. Candidate-only Drax and fusion parameters retain
their architecture-defined initialization. Missing or shape-incompatible body
tensors fail before training.

## Experiment contract

For paired comparisons, use the same split, augmentation sequence, batch size,
optimizer, scheduler, epochs, pretrained body, head-reset seed, and run seed.
Leave `nbs` unset when the physical batch is the intended effective batch and
leave gradient clipping disabled for both models. Record the source checkpoint
URL and SHA-256 digest.

Report paired per-seed AP differences with a confidence interval and an exact
paired sign-flip test. A non-significant superiority test is not evidence of
equivalence, so also run a two one-sided equivalence test with a predeclared
smallest effect size of interest.

## Provenance

The implementation is original LibreYOLO code. Architectural vocabulary is
informed by CSPNet (arXiv:1911.11929), YOLOX (arXiv:2107.08430), Selective
Kernel Networks (arXiv:1903.06586), EfficientDet (arXiv:1911.09070), ASFF
(arXiv:1911.09516), and the existing LibreYOLO Drax block. No third-party code
was copied or adapted.
