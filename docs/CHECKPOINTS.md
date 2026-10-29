# Checkpoints

No large model checkpoints are committed. Full diagnostics use official or
released checkpoints where applicable. Local variants, such as PointGPT relaxed
mask/order variants, should be produced by the provided scripts/configs or
loaded from user-provided paths.

Set checkpoint locations with environment variables:

| variable | used for |
|---|---|
| `CONCERTO_CKPT` | Concerto target-swap, coordinate closure, representation companion, scene support, and reported-score diagnostics. |
| `SONATA_CKPT` | scene support and reported-score comparator rows. |
| `UTONIA_CKPT` | scene support and reported-score comparator rows. |
| `PTV3_CKPT` | supervised scene control rows, when reported separately. |
| `POINTGPT_CKPT` | PointGPT mask/order, grouping, object support, and checkpoint reporting diagnostics. |
| `POINTMAE_CKPT` | Point-MAE object support and checkpoint reporting diagnostics. |
| `PCPMAE_CKPT` | PCP-MAE object support and checkpoint reporting diagnostics. |
| `CHECKPOINT_ROOT` | optional root directory containing multiple checkpoint subdirectories. |

External model implementations are also not bundled. When running Level 2
diagnostics from raw datasets/checkpoints, set the relevant code roots:

| variable | used for |
|---|---|
| `CONCERTO_CODE_ROOT` | Concerto-compatible target-swap, coordinate-closure, and representation companion runs. |
| `SCENE_CODE_ROOT` | scene support, Oracle@k, and class-pair diagnostics for scene backbones. |
| `POINTGPT_CODE_ROOT` | PointGPT relaxed pretraining, ScanObjectNN evaluation, ShapeNetPart evaluation, and grouping controls. |
| `POINTMAE_CODE_ROOT` | Point-MAE object support and checkpoint-reporting diagnostics. |
| `PCPMAE_CODE_ROOT` | PCP-MAE object support and checkpoint-reporting diagnostics. |

Official project and paper sources for these codebases are listed in
`docs/THIRD_PARTY_ASSETS.md`.

## PointGPT ShapeNetPart retention replay

The recorded PointGPT-S segmentation checkpoint for the corrected retention
results has SHA-256
`96079848c187d681f3a66aaf7860fa5a7b3a3882df66a4efddf5a9a5fd9c4002`.
It is not included here. Instructions for the replay and required external
assets are in `reproduce/pointgpt/original_r7c/README.md`.

## Local Variants

PointGPT relaxed mask/order variants are not redistributed as checkpoints.
The configs and excerpts under `reproduce/pointgpt/` document the changes;
they are not a complete end-to-end training implementation. A user-provided
checkpoint can be set with `POINTGPT_CKPT`.

Most patch files in this repository are sanitized excerpts and are not complete
vendored replacements for external repositories. The PointGPT grouping patch and
Concerto target-swap patch are apply-ready for the specified upstream commits;
see their respective setup instructions.

The wrapper scripts do not download checkpoints automatically. They check for
the required variables and print a clear message when a checkpoint is missing.
