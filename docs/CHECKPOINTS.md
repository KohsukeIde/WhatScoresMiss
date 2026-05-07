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

## Local Variants

PointGPT relaxed mask/order variants are not redistributed as checkpoints.
They can be recreated by running the corresponding external training pipeline
with `reproduce/pointgpt/configs/*.yaml` and the patches under
`reproduce/pointgpt/patches/`, or loaded by setting `POINTGPT_CKPT` to a
user-provided checkpoint.

Patch files in this repository are sanitized excerpts. They identify the
relevant changes used by the diagnostics, but they are not complete vendored
replacements for the external repositories. Full execution requires applying or
adapting the corresponding changes inside the listed official codebases.

The wrapper scripts do not download checkpoints automatically. They check for
the required variables and print a clear message when a checkpoint is missing.
