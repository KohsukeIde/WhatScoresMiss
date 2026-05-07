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

## Local Variants

PointGPT relaxed mask/order variants are not redistributed as checkpoints.
They can be recreated by running the corresponding external training pipeline
with `configs/pointgpt/mask_order_relaxations.yaml`, or loaded by setting
`POINTGPT_CKPT` to a user-provided checkpoint.

The wrapper scripts do not download checkpoints automatically. They check for
the required variables and print a clear message when a checkpoint is missing.
