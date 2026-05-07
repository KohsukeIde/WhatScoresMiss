#!/usr/bin/env bash
set -euo pipefail

: "${POINTGPT_CKPT:?Set POINTGPT_CKPT or provide relaxed variant checkpoints. See docs/CHECKPOINTS.md.}"
: "${SCANOBJECTNN_ROOT:?Set SCANOBJECTNN_ROOT. See docs/DATA.md.}"
: "${SHAPENETPART_ROOT:?Set SHAPENETPART_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

echo "PointGPT mask/order relaxations require the external PointGPT training/evaluation code."
echo "Use configs/pointgpt/mask_order_relaxations.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
