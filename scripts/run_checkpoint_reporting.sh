#!/usr/bin/env bash
set -euo pipefail

: "${SCANOBJECTNN_ROOT:?Set SCANOBJECTNN_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

for var in POINTGPT_CKPT POINTMAE_CKPT PCPMAE_CKPT; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing ${var}. See docs/CHECKPOINTS.md." >&2
    exit 2
  fi
done

echo "Checkpoint reporting requires external fine-tuning logs/checkpoints."
echo "Use configs/reported_score/checkpoint_reporting.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
