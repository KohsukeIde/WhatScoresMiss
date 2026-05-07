#!/usr/bin/env bash
set -euo pipefail

: "${SCANOBJECTNN_ROOT:?Set SCANOBJECTNN_ROOT. See docs/DATA.md.}"
: "${SHAPENETPART_ROOT:?Set SHAPENETPART_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

for var in POINTGPT_CKPT POINTMAE_CKPT PCPMAE_CKPT; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing ${var}. See docs/CHECKPOINTS.md." >&2
    exit 2
  fi
done

echo "Object support tests require external PointGPT/Point-MAE/PCP-MAE model code."
echo "Use configs/object_support/scanobjectnn_support.yaml and configs/object_support/shapenetpart_support.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
