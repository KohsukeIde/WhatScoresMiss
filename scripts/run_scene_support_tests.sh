#!/usr/bin/env bash
set -euo pipefail

: "${SCANNET_ROOT:?Set SCANNET_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

for var in CONCERTO_CKPT SONATA_CKPT UTONIA_CKPT; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing ${var}. See docs/CHECKPOINTS.md." >&2
    exit 2
  fi
done

echo "Scene support tests require external scene model inference code."
echo "Use configs/scene_support/scannet_support.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
