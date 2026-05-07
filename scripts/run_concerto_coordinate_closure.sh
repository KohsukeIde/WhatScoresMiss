#!/usr/bin/env bash
set -euo pipefail

: "${CONCERTO_CKPT:?Set CONCERTO_CKPT. See docs/CHECKPOINTS.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

for var in ARKITSCENES_ROOT SCANNET_ROOT SCANNETPP_ROOT S3DIS_ROOT HM3D_ROOT STRUCTURED3D_ROOT; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing ${var}. See docs/DATA.md." >&2
    exit 2
  fi
done

echo "Coordinate-closure diagnostics require external target-cache extraction and evaluation code."
echo "Use configs/concerto/coordinate_closure.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
