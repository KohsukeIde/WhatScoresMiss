#!/usr/bin/env bash
set -euo pipefail

: "${CONCERTO_CKPT:?Set CONCERTO_CKPT. See docs/CHECKPOINTS.md.}"
: "${SCANNET_ROOT:?Set SCANNET_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

echo "Representation companion runs require the external scene backbone and ScanNet linear-proxy code."
echo "Use configs/concerto/representation_companion.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
