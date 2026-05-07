#!/usr/bin/env bash
set -euo pipefail

: "${SCANNET_ROOT:?Set SCANNET_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

echo "Oracle@k diagnostics can run from cached logits or external scene inference code."
echo "Use configs/reported_score/oracle_scannet.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
