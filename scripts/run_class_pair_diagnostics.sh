#!/usr/bin/env bash
set -euo pipefail

: "${SCANNET_ROOT:?Set SCANNET_ROOT. See docs/DATA.md.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"

echo "Class-pair diagnostics require features/logits from external scene inference."
echo "Use configs/reported_score/class_pair_diagnostics.yaml."
echo "Expected output directory: ${OUTPUT_ROOT}"
