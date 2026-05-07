#!/usr/bin/env bash
set -euo pipefail

python reproduce/pointgpt/aggregate_pointgpt_results.py
echo "For full ScanObjectNN eval, set POINTGPT_CODE_ROOT, SCANOBJECTNN_ROOT, and POINTGPT_CKPT."
