#!/usr/bin/env bash
set -euo pipefail

python reproduce/pointgpt/aggregate_pointgpt_results.py
echo "For full ShapeNetPart eval, set POINTGPT_CODE_ROOT, SHAPENETPART_ROOT, and POINTGPT_CKPT."
