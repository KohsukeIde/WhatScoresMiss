#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-reproduce/pointgpt/configs/pointgpt_s_mask_on_random_order.yaml}"
python reproduce/pointgpt/aggregate_pointgpt_results.py >/dev/null

if [[ -z "${POINTGPT_CODE_ROOT:-}" ]]; then
  echo "POINTGPT_CODE_ROOT is not set; external PointGPT training is not launched." >&2
  echo "Diagnostic aggregation from sanitized ledgers succeeded." >&2
  exit 0
fi

echo "External command template:"
echo "cd ${POINTGPT_CODE_ROOT}"
echo "python main.py --config ${CONFIG} --exp_name <variant> --seed <seed>"
