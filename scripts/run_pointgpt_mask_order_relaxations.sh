#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.pointgpt.aggregate_pointgpt_results \
  --pretext-ledger results/ledgers/pointgpt_pretext_logs.csv \
  --downstream-ledger results/ledgers/pointgpt_downstream_eval_runs.csv \
  --output results/generated_tables/pointgpt_mask_order_from_ledgers.csv

bash scripts/_full_run_notice.sh pointgpt-mask-order \
  POINTGPT_CODE_ROOT SCANOBJECTNN_ROOT SHAPENETPART_ROOT OUTPUT_ROOT
