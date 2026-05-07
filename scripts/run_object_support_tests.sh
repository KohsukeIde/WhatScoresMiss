#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.support.aggregate_support_results \
  --scene-ledger results/ledgers/scannet_support_eval_runs.csv \
  --object-ledger results/ledgers/shapenetpart_support_eval_runs.csv \
  --output results/generated_tables/support_from_ledgers.csv

bash scripts/_full_run_notice.sh object-support \
  POINTGPT_CODE_ROOT POINTMAE_CODE_ROOT PCPMAE_CODE_ROOT SCANOBJECTNN_ROOT SHAPENETPART_ROOT OUTPUT_ROOT POINTGPT_CKPT POINTMAE_CKPT PCPMAE_CKPT
