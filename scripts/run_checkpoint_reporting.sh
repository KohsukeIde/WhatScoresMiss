#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.reported_score.checkpoint_reporting \
  --ledger results/ledgers/scanobjectnn_checkpoint_selection_runs.csv \
  --output results/generated_tables/checkpoint_reporting_from_ledger.csv

bash scripts/_full_run_notice.sh checkpoint-reporting \
  SCANOBJECTNN_ROOT OUTPUT_ROOT POINTGPT_CODE_ROOT POINTMAE_CODE_ROOT PCPMAE_CODE_ROOT
