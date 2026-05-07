#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.reported_score.oracle_from_logits \
  --ledger results/ledgers/scannet_oracle_eval_runs.csv \
  --output results/generated_tables/oracle_from_ledger.csv

bash scripts/_full_run_notice.sh oracle \
  SCENE_CODE_ROOT SCANNET_ROOT OUTPUT_ROOT CONCERTO_CKPT SONATA_CKPT UTONIA_CKPT PTV3_CKPT
