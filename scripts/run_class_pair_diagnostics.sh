#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.reported_score.class_pair_diagnostics \
  --ledger results/ledgers/class_pair_diagnostic_runs.csv \
  --output results/generated_tables/class_pair_from_ledger.csv

bash scripts/_full_run_notice.sh class-pair \
  SCENE_CODE_ROOT SCANNET_ROOT OUTPUT_ROOT CONCERTO_CKPT SONATA_CKPT UTONIA_CKPT
