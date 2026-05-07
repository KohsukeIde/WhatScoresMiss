#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.support.aggregate_support_results \
  --scene-ledger results/ledgers/scannet_support_eval_runs.csv \
  --object-ledger results/ledgers/shapenetpart_support_eval_runs.csv \
  --output results/generated_tables/support_from_ledgers.csv

bash scripts/_full_run_notice.sh scene-support \
  SCENE_CODE_ROOT SCANNET_ROOT OUTPUT_ROOT CONCERTO_CKPT SONATA_CKPT UTONIA_CKPT PTV3_CKPT
