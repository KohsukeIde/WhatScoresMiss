#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.concerto.representation_companion \
  --ledger results/ledgers/concerto_representation_companion_runs.csv \
  --output results/generated_tables/concerto_representation_companion_from_ledger.csv

bash scripts/_full_run_notice.sh representation-companion \
  CONCERTO_CODE_ROOT CONCERTO_CKPT SCANNET_ROOT OUTPUT_ROOT
