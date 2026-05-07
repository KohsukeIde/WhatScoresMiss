#!/usr/bin/env bash
set -euo pipefail

python -m reproduce.concerto.coordinate_only_closure \
  --config configs/concerto/coordinate_closure.yaml \
  --closure-ledger results/ledgers/concerto_closure_fit_runs.csv \
  --output results/generated_tables/concerto_coordinate_closure_from_ledger.csv

bash scripts/_full_run_notice.sh coordinate-closure \
  CONCERTO_CODE_ROOT CONCERTO_CKPT OUTPUT_ROOT ARKITSCENES_ROOT SCANNET_ROOT SCANNETPP_ROOT S3DIS_ROOT HM3D_ROOT STRUCTURED3D_ROOT
