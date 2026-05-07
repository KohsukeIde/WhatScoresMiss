#!/usr/bin/env bash
set -euo pipefail

args=(
  --config reproduce/concerto/configs/scannet_target_swap.yaml
  --loss-ledger results/ledgers/concerto_loss_runs.csv
  --output results/generated_tables/concerto_target_swap_from_ledger.csv
)

if [[ -n "${CONCERTO_CODE_ROOT:-}" ]]; then
  args+=(--external-code-root "$CONCERTO_CODE_ROOT")
fi

python -m reproduce.concerto.target_swap "${args[@]}"

bash scripts/_full_run_notice.sh target-swap \
  CONCERTO_CODE_ROOT CONCERTO_CKPT OUTPUT_ROOT ARKITSCENES_ROOT SCANNET_ROOT SCANNETPP_ROOT S3DIS_ROOT HM3D_ROOT STRUCTURED3D_ROOT
