#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/generated_figures

python - <<'PY'
from bds.plotting import (
    make_concerto_pretext_figure,
    make_oracle_scannet_figure,
    make_support_scannet_figure,
    make_support_scannet_curves_figure,
    make_support_shapenetpart_figure,
    make_support_shapenetpart_curves_figure,
)

make_concerto_pretext_figure(
    "results/paper_reported_results/concerto_pretext_diagnostics.csv",
    "results/generated_figures/concerto_pretext_diagnostics.png",
)
make_support_scannet_figure(
    "results/paper_reported_results/support_scannet.csv",
    "results/generated_figures/support_scannet.png",
)
make_support_shapenetpart_figure(
    "results/paper_reported_results/support_shapenetpart.csv",
    "results/generated_figures/support_shapenetpart.png",
)
make_support_scannet_curves_figure(
    "results/paper_reported_results/support_scannet_curves.csv",
    "results/generated_figures/support_scannet_curves.png",
)
make_support_shapenetpart_curves_figure(
    "results/paper_reported_results/support_shapenetpart_curves.csv",
    "results/generated_figures/support_shapenetpart_curves.png",
)
make_oracle_scannet_figure(
    "results/paper_reported_results/oracle_scannet.csv",
    "results/generated_figures/oracle_scannet.png",
)
print("wrote generated figures")
PY
