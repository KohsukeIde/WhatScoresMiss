# Reproducibility Map

This document maps paper claims to repository entry points.

| paper component | script/config |
|---|---|
| Concerto target-swap response | `scripts/run_concerto_target_swap.sh`, `configs/concerto/target_swap.yaml` |
| Coordinate-only closure | `scripts/run_concerto_coordinate_closure.sh`, `configs/concerto/coordinate_closure.yaml` |
| Concerto representation-level companion | `scripts/run_concerto_representation_companion.sh`, `configs/concerto/representation_companion.yaml` |
| PointGPT mask/order relaxations | `scripts/run_pointgpt_mask_order_relaxations.sh`, `configs/pointgpt/mask_order_relaxations.yaml` |
| Scene support perturbations | `scripts/run_scene_support_tests.sh`, `configs/scene_support/scannet_support.yaml` |
| Object support perturbations | `scripts/run_object_support_tests.sh`, `configs/object_support/scanobjectnn_support.yaml`, `configs/object_support/shapenetpart_support.yaml` |
| Oracle@k and reported-score diagnostics | `scripts/run_reported_score_oracle.sh`, `configs/reported_score/oracle_scannet.yaml` |
| Class-pair diagnostics | `scripts/run_class_pair_diagnostics.sh`, `configs/reported_score/class_pair_diagnostics.yaml` |
| ScanObjectNN checkpoint reporting | `scripts/run_checkpoint_reporting.sh`, `configs/reported_score/checkpoint_reporting.yaml` |
| Tables from CSVs | `scripts/make_tables.sh` |
| Figures from CSVs | `scripts/make_figures.sh` |

Level 0 reproduces paper tables and figures from `results/paper_reported_results`.
Level 1 validates the diagnostic operators on synthetic arrays. Level 2 requires
official datasets/checkpoints and, for several rows, external model code.

The Level 0 CSVs are paper-reported, anonymized summaries exported from the
experiment ledgers. They are intended to make the submitted tables and figures
auditable without redistributing raw datasets, checkpoints, generated caches, or
cluster-specific run scripts. The Level 2 scripts are reproducibility wrappers:
they document the required inputs and diagnostic entry points, but they are not
claimed to be byte-for-byte copies of the internal scripts used on the original
compute environment.
