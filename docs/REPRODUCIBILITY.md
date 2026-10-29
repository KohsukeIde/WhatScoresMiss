# Reproducibility Map

This document maps paper claims to repository entry points, sanitized
provenance summaries, and compact summary CSVs.

Level 0 generates tables and plots from `results/paper_reported_results`; plot
styling may differ from the figures in the paper.
Level 1 validates the diagnostic operators on synthetic arrays. Synthetic tests
do not reproduce the empirical paper results. Level 2 provides real diagnostic
entry points, sanitized patch excerpts, configs, and provenance-summary
aggregators; full execution requires official datasets/checkpoints and, for
several rows, external model code.

The patch files listed below are sanitized excerpts that identify the relevant
code changes. They are not complete vendored replacements for the external
repositories, and full execution requires applying or adapting the corresponding
changes inside the listed external codebases.

| paper component | Level 2 entry point | config or patch | required external assets | expected ledger/output | Level 0 summary |
|---|---|---|---|---|---|
| Concerto target-swap response | `scripts/run_concerto_target_swap.sh`; `reproduce/concerto/target_swap.py` | `reproduce/concerto/configs/*_target_swap.yaml`; `reproduce/concerto/patches/concerto_target_swap.patch` | Concerto-compatible code, scene datasets, `CONCERTO_CKPT` | `results/ledgers/concerto_loss_runs.csv` | `results/paper_reported_results/concerto_pretext_diagnostics.csv` |
| Coordinate-only closure | `scripts/run_concerto_coordinate_closure.sh`; `reproduce/concerto/coordinate_only_closure.py` | `configs/concerto/coordinate_closure.yaml`; `reproduce/concerto/patches/concerto_coordinate_target.patch` | target caches or external Concerto-compatible extraction code | `results/ledgers/concerto_closure_fit_runs.csv` | `results/paper_reported_results/concerto_pretext_diagnostics.csv` |
| Concerto representation-level companion | `scripts/run_concerto_representation_companion.sh`; `reproduce/concerto/representation_companion.py` | `configs/concerto/representation_companion.yaml`; `reproduce/concerto/representation_companion_config.md` | Concerto-compatible continuation/proxy code, ScanNet, checkpoint | `results/ledgers/concerto_representation_companion_runs.csv` | `results/paper_reported_results/concerto_representation_companion.csv` |
| PointGPT mask/order relaxations | `scripts/run_pointgpt_mask_order_relaxations.sh`; `reproduce/pointgpt/aggregate_pointgpt_results.py` | `reproduce/pointgpt/configs/*.yaml`; `reproduce/pointgpt/patches/disable_mask.patch`; `reproduce/pointgpt/patches/randomize_order.patch` | official PointGPT code, ScanObjectNN, ShapeNetPart, trained relaxed checkpoints | `results/ledgers/pointgpt_pretext_logs.csv`; `results/ledgers/pointgpt_downstream_eval_runs.csv` | `results/paper_reported_results/pointgpt_mask_order_relaxations.csv` |
| Scene support perturbations | `scripts/run_scene_support_tests.sh`; `reproduce/support/apply_scene_support.py`; `reproduce/support/full_scene_propagation.py` | `configs/scene_support/scannet_support.yaml` | external scene model code, ScanNet, scene checkpoints/logits | `results/ledgers/scannet_support_eval_runs.csv` | `results/paper_reported_results/support_scannet.csv`; `results/paper_reported_results/support_scannet_curves.csv` |
| Object support perturbations | `scripts/run_object_support_tests.sh`; `reproduce/support/apply_object_support.py`; `reproduce/support/score_retained_points.py` | `configs/object_support/scanobjectnn_support.yaml`; `configs/object_support/shapenetpart_support.yaml` | PointGPT/Point-MAE/PCP-MAE code, ScanObjectNN, ShapeNetPart, checkpoints/logits | `results/ledgers/shapenetpart_support_eval_runs.csv` | `results/paper_reported_results/support_shapenetpart.csv`; `results/paper_reported_results/support_shapenetpart_curves.csv` |
| Oracle@k reported-score diagnostics | `scripts/run_reported_score_oracle.sh`; `reproduce/reported_score/oracle_from_logits.py` | `configs/reported_score/oracle_scannet.yaml` | scene logits and labels from external model code | `results/ledgers/scannet_oracle_eval_runs.csv` | `results/paper_reported_results/oracle_scannet.csv`; `results/paper_reported_results/picture_wall_oracle_scannet.csv` |
| Class-pair diagnostics | `scripts/run_class_pair_diagnostics.sh`; `reproduce/reported_score/class_pair_diagnostics.py` | `configs/reported_score/class_pair_diagnostics.yaml` | class-pair logits/features from external scene code | `results/ledgers/class_pair_diagnostic_runs.csv` | `results/paper_reported_results/class_pair_diagnostics_scannet.csv` |
| ScanObjectNN checkpoint reporting | `scripts/run_checkpoint_reporting.sh`; `reproduce/reported_score/checkpoint_reporting.py` | `configs/reported_score/checkpoint_reporting.yaml` | official/held-out checkpoint trajectories for object models | `results/ledgers/scanobjectnn_checkpoint_selection_runs.csv` | `results/paper_reported_results/checkpoint_reporting_scanobjectnn.csv` |
| Tables from CSVs | `scripts/make_tables.sh` | none | none | `results/generated_tables/` | all `results/paper_reported_results/*.csv` |
| Figures from CSVs | `scripts/make_figures.sh` | none | none | `results/generated_figures/` | selected `results/paper_reported_results/*.csv` |

The original PointGPT rebuttal retention driver, helper modules, grouping patch,
and setup instructions are in `reproduce/pointgpt/original_r7c/`.

For ShapeNetPart retained-point scoring, `reproduce/support/score_retained_points.py`
expects one `.npz` file per shape with `logits`, per-input-position `target`,
`original_indices`, and the category-specific `valid_part_ids`. It averages
logits for duplicate original IDs, restricts predictions to the valid parts,
assigns IoU 1 to a valid part absent from prediction and target, and then averages
per-shape IoU across the supplied files. For example:

```bash
python -m reproduce.support.score_retained_points --logits-npz shape_1.npz shape_2.npz
```

The original PointGPT rebuttal scorer can be checked on synthetic inputs when
the original scorer and replay CSV are available:

```bash
python scripts/verify_pointgpt_scoring.py \
  --reference /path/to/rebuttal_eval/r7.py \
  --reference-csv /path/to/original/replay_curves.csv \
  --pointgpt-code-root /path/to/original/PointGPT
```

This checks the scoring operation; it does not run PointGPT inference.

The Level 0 CSVs are paper-reported summaries. The
`results/ledgers/` files are sanitized provenance summaries derived from the
paper-reported experimental records; they are not raw model outputs, raw run
logs, or checkpoint artifacts. The Level 2 scripts are reproducibility wrappers
around the diagnostic code in `reproduce/`: they document the required inputs and
entry points, but they are not claimed to be byte-for-byte copies of internal
cluster scripts.
