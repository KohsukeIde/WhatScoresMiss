# Beyond Downstream Scores: Reproducibility Package

This repository contains anonymized reproduction code for a NeurIPS 2026
Evaluations & Datasets Track submission. The paper does not introduce a new
dataset. Raw datasets are not redistributed.

The repository provides:

1. CPU smoke tests for diagnostic operators.
2. Figure and table reproduction from small CSV files.
3. Entry-point wrappers and configs for full diagnostics when official
   datasets, checkpoints, and external model implementations are available.

## Reproducibility Levels

### Level 0: Tables and Figures from CSVs

This level does not require raw datasets, checkpoints, or GPUs.

```bash
conda env create -f environment.yml
conda activate bds
bash scripts/make_tables.sh
bash scripts/make_figures.sh
```

Generated markdown tables are written to `results/generated_tables/`.
Generated figures are written to `results/generated_figures/`.

### Level 1: Unit Tests and Synthetic Smoke Tests

This level runs diagnostic operators on synthetic arrays.

```bash
pip install -e .
pytest -q
bash scripts/smoke_test.sh
```

### Level 2: Full-diagnostic entry points and configuration wrappers

This level requires official datasets, checkpoints, and suitable GPU hardware.
Set the required local paths first:

```bash
export SCANNET_ROOT=/path/to/scannet
export SCANOBJECTNN_ROOT=/path/to/scanobjectnn
export SHAPENETPART_ROOT=/path/to/shapenetpart
export ARKITSCENES_ROOT=/path/to/arkitscenes
export SCANNETPP_ROOT=/path/to/scannetpp
export S3DIS_ROOT=/path/to/s3dis
export HM3D_ROOT=/path/to/hm3d
export STRUCTURED3D_ROOT=/path/to/structured3d
export CHECKPOINT_ROOT=/path/to/checkpoints
export OUTPUT_ROOT=/path/to/outputs
```

Then run the relevant diagnostic wrappers:

```bash
bash scripts/run_concerto_target_swap.sh
bash scripts/run_concerto_coordinate_closure.sh
bash scripts/run_concerto_representation_companion.sh
bash scripts/run_pointgpt_mask_order_relaxations.sh
bash scripts/run_scene_support_tests.sh
bash scripts/run_object_support_tests.sh
bash scripts/run_reported_score_oracle.sh
bash scripts/run_class_pair_diagnostics.sh
bash scripts/run_checkpoint_reporting.sh
```

The Level 2 scripts validate environment variables and print explicit messages
when a full external training or evaluation implementation is required. They do
not pretend to run heavy experiments when datasets or checkpoints are absent.
These scripts do not bundle the full external model implementations. They
validate required dataset/checkpoint paths and document the diagnostic entry
points. Full execution requires official datasets, checkpoints, and the
corresponding external model codebases. Use the configs listed in
`docs/REPRODUCIBILITY.md`.

## Repository Map

- `src/bds/`: small, CPU-compatible diagnostic operators.
- `tests/`: synthetic unit tests for the operators.
- `configs/`: documented diagnostic configuration stubs.
- `scripts/`: Level 0/1 reproduction scripts and Level 2 wrappers.
- `results/paper_reported_results/`: compact CSVs used only for table and
  figure reproduction.
- `docs/`: dataset, checkpoint, compute, reproducibility, and third-party notes.

## Notes for Review

The CSV files under `results/paper_reported_results/` are paper-reported
summaries for Level 0 reproduction. They are not raw model outputs and should
not be interpreted as redistributed datasets or checkpoints. Full reproduction
requires the official datasets and checkpoints listed in `docs/DATA.md` and
`docs/CHECKPOINTS.md`.
