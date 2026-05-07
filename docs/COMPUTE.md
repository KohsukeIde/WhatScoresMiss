# Compute

Level 0 and Level 1 are CPU-compatible. They reproduce tables/figures from
small CSV files and run synthetic tests for diagnostic operators.

Full scene-level and object-level experiments require GPUs and dataset storage.
Full runs are compute-intensive and depend on the chosen backbone, dataset, and
checkpoint. The paper reports diagnostics rather than a new training pipeline.
We provide scripts and configs to reproduce the diagnostic operations when the
required datasets and checkpoints are available.

| experiment | expected hardware | approximate runtime | notes |
|---|---|---|---|
| table/figure reproduction | CPU | seconds to minutes | reads compact CSVs only |
| synthetic smoke tests | CPU | seconds to minutes | no datasets/checkpoints |
| Concerto target-swap response | GPU recommended | fill before submission | depends on cache size and scene datasets |
| coordinate-only closure | GPU recommended | fill before submission | depends on target-cache extraction and closure fit |
| representation companion | GPU recommended | fill before submission | continuation/proxy readout runtime depends on backbone and data |
| PointGPT mask/order relaxations | GPU required for full training | fill before submission | relaxed pretraining and downstream fine-tuning are heavy |
| scene support perturbations | GPU recommended | fill before submission | can be run from existing scene checkpoints |
| object support perturbations | GPU recommended | fill before submission | can be run from existing object checkpoints |
| Oracle@k reported-score diagnostics | CPU/GPU depending on logit extraction | fill before submission | fast if logits are already cached |
| checkpoint reporting | GPU required for held-out fine-tuning | fill before submission | depends on number of checkpoints and splits |
