# Compute

Level 0 and Level 1 are CPU-compatible. They reproduce tables/figures from
small CSV files and run synthetic tests for diagnostic operators.

Full scene-level and object-level diagnostics require GPUs, local dataset
storage, official checkpoints, and the corresponding external model codebases.
Full runs are compute-intensive and depend on the chosen backbone, dataset,
cache layout, and checkpoint schedule. This repository provides lightweight
diagnostic operators, paper-reported CSV summaries, and wrappers/configs for
running the diagnostic operations when the required external assets are present.

| experiment | expected hardware | approximate runtime | notes |
|---|---|---|---|
| table/figure reproduction | CPU | seconds-minutes | reads compact CSVs only |
| synthetic smoke tests | CPU | seconds-minutes | no datasets/checkpoints |
| Concerto target-swap response | GPU recommended | GPU recommended; runtime depends on dataset/cache size and external model code | requires scene datasets and external Concerto-compatible feature extraction |
| coordinate-only closure | CPU/GPU depending on cache size | minutes once logits or target caches are available | closure fitting is lightweight after target-cache extraction |
| representation companion | GPU required for continuation/proxy runs | GPU required for full training; runtime depends on backbone, split, and checkpoint schedule | wrapper documents required paths/configs |
| PointGPT mask/order relaxations | GPU required for full training | GPU required for full training; runtime depends on backbone, split, and checkpoint schedule | relaxed pretraining and downstream fine-tuning require the external PointGPT codebase |
| scene support perturbations | GPU recommended | GPU recommended; runtime depends on dataset/cache size and external model code | can be run from existing scene checkpoints when logits/features are available |
| object support perturbations | GPU recommended | GPU recommended; runtime depends on dataset/cache size and external model code | can be run from existing object checkpoints |
| Oracle@k reported-score diagnostics | CPU/GPU depending on logit extraction | minutes once logits are cached | logit extraction may require GPU, but Oracle@k computation is lightweight |
| checkpoint reporting | GPU required for held-out fine-tuning | GPU required for full training; runtime depends on backbone, split, and checkpoint schedule | requires checkpoint trajectories or explicit held-out fine-tuning runs |
