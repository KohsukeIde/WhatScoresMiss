# External Code Policy

This repository does not vendor full third-party model codebases. Full empirical
diagnostics require the corresponding official implementations, official
datasets, and checkpoints. The files under `reproduce/` provide the diagnostic
entry points, sanitized patch excerpts, configs, run ledgers, and aggregation
scripts used to make the empirical claims auditable.

The `results/paper_reported_results/` CSVs are compact paper-summary exports.
The `results/ledgers/` CSVs provide row-level provenance for those summaries.
Synthetic tests in `src/bds/` validate operators; they are not empirical
reproduction of the paper results.
