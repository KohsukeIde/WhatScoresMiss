# External Code Policy

This repository does not vendor full third-party model codebases. Full empirical
diagnostics require the corresponding official implementations, official
datasets, and checkpoints. The files under `reproduce/` provide the diagnostic
entry points, sanitized patch excerpts, configs, provenance-summary builders,
and aggregation scripts used to make the empirical claims auditable.

The `results/paper_reported_results/` CSVs are compact paper-summary exports.
The `results/ledgers/` CSVs provide row-level provenance summaries for those
records. They are not raw model outputs, raw cluster logs, checkpoints, or
dataset artifacts. Synthetic tests in `src/bds/` validate operators; they are
not empirical reproduction of the paper results.
