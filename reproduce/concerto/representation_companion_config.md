# Representation Companion

The representation companion rows compare a cross-modal continuation, a
coordinate-MLP continuation, and a no-image-encoder continuation under the same
ScanNet linear-proxy protocol. Full execution requires the external scene
backbone code and the corresponding ScanNet preprocessing.

Ledger:

- `results/ledgers/concerto_representation_companion_runs.csv`

Summary:

- `results/paper_reported_results/concerto_representation_companion.csv`

The reported `gap_recovery_percent` is computed as:

```text
100 * (candidate_score - lower_reference_score)
    / (full_score - lower_reference_score)
```
