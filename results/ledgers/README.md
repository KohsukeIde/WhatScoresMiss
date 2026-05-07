# Run Ledgers

These CSV files provide sanitized row-level provenance for the paper-reported
diagnostics. They are not raw datasets, large logs, model checkpoints, or
generated caches. Each row records the experiment id, model, dataset, split,
condition, checkpoint source, seed when applicable, metric, value, source
script, config, and notes.

The paper-summary CSVs under `results/paper_reported_results/` are compact
table/figure inputs exported from these ledgers.

Regenerate ledgers from the compact summaries with:

```bash
python reproduce/external/build_ledgers_from_summaries.py
```
