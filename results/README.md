# Results

`results/paper_reported_results/` contains compact paper-reported CSV summaries
used for Level 0 table and figure reproduction. These files are not raw model
outputs, raw datasets, checkpoints, or generated caches.

The paper-reported CSVs are sanitized summaries copied from experiment ledgers.
They preserve relative source-result names for provenance, but omit local
machine paths, checkpoint paths, cluster job scripts, and other environment
details that are not appropriate for an anonymized review package.

The full experiment scripts in this repository are lightweight wrappers and
diagnostic entry points for users who already have the official datasets,
checkpoints, and any required external model code. They should not be read as
the exact internal cluster scripts used for the submitted runs.

`results/generated_tables/` and `results/generated_figures/` are produced by:

```bash
bash scripts/make_tables.sh
bash scripts/make_figures.sh
```

Generated files may be deleted and recreated.
