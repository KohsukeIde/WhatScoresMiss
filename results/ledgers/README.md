# Provenance Summaries

These CSV files provide sanitized row-level provenance summaries for the
paper-reported diagnostics. They are derived from the paper-reported
experimental records. They are not raw run logs, raw model outputs, raw
datasets, model checkpoints, generated caches, cluster logs, or personal-path
records.

Each row identifies the script, config, condition, checkpoint source, dataset,
split, metric, value, and notes associated with a reported diagnostic row. The
CSV files are intended to make the paper tables auditable without redistributing
large or private artifacts.

The paper-summary CSVs under `results/paper_reported_results/` are compact
Level 0 table/figure inputs. Regenerate the sanitized provenance summaries from
those compact summaries with:

```bash
python reproduce/external/build_ledgers_from_summaries.py
```
