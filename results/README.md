# Frozen result tables

Small, commit-able tables that the report cites. `outputs/` stays gitignored; this directory is
the single source of truth.

Regenerate with:

```bash
python experiments/perturbation_direction/freeze_results.py \
  --eval-dir outputs/perturbation_direction/replogle/eval_freeze_20261021 \
  --tag replogle_hepg2_hvg2000 --results-dir results
```

Each tag produces six files:

| file | content |
|---|---|
| `_summary.csv` | median balanced accuracy per method with a 95% percentile bootstrap CI |
| `_summary_raw.csv` | `evaluate.py`'s own summary, kept so the point estimates stay auditable |
| `_summary_paired.csv` | paired tests vs the reference method (median difference, CI, Wilcoxon p, Holm) |
| `_summary_specific*.csv` | the same three for the perturbation-specific report |
| `<tag>.provenance.json` | git commit, input hashes, bootstrap/test settings |

## Conventions and caveats

- **Unit of uncertainty is the perturbation**, not the gene or the cell. Intervals describe
  variability across this panel and are conditional on it; they do not generalise to new
  perturbations.
- **Comparisons are paired** on perturbations both methods score; `n` differs per row for that
  reason.
- **Holm correction** is applied within one declared family: `scLDM_w1 vs each other method`.
  Raw p-values are kept beside the corrected ones.
- **The noise ceiling has fewer perturbations** (201 vs 251) because its DEGs come from half the
  cells, so it clears the >=5-DEG filter less often. Comparisons involving it use the smaller
  paired cohort.
- **Shuffled and empirical baselines are included deliberately** as controls; a table where the
  shuffled control sits near chance is the evidence that the metric is not trivially satisfiable.

## Replogle HepG2, current code (`f885234`)

Reproduced exactly against the earlier run: every metric and method matched to 0.0 absolute
difference, so these numbers are stable on the current code.

| method | n | median | 95% CI |
|---|---:|---:|---|
| noise ceiling (split-half) | 201 | 0.965 | 0.961-0.969 |
| kNN | 251 | 0.936 | 0.917-0.952 |
| **scLDM w1** | 251 | **0.911** | 0.888-0.930 |
| MLP | 251 | 0.898 | 0.878-0.917 |
| calibrated | 251 | 0.894 | 0.883-0.906 |
| bilinear | 251 | 0.875 | 0.847-0.906 |
| generic | 251 | 0.864 | 0.840-0.875 |
| GBDT | 251 | 0.850 | 0.833-0.864 |
| transfer | 251 | 0.821 | 0.811-0.829 |
| prior | 251 | 0.800 | 0.786-0.818 |
| scLDM shuffled | 251 | 0.738 | 0.700-0.755 |

Paired against scLDM w1: kNN **-0.013** (-0.020 to -0.009, p < 1e-4) — a statistically clear but
small deficit; MLP +0.010 (p=0.010) and calibrated +0.018 (p=0.017) are small wins; every other
baseline loses by 0.02-0.16 with p < 1e-4.
