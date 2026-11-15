# Frozen result tables

Small, commit-able tables that the report cites. `outputs/` stays gitignored; this directory is
the single source of truth.

## Guide to the current report

The report's main comparison uses the **raw rebuild shared panel**, not the older
STATE-route summaries described below:

- Main results, paired comparisons and filtered-versus-all robustness:
  [`replogle_raw_hvg2000/replogle_raw_test312_w2ema_3seeds_side_by_side.md`](replogle_raw_hvg2000/replogle_raw_test312_w2ema_3seeds_side_by_side.md).
  Its companion `_specific.csv`, `_unchanged.csv`, `_long.csv`, `_panel.csv` and
  provenance JSON contain the corresponding machine-readable tables and inputs.
- Generation controls: the
  [`controls_side_by_side.md`](replogle_raw_hvg2000/replogle_raw_test312_w2ema_3seeds_controls_side_by_side.md)
  table and its companion CSVs and provenance JSON.
- The earlier `replogle_test312_w2ema_3seeds_summary*.csv` files are legacy STATE-route
  results, not the source for the current report's shared-panel table.

The report also notes that its expression-quartile and unchanged-threshold sensitivity
analysis is not in the committed result tables; it needs independent verification before
being cited as a reproducible result. See [methods](../docs/METHODS.md) and
[server instructions](../docs/SERVER.md) for definitions and reproduction.

Regenerate with:

```bash
python experiments/perturbation_direction/evaluation/freeze_results.py \
  --eval-dir outputs/perturbation_direction/legacy/replogle/eval_freeze_20261021 \
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
- **Changed vs unchanged genes is a third report with its own conventions** (`side_by_side.py`
  only; METHODS.md section g). File `_side_by_side_unchanged.csv`, rows `report == unchanged` in
  `_side_by_side_long.csv` and `_side_by_side_panel.csv` (which gains `n_unchanged_<version>`
  columns), a third section in `_side_by_side.md`. Headline: median `detect_auroc`, the AUROC of
  |predicted Δ| for DEGs vs clearly unchanged genes (not a DEG, |Δ| < 0.05, padj >= 0.5; genes in
  between are left out). Panel: >= 5 DEGs **and** >= 20 unchanged genes in both versions. MLP and
  prior are direction-only (P(up) − 0.5, no effect size) and are listed as not scored. Tables
  generated before this report existed (commit `ab1287d` and earlier, including every table
  described below) do not have it.

## Replogle HepG2, earlier STATE-route results (`f885234`)

These tables were produced from STATE's combined log-normalised matrix (372-perturbation panel,
scLDM w1). Kept as a frozen reference. The raw rebuild below reproduces the STATE-route data and
ground truth exactly (gates A1–A5); its filtered own-panel numbers equal the frozen test-312 tables
(`replogle_test312_w2ema_3seeds_*`) except for the now-seeded MLP.

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

## Replogle HepG2, raw rebuild: filtered | all side by side

Tables written by `side_by_side.py` into `results/replogle_raw_hvg2000/` with tag
`replogle_raw_test312_w2ema_3seeds`, generated on student03 from the clean commit `108a8c6`
(provenance `dirty=false`). The full-effect and perturbation-specific wide tables are byte-identical
to the earlier run from the clean commit `ab1287d`; `108a8c6` adds the changed-vs-unchanged report.

Output files per `side_by_side.py table`:

| file | content |
|---|---|
| `_side_by_side.md` | main markdown table (shared panel, full + specific, own panels, vs frozen) |
| `_side_by_side.csv` | full-effect wide table: one row per method, columns `<version>_<stat>` |
| `_side_by_side_specific.csv` | same for the perturbation-specific report |
| `_side_by_side_unchanged.csv` | same for the changed-vs-unchanged report (headline `detect_auroc`) |
| `_side_by_side_long.csv` | all numbers in long form (`report`, `version`, `method`, stats) |
| `_side_by_side_panel.csv` | per report and perturbation: n_deg per version, panel membership |
| `_side_by_side_vs_frozen.csv` | filtered own panel vs the frozen STATE-based summary tables |
| `<tag>.provenance.json` | git commit + dirty flag, sha256 of every input, resolved config, timestamp |
| `_controls_*` | a second table (`_controls` tag) with the same structure, for generation controls |

Columns per version in the wide tables: `status`, `n`, `median`, `ci_low`, `ci_high`, `n_paired`,
`diff`, `diff_ci_low`, `diff_ci_high`, `p_raw`, `p_holm`, `n_own`, `median_own`, `ci_low_own`,
`ci_high_own`.

### Full effect, shared panel (191 perturbations; filtered own 193, all own 194)

| method | filtered n | filtered median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| **scLDM_s0** (reference) | 190 | 0.934 [0.913, 0.945] | 190 | 0.938 [0.915, 0.949] |
| scLDM_s1 | 190 | 0.926 [0.901, 0.942] | 190 | 0.927 [0.904, 0.946] |
| scLDM_s2 | 190 | 0.934 [0.911, 0.945] | 190 | 0.932 [0.910, 0.943] |
| knn | 190 | 0.938 [0.916, 0.950] | 190 | 0.940 [0.928, 0.955] |
| mlp | 190 | 0.892 [0.875, 0.906] | 190 | 0.910 [0.885, 0.925] |
| generic | 190 | 0.859 [0.835, 0.875] | 190 | 0.855 [0.840, 0.879] |
| transfer | 190 | 0.821 [0.810, 0.829] | 190 | 0.830 [0.820, 0.839] |
| prior | 190 | 0.800 [0.783, 0.817] | 190 | 0.805 [0.792, 0.816] |
| scLDM_s0 (shuffled) | 190 | 0.704 [0.661, 0.729] | 190 | 0.701 [0.655, 0.728] |
| scLDM_s1 (shuffled) | 190 | 0.695 [0.658, 0.721] | 190 | 0.688 [0.651, 0.723] |
| scLDM_s2 (shuffled) | 190 | 0.703 [0.664, 0.734] | 190 | 0.693 [0.667, 0.724] |
| noise ceiling (split-half) | 154 | 0.966 [0.959, 0.969] | 154 | 0.963 [0.957, 0.968] |

### Perturbation-specific, shared panel (190 perturbations; filtered own 192, all own 193)

| method | filtered n | filtered median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| **scLDM_s0** (reference) | 189 | 0.770 [0.730, 0.793] | 189 | 0.762 [0.743, 0.784] |
| scLDM_s1 | 189 | 0.755 [0.715, 0.787] | 189 | 0.750 [0.729, 0.780] |
| scLDM_s2 | 189 | 0.774 [0.746, 0.799] | 189 | 0.766 [0.734, 0.790] |
| knn | 189 | 0.820 [0.792, 0.833] | 189 | 0.808 [0.768, 0.831] |
| mlp | 189 | 0.731 [0.711, 0.750] | 189 | 0.744 [0.722, 0.758] |
| generic | — | — | — | — |
| transfer | 189 | 0.659 [0.646, 0.682] | 189 | 0.680 [0.663, 0.692] |
| prior | 189 | 0.524 [0.513, 0.543] | 189 | 0.515 [0.503, 0.531] |
| scLDM_s0 (shuffled) | 189 | 0.482 [0.464, 0.505] | 189 | 0.486 [0.462, 0.502] |
| scLDM_s1 (shuffled) | 189 | 0.493 [0.467, 0.511] | 189 | 0.487 [0.463, 0.513] |
| scLDM_s2 (shuffled) | 189 | 0.489 [0.464, 0.506] | 189 | 0.491 [0.455, 0.505] |
| noise ceiling (split-half) | 154 | 0.849 [0.833, 0.874] | 153 | 0.833 [0.820, 0.856] |

The generic baseline is not scored in the perturbation-specific report: it is the shared response
itself, which that report projects out.

Paired differences vs scLDM_s0 on the shared panel (Δ = scLDM_s0 − method, > 0: scLDM better;
p raw / Holm), copied from `replogle_raw_test312_w2ema_3seeds_side_by_side.md`:

| report | method | filtered Δ [95% CI] | filtered p | all Δ [95% CI] | all p |
|---|---|---|---|---|---|
| full | knn | -0.003 [-0.008, +0.000] | 0.099 / 0.2 | -0.005 [-0.009, +0.000] | 0.18 / 0.18 |
| full | mlp | +0.022 [+0.014, +0.029] | 6.3e-07 / 3.2e-06 | +0.011 [+0.004, +0.018] | 0.0015 / 0.006 |
| specific | knn | -0.031 [-0.044, -0.017] | 1.1e-05 / 4.5e-05 | -0.025 [-0.037, -0.008] | 0.0006 / 0.0024 |
| specific | mlp | +0.030 [+0.015, +0.038] | 0.00037 / 0.0011 | +0.021 [-0.003, +0.032] | 0.47 / 0.59 |

generic, transfer, prior and every shuffled control lose to scLDM_s0 in both versions and both reports
(Holm p <= 3e-5); the other two scLDM seeds differ from seed 0 by at most 0.002 (paired median); the
split-half noise ceiling lies above scLDM. The full table is in the `.md` file.

### Changed vs unchanged genes, shared panel (191 perturbations; filtered own 193, all own 194)

Median per-perturbation AUROC of |predicted Δ| for DEGs vs clearly unchanged genes (not a DEG,
|Δ| < 0.05, padj >= 0.5); Δ = scLDM_s0 − method, p raw / Holm. MLP and prior are not scored
(direction confidences, not effect sizes).

| method | filtered median [95% CI] | filtered Δ, p | all median [95% CI] | all Δ, p |
|---|---|---|---|---|
| **scLDM_s0** (reference) | 0.845 [0.823, 0.864] | | 0.847 [0.821, 0.865] | |
| scLDM_s1 | 0.839 [0.814, 0.863] | +0.002, 0.061 / 0.12 | 0.837 [0.815, 0.861] | +0.002, 0.067 / 0.13 |
| scLDM_s2 | 0.849 [0.819, 0.865] | +0.001, 0.48 / 0.48 | 0.849 [0.821, 0.864] | +0.000, 0.4 / 0.4 |
| knn | 0.879 [0.860, 0.892] | −0.023, 1.4e-14 / 8.4e-14 | 0.872 [0.856, 0.890] | −0.017, 1.2e-14 / 7.5e-14 |
| generic | 0.821 [0.805, 0.832] | +0.021, 0.034 / 0.1 | 0.817 [0.805, 0.833] | +0.025, 0.016 / 0.047 |
| transfer | 0.743 [0.731, 0.756] | +0.070, 2.7e-10 / 1.3e-09 | 0.755 [0.746, 0.769] | +0.067, 3.9e-09 / 1.9e-08 |
| scLDM_s0 (shuffled) | 0.679 [0.659, 0.711] | +0.113, 2.5e-21 / 2e-20 | 0.685 [0.656, 0.716] | +0.115, 2e-21 / 1.8e-20 |
| noise ceiling (split-half) | 0.879 [0.871, 0.891] (n=155) | −0.015, 0.00092 / 0.0037 | 0.880 [0.869, 0.890] (n=157) | −0.010, 0.0012 / 0.0048 |

kNN reaches the split-half ceiling here: half-B effects carry the sampling noise of half the cells,
so this ceiling is a noise reference, not an upper bound (METHODS.md g). Secondary medians
(three-class balanced accuracy, call rates) are in the `.md` file.

### Own panels (each version's own ≥ 5-DEG perturbations)

Full effect, filtered own: scLDM_s0 0.933 [0.913, 0.944] (n=191), scLDM_s1 0.926 [0.900, 0.941],
scLDM_s2 0.934 [0.909, 0.944], knn 0.936 [0.916, 0.951], mlp 0.890 [0.875, 0.905], generic
0.858 [0.833, 0.875], transfer 0.821 [0.810, 0.830], prior 0.800 [0.783, 0.818], ceiling 0.965
[0.959, 0.969] (n=156). Full effect, all own: scLDM_s0 0.938 [0.915, 0.949] (n=192), scLDM_s1
0.929 [0.907, 0.946], scLDM_s2 0.933 [0.912, 0.944], knn 0.939 [0.926, 0.955], mlp 0.910
[0.884, 0.924], generic 0.854 [0.833, 0.877], transfer 0.830 [0.820, 0.841], prior 0.805
[0.793, 0.816], ceiling 0.963 [0.957, 0.968] (n=154).

Perturbation-specific, filtered own: scLDM_s0 0.767 [0.729, 0.794] (n=190), scLDM_s1 0.754
[0.714, 0.787], scLDM_s2 0.772 [0.746, 0.799], knn 0.821 [0.795, 0.833], mlp 0.730
[0.711, 0.750], transfer 0.659 [0.643, 0.682], prior 0.524 [0.514, 0.543], ceiling 0.848
[0.833, 0.873] (n=156). Perturbation-specific, all own: scLDM_s0 0.762 [0.742, 0.783] (n=191),
scLDM_s1 0.750 [0.729, 0.780], scLDM_s2 0.766 [0.734, 0.790], knn 0.803 [0.768, 0.831], mlp
0.745 [0.724, 0.758], transfer 0.680 [0.663, 0.692], prior 0.515 [0.503, 0.531], ceiling 0.833
[0.820, 0.856] (n=153).

### vs frozen STATE tables (filtered own panel only)

All rows match to 0.0000 except the MLP, which is now seeded and differs by −0.0017 (full) /
−0.0065 (specific). The frozen tables also contain calibrated, bilinear and GBDT rows that have
been removed from the current baseline set.
