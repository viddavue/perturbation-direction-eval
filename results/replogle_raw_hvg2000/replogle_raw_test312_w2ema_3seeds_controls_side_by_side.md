# Replogle HepG2 test perturbations, raw rebuild: generation controls

Median balanced sign accuracy on the **shared panel**: perturbations with >= 5 scored DEGs (target gene excluded) in every version (filtered, all), per report. 95% percentile bootstrap CI over perturbations (2000 resamples, seed 0). Δ = paired median difference scLDM_s0 − method (> 0: scLDM_s0 better) with its bootstrap CI; p = two-sided Wilcoxon signed-rank, raw / Holm within each column (report × version). n = perturbations entering the median: at most the panel size, fewer where the metric is undefined (balanced accuracy needs up and down DEGs) and for the split-half ceiling, which also needs >= 5 DEGs in its half. `—` = not scored there. The third report (changed vs unchanged genes) has its own headline metric and panel rule, stated in its section; its statistics are computed the same way.

## Full effect

Shared panel: **191** perturbations (filtered own 193, all own 194).

| method | filtered (primary) n | filtered (primary) median [95% CI] | Δ [95% CI] | p raw / Holm | all n | all median [95% CI] | Δ [95% CI] | p raw / Holm |
|---|---:|---|---|---|---:|---|---|---|
| **scLDM_s0** (reference) | 190 | 0.934 [0.913, 0.945] |  |  | 190 | 0.938 [0.915, 0.949] |  |  |
| scLDM_w2_ema_cells128 | 190 | 0.939 [0.920, 0.950] | -0.001 [-0.002, +0.000] | 0.00054 / 0.00054 | 190 | 0.937 [0.920, 0.951] | -0.002 [-0.003, +0.000] | 8.7e-05 / 0.00017 |
| scLDM_w1 | 190 | 0.910 [0.883, 0.932] | +0.011 [+0.006, +0.014] | 2.9e-08 / 8.8e-08 | 190 | 0.909 [0.882, 0.933] | +0.012 [+0.005, +0.015] | 1.3e-09 / 3.9e-09 |
| scLDM_w0 | 65 | 0.536 [0.528, 0.545] | +0.391 [+0.363, +0.417] | 3.3e-12 / 1.7e-11 | 65 | 0.536 [0.522, 0.548] | +0.393 [+0.369, +0.435] | 3e-12 / 1.5e-11 |
| scLDM_s0 (shuffled) | 190 | 0.704 [0.661, 0.729] | +0.205 [+0.165, +0.225] | 1.2e-26 / 7.4e-26 | 190 | 0.701 [0.655, 0.728] | +0.204 [+0.161, +0.236] | 1.4e-26 / 8.2e-26 |
| scLDM_w2_ema_cells128 (shuffled) | 190 | 0.700 [0.667, 0.723] | +0.213 [+0.175, +0.232] | 5.3e-27 / 3.7e-26 | 190 | 0.697 [0.663, 0.723] | +0.220 [+0.180, +0.239] | 9.3e-28 / 7.4e-27 |
| scLDM_w1 (shuffled) | 190 | 0.701 [0.669, 0.724] | +0.203 [+0.165, +0.232] | 6.9e-28 / 5.5e-27 | 190 | 0.701 [0.670, 0.719] | +0.193 [+0.167, +0.229] | 1e-27 / 7.4e-27 |
| scLDM_w0 (shuffled) | 58 | 0.536 [0.522, 0.545] | +0.393 [+0.318, +0.414] | 9.7e-11 / 3.9e-10 | 58 | 0.537 [0.516, 0.551] | +0.397 [+0.339, +0.412] | 7.9e-11 / 3.1e-10 |
| noise ceiling (split-half) | 154 | 0.966 [0.959, 0.969] | -0.016 [-0.027, -0.002] | 6e-06 / 1.2e-05 | 154 | 0.963 [0.957, 0.968] | -0.008 [-0.024, +0.000] | 0.00013 / 0.00017 |

Own panels: each version's own >= 5-DEG perturbations, as freeze_results.py would report that eval dir alone.

| method | filtered (primary) n | filtered (primary) median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| scLDM_s0 | 191 | 0.933 [0.913, 0.944] | 192 | 0.938 [0.915, 0.949] |
| scLDM_w2_ema_cells128 | 191 | 0.938 [0.920, 0.949] | 192 | 0.937 [0.919, 0.951] |
| scLDM_w1 | 191 | 0.909 [0.882, 0.930] | 192 | 0.909 [0.883, 0.933] |
| scLDM_w0 | 65 | 0.536 [0.528, 0.545] | 65 | 0.536 [0.522, 0.548] |
| scLDM_s0 (shuffled) | 191 | 0.703 [0.661, 0.725] | 192 | 0.701 [0.655, 0.724] |
| scLDM_w2_ema_cells128 (shuffled) | 191 | 0.700 [0.667, 0.722] | 192 | 0.697 [0.663, 0.723] |
| scLDM_w1 (shuffled) | 191 | 0.700 [0.667, 0.721] | 192 | 0.701 [0.670, 0.719] |
| scLDM_w0 (shuffled) | 58 | 0.536 [0.522, 0.545] | 59 | 0.536 [0.516, 0.548] |
| noise ceiling (split-half) | 156 | 0.965 [0.959, 0.969] | 154 | 0.963 [0.957, 0.968] |

- in the filtered panel only (2): GART, SETX
- in the all panel only (3): GFER, RRP15, USP10

## Perturbation-specific component (shared response projected out)

Shared panel: **190** perturbations (filtered own 192, all own 193).

| method | filtered (primary) n | filtered (primary) median [95% CI] | Δ [95% CI] | p raw / Holm | all n | all median [95% CI] | Δ [95% CI] | p raw / Holm |
|---|---:|---|---|---|---:|---|---|---|
| **scLDM_s0** (reference) | 189 | 0.770 [0.730, 0.793] |  |  | 189 | 0.762 [0.743, 0.784] |  |  |
| scLDM_w2_ema_cells128 | 189 | 0.787 [0.759, 0.808] | -0.009 [-0.014, -0.005] | 5.3e-09 / 1.6e-08 | 189 | 0.778 [0.750, 0.800] | -0.009 [-0.013, -0.004] | 3.9e-10 / 1.6e-09 |
| scLDM_w1 | 189 | 0.716 [0.696, 0.750] | +0.028 [+0.022, +0.038] | 7.5e-10 / 3e-09 | 189 | 0.718 [0.686, 0.743] | +0.031 [+0.022, +0.041] | 7.9e-12 / 3.9e-11 |
| scLDM_w0 | 65 | 0.610 [0.583, 0.636] | +0.163 [+0.132, +0.185] | 2e-06 / 2e-06 | 65 | 0.619 [0.589, 0.636] | +0.137 [+0.120, +0.177] | 4.1e-06 / 4.1e-06 |
| scLDM_s0 (shuffled) | 189 | 0.482 [0.464, 0.505] | +0.236 [+0.206, +0.278] | 4.1e-26 / 2.9e-25 | 189 | 0.486 [0.462, 0.502] | +0.236 [+0.201, +0.273] | 2.5e-26 / 1.8e-25 |
| scLDM_w2_ema_cells128 (shuffled) | 189 | 0.489 [0.463, 0.515] | +0.244 [+0.199, +0.287] | 8.1e-26 / 4.9e-25 | 189 | 0.485 [0.465, 0.512] | +0.240 [+0.200, +0.275] | 2.9e-26 / 1.8e-25 |
| scLDM_w1 (shuffled) | 189 | 0.487 [0.469, 0.504] | +0.240 [+0.212, +0.273] | 9e-28 / 7.2e-27 | 189 | 0.480 [0.468, 0.500] | +0.249 [+0.213, +0.271] | 3.9e-28 / 3.1e-27 |
| scLDM_w0 (shuffled) | 58 | 0.602 [0.578, 0.626] | +0.165 [+0.107, +0.211] | 6e-07 / 1.2e-06 | 58 | 0.614 [0.584, 0.635] | +0.157 [+0.104, +0.208] | 4.2e-07 / 8.4e-07 |
| noise ceiling (split-half) | 154 | 0.849 [0.833, 0.874] | -0.086 [-0.107, -0.035] | 3.5e-11 / 1.8e-10 | 153 | 0.833 [0.820, 0.856] | -0.041 [-0.094, -0.019] | 3.3e-08 / 9.9e-08 |

Own panels: each version's own >= 5-DEG perturbations, as freeze_results.py would report that eval dir alone.

| method | filtered (primary) n | filtered (primary) median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| scLDM_s0 | 190 | 0.767 [0.729, 0.794] | 191 | 0.762 [0.742, 0.783] |
| scLDM_w2_ema_cells128 | 190 | 0.786 [0.759, 0.808] | 191 | 0.778 [0.750, 0.800] |
| scLDM_w1 | 190 | 0.714 [0.696, 0.750] | 191 | 0.718 [0.686, 0.743] |
| scLDM_w0 | 65 | 0.610 [0.583, 0.636] | 65 | 0.619 [0.589, 0.636] |
| scLDM_s0 (shuffled) | 190 | 0.481 [0.464, 0.502] | 191 | 0.486 [0.463, 0.508] |
| scLDM_w2_ema_cells128 (shuffled) | 190 | 0.488 [0.460, 0.515] | 191 | 0.483 [0.464, 0.508] |
| scLDM_w1 (shuffled) | 190 | 0.486 [0.469, 0.504] | 191 | 0.480 [0.468, 0.498] |
| scLDM_w0 (shuffled) | 58 | 0.602 [0.578, 0.626] | 59 | 0.619 [0.580, 0.636] |
| noise ceiling (split-half) | 156 | 0.848 [0.833, 0.873] | 153 | 0.833 [0.820, 0.856] |

- in the filtered panel only (2): GART, SETX
- in the all panel only (3): GFER, RRP15, USP10

## Changed vs unchanged genes

Does the predicted effect **size** separate the genes that change from the genes that clearly do not? (The reports above score the direction, on DEGs only.) Per perturbation, `detect_auroc` is the AUROC of |predicted effect| for its changed genes (the DEGs) against its unchanged genes (not a DEG, |Δ| < 0.05, padj ≥ 0.5); the genes in between (not significant, but not clearly unaffected) are left out. 0.5 = no separation. The table gives the median over the shared panel. Panel rule of this report: >= 5 scored DEGs **and** >= 20 unchanged genes (target gene excluded) in every version that has the report. The split-half ceiling takes the changed / unchanged labels from half A and |Δ| from half B, and needs both counts in its own half.

Shared panel: **191** perturbations (filtered own 193, all own 194).

| method | filtered (primary) n | filtered (primary) median [95% CI] | Δ [95% CI] | p raw / Holm | all n | all median [95% CI] | Δ [95% CI] | p raw / Holm |
|---|---:|---|---|---|---:|---|---|---|
| **scLDM_s0** (reference) | 191 | 0.845 [0.823, 0.864] |  |  | 191 | 0.847 [0.821, 0.865] |  |  |
| scLDM_w2_ema_cells128 | 191 | 0.857 [0.825, 0.874] | -0.007 [-0.009, -0.006] | 2e-19 / 6.1e-19 | 191 | 0.857 [0.824, 0.871] | -0.007 [-0.008, -0.005] | 3.6e-18 / 1.1e-17 |
| scLDM_w1 | 191 | 0.816 [0.797, 0.834] | +0.020 [+0.017, +0.025] | 8.4e-16 / 1.7e-15 | 191 | 0.817 [0.801, 0.834] | +0.020 [+0.017, +0.023] | 2e-13 / 4e-13 |
| scLDM_w0 | 191 | 0.500 [0.500, 0.500] | +0.323 [+0.296, +0.356] | 3.5e-32 / 2.8e-31 | 191 | 0.500 [0.500, 0.500] | +0.326 [+0.300, +0.353] | 5.2e-32 / 4.2e-31 |
| scLDM_s0 (shuffled) | 191 | 0.679 [0.659, 0.711] | +0.113 [+0.095, +0.135] | 2.5e-21 / 1.3e-20 | 191 | 0.685 [0.656, 0.716] | +0.115 [+0.098, +0.136] | 2e-21 / 1e-20 |
| scLDM_w2_ema_cells128 (shuffled) | 191 | 0.695 [0.660, 0.724] | +0.108 [+0.082, +0.134] | 1.4e-19 / 5.5e-19 | 191 | 0.689 [0.663, 0.720] | +0.106 [+0.084, +0.132] | 2e-19 / 8.1e-19 |
| scLDM_w1 (shuffled) | 191 | 0.671 [0.647, 0.695] | +0.125 [+0.106, +0.151] | 2.1e-23 / 1.3e-22 | 191 | 0.671 [0.655, 0.703] | +0.124 [+0.109, +0.159] | 5.7e-23 / 3.4e-22 |
| scLDM_w0 (shuffled) | 191 | 0.500 [0.500, 0.500] | +0.334 [+0.316, +0.357] | 5.9e-32 / 4.2e-31 | 191 | 0.500 [0.500, 0.500] | +0.339 [+0.312, +0.354] | 9.9e-32 / 6.9e-31 |
| noise ceiling (split-half) | 155 | 0.879 [0.871, 0.891] | -0.015 [-0.021, +0.001] | 0.00092 / 0.00092 | 157 | 0.880 [0.869, 0.890] | -0.010 [-0.031, +0.001] | 0.0012 / 0.0012 |

Own panels: each version's own perturbations with >= 5 DEGs and >= 20 unchanged genes.

| method | filtered (primary) n | filtered (primary) median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| scLDM_s0 | 193 | 0.841 [0.823, 0.864] | 194 | 0.846 [0.817, 0.860] |
| scLDM_w2_ema_cells128 | 193 | 0.857 [0.825, 0.872] | 194 | 0.853 [0.822, 0.871] |
| scLDM_w1 | 193 | 0.815 [0.795, 0.834] | 194 | 0.814 [0.797, 0.831] |
| scLDM_w0 | 193 | 0.500 [0.500, 0.500] | 194 | 0.500 [0.500, 0.500] |
| scLDM_s0 (shuffled) | 193 | 0.679 [0.657, 0.711] | 194 | 0.684 [0.655, 0.715] |
| scLDM_w2_ema_cells128 (shuffled) | 193 | 0.695 [0.660, 0.724] | 194 | 0.689 [0.662, 0.720] |
| scLDM_w1 (shuffled) | 193 | 0.669 [0.646, 0.687] | 194 | 0.670 [0.655, 0.701] |
| scLDM_w0 (shuffled) | 193 | 0.500 [0.500, 0.500] | 194 | 0.500 [0.500, 0.500] |
| noise ceiling (split-half) | 158 | 0.877 [0.869, 0.890] | 157 | 0.880 [0.869, 0.890] |

Secondary medians on the shared panel, per version. A gene is called changed if |predicted effect| > τ (τ = 0.1). `three_class_balanced_acc`: mean recall of up-DEGs (predicted > τ), down-DEGs (< −τ) and unchanged genes (|.| ≤ τ); `deg_call_rate` / `unchanged_call_rate`: share of DEGs / unchanged genes called changed; `deg_hit_rate`: share of DEGs called with the correct sign; `abs_pred_*_median`: median |predicted effect| on DEGs / unchanged genes; `detect_auroc_all`: the headline AUROC with every non-DEG (unchanged + grey zone) as the negative class. n = perturbations entering (a metric undefined for a perturbation skips it). These depend on τ and on the scale of the prediction; the headline does not.

filtered (primary):

| method | n | `three_class_balanced_acc` | `deg_call_rate` | `deg_hit_rate` | `unchanged_call_rate` | `abs_pred_deg_median` | `abs_pred_unchanged_median` | `detect_auroc_all` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scLDM_s0 | 191 | 0.735 | 0.778 | 0.753 | 0.231 | 0.237 | 0.049 | 0.794 |
| scLDM_w2_ema_cells128 | 191 | 0.735 | 0.777 | 0.739 | 0.198 | 0.214 | 0.044 | 0.805 |
| scLDM_w1 | 191 | 0.674 | 0.636 | 0.631 | 0.121 | 0.159 | 0.036 | 0.776 |
| scLDM_w0 | 191 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| scLDM_s0 (shuffled) | 191 | 0.460 | 0.400 | 0.313 | 0.088 | 0.077 | 0.028 | 0.654 |
| scLDM_w2_ema_cells128 (shuffled) | 191 | 0.471 | 0.372 | 0.307 | 0.077 | 0.071 | 0.026 | 0.653 |
| scLDM_w1 (shuffled) | 191 | 0.453 | 0.263 | 0.232 | 0.053 | 0.061 | 0.023 | 0.632 |
| scLDM_w0 (shuffled) | 191 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| noise ceiling (split-half) | 155 | 0.781 | 0.895 | 0.887 | 0.398 | 0.319 | 0.078 | 0.839 |

all:

| method | n | `three_class_balanced_acc` | `deg_call_rate` | `deg_hit_rate` | `unchanged_call_rate` | `abs_pred_deg_median` | `abs_pred_unchanged_median` | `detect_auroc_all` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scLDM_s0 | 191 | 0.725 | 0.778 | 0.752 | 0.233 | 0.234 | 0.050 | 0.789 |
| scLDM_w2_ema_cells128 | 191 | 0.731 | 0.765 | 0.734 | 0.200 | 0.215 | 0.043 | 0.798 |
| scLDM_w1 | 191 | 0.675 | 0.647 | 0.640 | 0.126 | 0.156 | 0.035 | 0.769 |
| scLDM_w0 | 191 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| scLDM_s0 (shuffled) | 191 | 0.463 | 0.390 | 0.329 | 0.091 | 0.077 | 0.028 | 0.658 |
| scLDM_w2_ema_cells128 (shuffled) | 191 | 0.466 | 0.378 | 0.316 | 0.074 | 0.069 | 0.026 | 0.655 |
| scLDM_w1 (shuffled) | 191 | 0.429 | 0.261 | 0.200 | 0.051 | 0.052 | 0.023 | 0.637 |
| scLDM_w0 (shuffled) | 191 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| noise ceiling (split-half) | 157 | 0.780 | 0.878 | 0.868 | 0.379 | 0.308 | 0.075 | 0.837 |

- in the filtered panel only (2): GART, SETX
- in the all panel only (3): GFER, RRP15, USP10
