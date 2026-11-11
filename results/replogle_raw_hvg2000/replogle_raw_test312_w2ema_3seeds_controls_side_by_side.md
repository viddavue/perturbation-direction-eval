# Replogle HepG2 test perturbations, raw rebuild: generation controls

Median balanced sign accuracy on the **shared panel**: perturbations with >= 5 scored DEGs (target gene excluded) in every version (filtered, all), per report. 95% percentile bootstrap CI over perturbations (2000 resamples, seed 0). Δ = paired median difference scLDM_s0 − method (> 0: scLDM_s0 better) with its bootstrap CI; p = two-sided Wilcoxon signed-rank, raw / Holm within each column (report × version). n = perturbations entering the median: at most the panel size, fewer where the metric is undefined (balanced accuracy needs up and down DEGs) and for the split-half ceiling, which also needs >= 5 DEGs in its half. `—` = not scored there.

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
