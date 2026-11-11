# Replogle HepG2 test perturbations, raw rebuild: filtered all side by side

Median balanced sign accuracy on the **shared panel**: perturbations with >= 5 scored DEGs (target gene excluded) in every version (filtered, all), per report. 95% percentile bootstrap CI over perturbations (2000 resamples, seed 0). Δ = paired median difference scLDM_s0 − method (> 0: scLDM_s0 better) with its bootstrap CI; p = two-sided Wilcoxon signed-rank, raw / Holm within each column (report × version). n = perturbations entering the median: at most the panel size, fewer where the metric is undefined (balanced accuracy needs up and down DEGs) and for the split-half ceiling, which also needs >= 5 DEGs in its half. `—` = not scored there.

## Full effect

Shared panel: **191** perturbations (filtered own 193, all own 194).

| method | filtered (primary) n | filtered (primary) median [95% CI] | Δ [95% CI] | p raw / Holm | all n | all median [95% CI] | Δ [95% CI] | p raw / Holm |
|---|---:|---|---|---|---:|---|---|---|
| **scLDM_s0** (reference) | 190 | 0.934 [0.913, 0.945] |  |  | 190 | 0.938 [0.915, 0.949] |  |  |
| scLDM_s1 | 190 | 0.926 [0.901, 0.942] | +0.001 [+0.000, +0.005] | 0.0036 / 0.011 | 190 | 0.927 [0.904, 0.946] | +0.001 [+0.000, +0.004] | 0.0044 / 0.013 |
| scLDM_s2 | 190 | 0.934 [0.911, 0.945] | +0.000 [+0.000, +0.003] | 0.25 / 0.25 | 190 | 0.932 [0.910, 0.943] | +0.000 [+0.000, +0.002] | 0.085 / 0.17 |
| knn | 190 | 0.938 [0.916, 0.950] | -0.003 [-0.008, +0.000] | 0.099 / 0.2 | 190 | 0.940 [0.928, 0.955] | -0.005 [-0.009, +0.000] | 0.18 / 0.18 |
| mlp | 190 | 0.892 [0.875, 0.906] | +0.022 [+0.014, +0.029] | 6.3e-07 / 3.2e-06 | 190 | 0.910 [0.885, 0.925] | +0.011 [+0.004, +0.018] | 0.0015 / 0.006 |
| generic | 190 | 0.859 [0.835, 0.875] | +0.078 [+0.064, +0.088] | 1.5e-17 / 1.1e-16 | 190 | 0.855 [0.840, 0.879] | +0.071 [+0.059, +0.082] | 5.2e-16 / 3.7e-15 |
| transfer | 190 | 0.821 [0.810, 0.829] | +0.097 [+0.084, +0.117] | 4.1e-15 / 2.5e-14 | 190 | 0.830 [0.820, 0.839] | +0.078 [+0.066, +0.096] | 1.2e-12 / 7.3e-12 |
| prior | 190 | 0.800 [0.783, 0.817] | +0.135 [+0.126, +0.141] | 4.3e-23 / 3.4e-22 | 190 | 0.805 [0.792, 0.816] | +0.133 [+0.125, +0.140] | 7.9e-23 / 6.3e-22 |
| scLDM_s0 (shuffled) | 190 | 0.704 [0.661, 0.729] | +0.205 [+0.165, +0.225] | 1.2e-26 / 1.2e-25 | 190 | 0.701 [0.655, 0.728] | +0.204 [+0.161, +0.236] | 1.4e-26 / 1.2e-25 |
| scLDM_s1 (shuffled) | 190 | 0.695 [0.658, 0.721] | +0.220 [+0.186, +0.242] | 2.2e-27 / 2.4e-26 | 190 | 0.688 [0.651, 0.723] | +0.227 [+0.179, +0.247] | 3.8e-27 / 4.2e-26 |
| scLDM_s2 (shuffled) | 190 | 0.703 [0.664, 0.734] | +0.218 [+0.171, +0.246] | 2.7e-26 / 2.4e-25 | 190 | 0.693 [0.667, 0.724] | +0.223 [+0.178, +0.242] | 9.6e-27 / 9.6e-26 |
| noise ceiling (split-half) | 154 | 0.966 [0.959, 0.969] | -0.016 [-0.027, -0.002] | 6e-06 / 2.4e-05 | 154 | 0.963 [0.957, 0.968] | -0.008 [-0.024, +0.000] | 0.00013 / 0.00063 |

Own panels: each version's own >= 5-DEG perturbations, as freeze_results.py would report that eval dir alone.

| method | filtered (primary) n | filtered (primary) median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| scLDM_s0 | 191 | 0.933 [0.913, 0.944] | 192 | 0.938 [0.915, 0.949] |
| scLDM_s1 | 191 | 0.926 [0.900, 0.941] | 192 | 0.929 [0.907, 0.946] |
| scLDM_s2 | 191 | 0.934 [0.909, 0.944] | 192 | 0.933 [0.912, 0.944] |
| knn | 191 | 0.936 [0.916, 0.951] | 192 | 0.939 [0.926, 0.955] |
| mlp | 191 | 0.890 [0.875, 0.905] | 192 | 0.910 [0.884, 0.924] |
| generic | 191 | 0.858 [0.833, 0.875] | 192 | 0.854 [0.833, 0.877] |
| transfer | 191 | 0.821 [0.810, 0.830] | 192 | 0.830 [0.820, 0.841] |
| prior | 191 | 0.800 [0.783, 0.818] | 192 | 0.805 [0.793, 0.816] |
| scLDM_s0 (shuffled) | 191 | 0.703 [0.661, 0.725] | 192 | 0.701 [0.655, 0.724] |
| scLDM_s1 (shuffled) | 191 | 0.694 [0.650, 0.721] | 192 | 0.688 [0.650, 0.725] |
| scLDM_s2 (shuffled) | 191 | 0.703 [0.664, 0.732] | 192 | 0.693 [0.667, 0.726] |
| noise ceiling (split-half) | 156 | 0.965 [0.959, 0.969] | 154 | 0.963 [0.957, 0.968] |

- in the filtered panel only (2): GART, SETX
- in the all panel only (3): GFER, RRP15, USP10

## Perturbation-specific component (shared response projected out)

Shared panel: **190** perturbations (filtered own 192, all own 193).

| method | filtered (primary) n | filtered (primary) median [95% CI] | Δ [95% CI] | p raw / Holm | all n | all median [95% CI] | Δ [95% CI] | p raw / Holm |
|---|---:|---|---|---|---:|---|---|---|
| **scLDM_s0** (reference) | 189 | 0.770 [0.730, 0.793] |  |  | 189 | 0.762 [0.743, 0.784] |  |  |
| scLDM_s1 | 189 | 0.755 [0.715, 0.787] | +0.001 [+0.000, +0.008] | 0.087 / 0.17 | 189 | 0.750 [0.729, 0.780] | +0.000 [+0.000, +0.005] | 0.29 / 0.59 |
| scLDM_s2 | 189 | 0.774 [0.746, 0.799] | +0.000 [+0.000, +0.005] | 0.66 / 0.66 | 189 | 0.766 [0.734, 0.790] | -0.002 [-0.008, +0.003] | 0.19 / 0.57 |
| knn | 189 | 0.820 [0.792, 0.833] | -0.031 [-0.044, -0.017] | 1.1e-05 / 4.5e-05 | 189 | 0.808 [0.768, 0.831] | -0.025 [-0.037, -0.008] | 0.0006 / 0.0024 |
| mlp | 189 | 0.731 [0.711, 0.750] | +0.030 [+0.015, +0.038] | 0.00037 / 0.0011 | 189 | 0.744 [0.722, 0.758] | +0.021 [-0.003, +0.032] | 0.47 / 0.59 |
| generic | — | — |  |  | — | — |  |  |
| transfer | 189 | 0.659 [0.646, 0.682] | +0.074 [+0.043, +0.119] | 1.9e-08 / 9.6e-08 | 189 | 0.680 [0.663, 0.692] | +0.067 [+0.044, +0.101] | 5.9e-06 / 2.9e-05 |
| prior | 189 | 0.524 [0.513, 0.543] | +0.230 [+0.202, +0.255] | 5.5e-28 / 5.5e-27 | 189 | 0.515 [0.503, 0.531] | +0.240 [+0.201, +0.262] | 2.3e-27 / 1.8e-26 |
| scLDM_s0 (shuffled) | 189 | 0.482 [0.464, 0.505] | +0.236 [+0.206, +0.278] | 4.1e-26 / 2.9e-25 | 189 | 0.486 [0.462, 0.502] | +0.236 [+0.201, +0.273] | 2.5e-26 / 1.8e-25 |
| scLDM_s1 (shuffled) | 189 | 0.493 [0.467, 0.511] | +0.243 [+0.208, +0.274] | 8e-27 / 7.2e-26 | 189 | 0.487 [0.463, 0.513] | +0.233 [+0.202, +0.284] | 8.9e-28 / 8.9e-27 |
| scLDM_s2 (shuffled) | 189 | 0.489 [0.464, 0.506] | +0.238 [+0.201, +0.271] | 1e-26 / 8.3e-26 | 189 | 0.491 [0.455, 0.505] | +0.246 [+0.197, +0.265] | 1.7e-27 / 1.5e-26 |
| noise ceiling (split-half) | 154 | 0.849 [0.833, 0.874] | -0.086 [-0.107, -0.035] | 3.5e-11 / 2.1e-10 | 153 | 0.833 [0.820, 0.856] | -0.041 [-0.094, -0.019] | 3.3e-08 / 2e-07 |

Own panels: each version's own >= 5-DEG perturbations, as freeze_results.py would report that eval dir alone.

| method | filtered (primary) n | filtered (primary) median [95% CI] | all n | all median [95% CI] |
|---|---:|---|---:|---|
| scLDM_s0 | 190 | 0.767 [0.729, 0.794] | 191 | 0.762 [0.742, 0.783] |
| scLDM_s1 | 190 | 0.754 [0.714, 0.787] | 191 | 0.750 [0.729, 0.780] |
| scLDM_s2 | 190 | 0.772 [0.746, 0.799] | 191 | 0.766 [0.734, 0.790] |
| knn | 190 | 0.821 [0.795, 0.833] | 191 | 0.803 [0.768, 0.831] |
| mlp | 190 | 0.730 [0.711, 0.750] | 191 | 0.745 [0.724, 0.758] |
| generic | — | — | — | — |
| transfer | 190 | 0.659 [0.643, 0.682] | 191 | 0.680 [0.663, 0.692] |
| prior | 190 | 0.524 [0.514, 0.543] | 191 | 0.515 [0.503, 0.531] |
| scLDM_s0 (shuffled) | 190 | 0.481 [0.464, 0.502] | 191 | 0.486 [0.463, 0.508] |
| scLDM_s1 (shuffled) | 190 | 0.491 [0.466, 0.510] | 191 | 0.487 [0.463, 0.512] |
| scLDM_s2 (shuffled) | 190 | 0.491 [0.466, 0.506] | 191 | 0.491 [0.455, 0.502] |
| noise ceiling (split-half) | 156 | 0.848 [0.833, 0.873] | 153 | 0.833 [0.820, 0.856] |

- not scored in filtered: generic
- in the filtered panel only (2): GART, SETX
- not scored in all: generic
- in the all panel only (3): GFER, RRP15, USP10

## filtered own panel vs the frozen STATE-based tables

Same perturbations and truth when A5 passed; a non-zero Δ is a method whose inputs changed (e.g. the MLP, now seeded).

| report | method | frozen n | frozen median | own n | own median | Δ |
|---|---|---:|---:|---:|---:|---:|
| full | scLDM_s0 | 191 | 0.933 | 191 | 0.933 | +0.0000 |
| full | scLDM_s1 | 191 | 0.926 | 191 | 0.926 | +0.0000 |
| full | scLDM_s2 | 191 | 0.934 | 191 | 0.934 | +0.0000 |
| full | knn | 191 | 0.936 | 191 | 0.936 | +0.0000 |
| full | mlp | 191 | 0.892 | 191 | 0.890 | -0.0017 |
| full | generic | 191 | 0.858 | 191 | 0.858 | +0.0000 |
| full | transfer | 191 | 0.821 | 191 | 0.821 | +0.0000 |
| full | prior | 191 | 0.800 | 191 | 0.800 | +0.0000 |
| full | scLDM_s0 (shuffled) | 191 | 0.703 | 191 | 0.703 | +0.0000 |
| full | scLDM_s1 (shuffled) | 191 | 0.694 | 191 | 0.694 | +0.0000 |
| full | scLDM_s2 (shuffled) | 191 | 0.703 | 191 | 0.703 | +0.0000 |
| full | noise ceiling (split-half) | 156 | 0.965 | 156 | 0.965 | +0.0000 |
| specific | scLDM_s0 | 190 | 0.767 | 190 | 0.767 | +0.0000 |
| specific | scLDM_s1 | 190 | 0.754 | 190 | 0.754 | +0.0000 |
| specific | scLDM_s2 | 190 | 0.772 | 190 | 0.772 | +0.0000 |
| specific | knn | 190 | 0.821 | 190 | 0.821 | +0.0000 |
| specific | mlp | 190 | 0.736 | 190 | 0.730 | -0.0065 |
| specific | transfer | 190 | 0.659 | 190 | 0.659 | +0.0000 |
| specific | prior | 190 | 0.524 | 190 | 0.524 | +0.0000 |
| specific | scLDM_s0 (shuffled) | 190 | 0.481 | 190 | 0.481 | +0.0000 |
| specific | scLDM_s1 (shuffled) | 190 | 0.491 | 190 | 0.491 | +0.0000 |
| specific | scLDM_s2 (shuffled) | 190 | 0.491 | 190 | 0.491 | +0.0000 |
| specific | noise ceiling (split-half) | 156 | 0.848 | 156 | 0.848 | +0.0000 |

In one of them only: bilinear (full, frozen only), calibrated (full, frozen only), gbdt (full, frozen only), bilinear (specific, frozen only), calibrated (specific, frozen only), gbdt (specific, frozen only)
