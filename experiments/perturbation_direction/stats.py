"""Uncertainty and paired tests over per-perturbation scores.

The evaluation reports a median across perturbations, which is a point estimate. This module adds
what a reviewer will ask for: a confidence interval for each method, and paired tests for every
method comparison on the perturbations they both score.

Conventions
-----------
* The statistic is the **median** across perturbations, matching `evaluate.py`'s summary.
* CIs are **percentile bootstrap** intervals over perturbations, not over genes or cells. They
  describe variability across the perturbation panel and are conditional on that panel.
* Method comparisons are **paired**: only perturbations scored by both methods enter, and the
  interval is a bootstrap of the paired difference.
* P-values are **Wilcoxon signed-rank** (two-sided). They are Holm-corrected within one declared
  family of comparisons; corrected values are reported separately so the raw ones stay auditable.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEFAULT_N_BOOT = 2000


def usable(per_pert: pd.DataFrame, metric: str = "balanced_acc", min_degs: int = 5) -> pd.Series:
    """Per-perturbation values entering the summary: enough scored DEGs, metric defined."""
    if metric not in per_pert:
        raise KeyError(f"metric {metric!r} not in {list(per_pert.columns)}")
    keep = per_pert["n_deg"] >= min_degs
    return per_pert.loc[keep, metric].dropna()


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic=np.median,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for one statistic over perturbation values."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    draws = statistic(values[idx], axis=1)
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def summarize(
    per_pert: dict[str, pd.DataFrame],
    *,
    metric: str = "balanced_acc",
    min_degs: int = 5,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """Median, bootstrap CI and n for every method."""
    rows = {}
    for name, frame in per_pert.items():
        values = usable(frame, metric, min_degs)
        lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
        rows[name] = {
            "n_perturbations": int(values.size),
            "median": float(np.median(values)) if values.size else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
        }
    return pd.DataFrame(rows).T.sort_values("median", ascending=False)


def paired_difference(
    a: pd.Series,
    b: pd.Series,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI for median(a - b) plus a two-sided Wilcoxon signed-rank test on shared rows."""
    common = a.index.intersection(b.index)
    left, right = a.loc[common].astype(float), b.loc[common].astype(float)
    diff = (left - right).to_numpy()
    if diff.size == 0:
        return {"n": 0, "median_difference": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_raw": float("nan")}
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, alternative="two-sided").pvalue)
    return {"n": int(diff.size), "median_difference": float(np.median(diff)),
            "ci_low": lo, "ci_high": hi, "p_raw": p}


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjustment, returned in the original order."""
    p = pvalues.astype(float)
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order]
    m = len(ranked)
    adjusted = np.empty(m)
    running = 0.0
    for i, value in enumerate(ranked):
        running = max(running, min(1.0, value * (m - i)))
        adjusted[i] = running
    out = np.empty(m)
    out[order] = adjusted
    return pd.Series(out, index=p.index)


def pairwise_tests(
    per_pert: dict[str, pd.DataFrame],
    *,
    metric: str = "balanced_acc",
    min_degs: int = 5,
    reference: str | None = "scLDM_w1",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired comparisons; ``family`` marks the comparisons Holm-corrected together.

    With a reference, the family is ``reference vs each other method`` -- one question: is our
    model different from each comparator? Without one, every pair is in the family.
    """
    series = {name: usable(frame, metric, min_degs) for name, frame in per_pert.items()}
    if reference is not None:
        pairs = [(reference, other) for other in series if other != reference]
    else:
        pairs = list(itertools.combinations(series, 2))
    rows = []
    for a, b in pairs:
        stats = paired_difference(series[a], series[b], n_boot=n_boot, seed=seed)
        rows.append({"left": a, "right": b, **stats})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = holm(frame["p_raw"]).to_numpy()
    frame["family"] = f"{reference} vs others" if reference is not None else "all pairs"
    return frame.sort_values("median_difference", ascending=False).reset_index(drop=True)
