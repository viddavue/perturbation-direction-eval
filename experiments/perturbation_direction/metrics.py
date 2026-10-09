"""Direction-of-effect metrics for perturbation predictions.

All functions are pure and operate on per-gene effect vectors ``delta`` (perturbed minus control,
in log1p(CP10k) space). A gene is "up" if ``delta > 0`` and "down" if ``delta < 0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

TOP_K = (10, 20, 50)


def log1p_cp10k(X) -> np.ndarray:
    """Library-size normalize counts to 10k and log1p. Accepts dense or sparse, returns dense float64."""
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X = X.astype(np.float64)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * 1e4)


def benjamini_hochberg(p) -> np.ndarray:
    """BH-adjusted p-values for a 1-D array (NaNs stay NaN)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    if q.size:
        order = np.argsort(q)
        ranked = q[order] * q.size / np.arange(1, q.size + 1)
        adj = np.empty_like(q)
        adj[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
        out[ok] = adj
    return out


def mean_delta(X_pert: np.ndarray, X_ctrl: np.ndarray) -> np.ndarray:
    """Pseudobulk effect: per-gene mean of perturbed cells minus mean of control cells."""
    return np.asarray(X_pert).mean(axis=0) - np.asarray(X_ctrl).mean(axis=0)


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(fn(a, b)[0])


def direction_metrics(
    delta_pred: np.ndarray,
    delta_true: np.ndarray,
    deg_mask: np.ndarray,
    eps: float = 0.0,
    top_k: tuple[int, ...] = TOP_K,
) -> dict[str, float]:
    """Score predicted effect directions against the true directions on a set of DEGs.

    Args:
        delta_pred: predicted per-gene effect.
        delta_true: observed per-gene effect.
        deg_mask: boolean mask of genes considered truly differentially expressed (the scored set).
        eps: predictions with ``|delta_pred| <= eps`` are "no call" and excluded from sign accuracy
            (their rate is reported as ``no_call_rate``).
        top_k: also report sign accuracy on the k DEGs with largest ``|delta_true|``.

    Returns
    -------
        Dict of metrics; NaN where a metric is undefined (e.g. no DEGs, single class for AUROC).
    """
    delta_pred = np.asarray(delta_pred, dtype=np.float64)
    delta_true = np.asarray(delta_true, dtype=np.float64)
    deg_mask = np.asarray(deg_mask, dtype=bool)
    if not (delta_pred.shape == delta_true.shape == deg_mask.shape):
        raise ValueError("delta_pred, delta_true and deg_mask must have the same shape")

    # genes with a zero true effect have no direction and cannot be scored
    deg_mask = deg_mask & (delta_true != 0)
    p, t = delta_pred[deg_mask], delta_true[deg_mask]
    true_up = t > 0
    called = np.abs(p) > eps
    correct = np.sign(p) == np.sign(t)

    out: dict[str, float] = {
        "n_deg": int(deg_mask.sum()),
        "n_up": int(true_up.sum()),
        "n_down": int((~true_up).sum()),
        "no_call_rate": float(1 - called.mean()) if len(p) else np.nan,
    }
    out["sign_acc"] = float(correct[called].mean()) if called.any() else np.nan

    # balanced accuracy: mean of recall on up-DEGs and on down-DEGs (among called genes)
    recalls = [correct[called & m].mean() for m in (true_up, ~true_up) if (called & m).any()]
    out["balanced_acc"] = float(np.mean(recalls)) if len(recalls) == 2 else np.nan

    order = np.argsort(-np.abs(t), kind="stable")
    for k in top_k:
        idx = order[:k]
        idx = idx[called[idx]]
        out[f"sign_acc_top{k}"] = float(correct[idx].mean()) if len(order) >= k and len(idx) else np.nan

    out["auroc"] = float(roc_auc_score(true_up, p)) if 0 < true_up.sum() < len(t) else np.nan
    out["pearson_deg"] = _safe_corr(pearsonr, p, t)
    out["spearman_deg"] = _safe_corr(spearmanr, p, t)
    out["pearson_all"] = _safe_corr(pearsonr, delta_pred, delta_true)
    out["spearman_all"] = _safe_corr(spearmanr, delta_pred, delta_true)
    return out


def evaluate(
    delta_pred: pd.DataFrame,
    delta_true: pd.DataFrame,
    deg_mask: pd.DataFrame,
    exclude_genes: dict[str, list[str]] | None = None,
    eps: float = 0.0,
) -> pd.DataFrame:
    """Per-perturbation direction metrics.

    All frames are indexed by perturbation (rows) and gene (columns). Only perturbations and genes
    present in all three frames are scored. ``exclude_genes[p]`` lists genes dropped for perturbation
    ``p`` (e.g. the CRISPRi target itself).
    """
    perts = delta_pred.index.intersection(delta_true.index).intersection(deg_mask.index)
    genes = delta_pred.columns.intersection(delta_true.columns).intersection(deg_mask.columns)
    rows = []
    for p in perts:
        mask = deg_mask.loc[p, genes].to_numpy(dtype=bool).copy()
        if exclude_genes and p in exclude_genes:
            mask &= ~genes.isin(exclude_genes[p])
        m = direction_metrics(delta_pred.loc[p, genes].to_numpy(), delta_true.loc[p, genes].to_numpy(), mask, eps=eps)
        rows.append({"perturbation": p, **m})
    return pd.DataFrame(rows).set_index("perturbation")


def remove_shared(delta: pd.DataFrame, ref: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Project a shared response vector out of every perturbation's effect.

    Each row d becomes d - beta * ref with beta = <d, ref> / <ref, ref> (least squares, per perturbation),
    so both the direction and the strength of the shared response are removed. Returns (residual, beta).
    """
    r = ref.reindex(delta.columns).to_numpy(dtype=np.float64)
    d = delta.to_numpy(dtype=np.float64)
    beta = np.nan_to_num(d) @ r / (r @ r)
    resid = pd.DataFrame(d - np.outer(beta, r), index=delta.index, columns=delta.columns)
    return resid, pd.Series(beta, index=delta.index)


def derange(index: pd.Index, seed: int = 0) -> pd.Index:
    """Random permutation of ``index`` with no fixed points (each perturbation gets another's prediction)."""
    if len(index) < 2:
        raise ValueError("need at least 2 perturbations to derange")
    rng = np.random.default_rng(seed)
    n = len(index)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return index[perm]


def shuffled_prediction(delta_pred: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Specificity control: assign each perturbation the prediction made for a different perturbation."""
    out = delta_pred.loc[derange(delta_pred.index, seed)].copy()
    out.index = delta_pred.index
    return out


def summarize(per_pert: pd.DataFrame, min_degs: int = 5) -> pd.Series:
    """Median (and IQR) across perturbations with at least ``min_degs`` scored DEGs."""
    kept = per_pert[per_pert["n_deg"] >= min_degs]
    metric_cols = [c for c in kept.columns if c not in ("n_deg", "n_up", "n_down")]
    out = {"n_perturbations": len(kept), "n_perturbations_total": len(per_pert)}
    for c in metric_cols:
        q1, med, q3 = kept[c].quantile([0.25, 0.5, 0.75])
        out[f"{c}_median"] = med
        out[f"{c}_iqr"] = q3 - q1
    # pooled sign accuracy: every scored DEG across perturbations weighted equally
    called_correct = (kept["sign_acc"] * kept["n_deg"] * (1 - kept["no_call_rate"])).sum()
    called_total = (kept["n_deg"] * (1 - kept["no_call_rate"])).sum()
    out["sign_acc_pooled"] = called_correct / called_total if called_total else np.nan
    return pd.Series(out)
