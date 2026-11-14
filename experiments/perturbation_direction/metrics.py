"""Direction-of-effect metrics for perturbation predictions.

All functions are pure and operate on per-gene effect vectors ``delta`` (perturbed minus control,
in log1p(CP10k) space). A gene is "up" if ``delta > 0`` and "down" if ``delta < 0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

TOP_K = (10, 20, 50)
# per-knockdown gene counts of the changed-vs-unchanged report (never summarised as metrics)
DETECTION_COUNTS = ("n_deg", "n_unchanged", "n_grey")


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


# ---------------------------------------------------------------------------
# changed vs unchanged genes: does the predicted effect SIZE separate them?
# ---------------------------------------------------------------------------
def unchanged_mask(delta_true, padj, deg_mask, max_effect: float = 0.05, min_padj: float = 0.5):
    """Genes that clearly do not change: not a DEG, ``|delta_true| < max_effect`` and ``padj >= min_padj``.

    A gene with a NaN ``padj`` (or a NaN effect) is never unchanged: without a test there is no evidence
    either way. Genes that are neither a DEG nor unchanged form the grey zone (not significant, but not
    clearly unaffected either).

    Parameters
    ----------
    delta_true, padj, deg_mask
        Arrays of one shape, or perturbation x gene frames; frames are aligned to ``delta_true`` (a
        perturbation or gene absent from ``padj`` is not unchanged, one absent from ``deg_mask`` is not a DEG).
    max_effect
        Strict upper bound on ``|delta_true|``.
    min_padj
        Inclusive lower bound on the BH-adjusted p-value.

    Returns
    -------
    Boolean mask, a frame shaped like ``delta_true`` if that is a frame, else an array.
    """
    if isinstance(delta_true, pd.DataFrame):
        padj = padj.reindex(index=delta_true.index, columns=delta_true.columns)
        deg_mask = deg_mask.reindex(index=delta_true.index, columns=delta_true.columns, fill_value=False)
    d = np.asarray(delta_true, dtype=np.float64)
    q = np.asarray(padj, dtype=np.float64)
    deg = np.asarray(deg_mask).astype(bool)
    if not (d.shape == q.shape == deg.shape):
        raise ValueError("delta_true, padj and deg_mask must have the same shape")
    mask = ~deg & (np.abs(d) < max_effect) & (q >= min_padj)  # comparisons with NaN are False
    if isinstance(delta_true, pd.DataFrame):
        return pd.DataFrame(mask, index=delta_true.index, columns=delta_true.columns)
    return mask


def _rank_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC of scores ``pos`` (positives) against ``neg``: P(pos > neg) + 0.5 P(pos == neg).

    The Mann-Whitney form with average ranks, equal to sklearn's ``roc_auc_score``. Tied scores get half
    credit, so a constant score gives exactly 0.5; NaN when either class is empty.
    """
    if not len(pos) or not len(neg):
        return np.nan
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detection_metrics(
    delta_pred: np.ndarray,
    delta_true: np.ndarray,
    deg_mask: np.ndarray,
    unchanged: np.ndarray,
    tau: float = 0.1,
    scored: np.ndarray | None = None,
) -> dict[str, float]:
    """Does the predicted effect size ``|delta_pred|`` separate changed genes from clearly unchanged ones?

    Classes, among the ``scored`` genes: **changed** = ``deg_mask`` with a non-zero true effect (the set
    :func:`direction_metrics` scores), **unchanged** = ``unchanged`` and not a DEG (see :func:`unchanged_mask`),
    **grey zone** = everything else. ``n_deg``, ``n_unchanged`` and ``n_grey`` count them; they are properties
    of the truth. A gene whose prediction is NaN has no score and is left out of every metric (not the counts).

    Parameters
    ----------
    delta_pred, delta_true
        Predicted and observed per-gene effect.
    deg_mask, unchanged
        Boolean masks of the changed and of the clearly unchanged genes.
    tau
        Call threshold of the secondary metrics: a gene is called changed if ``|delta_pred| > tau`` (strict;
        ``|delta_pred| == tau`` is not a call). The truth's own effect threshold (``min_lfc``) is the natural choice.
    scored
        Genes considered at all (default: all); e.g. False for the CRISPRi target gene.

    Returns
    -------
    Dict of metrics, NaN where undefined:

    ``detect_auroc``
        AUROC of ``|delta_pred|`` for changed (positive) vs unchanged (negative) genes; the grey zone is left
        out. Ties get half credit, so a constant prediction scores 0.5. NaN if either class is empty.
    ``detect_auroc_all``
        The same for changed vs every other scored gene (unchanged + grey zone).
    ``abs_pred_deg_median``, ``abs_pred_unchanged_median``
        Median ``|delta_pred|`` on each class.
    ``unchanged_call_rate``, ``deg_call_rate``
        Share of unchanged / changed genes called changed (``|delta_pred| > tau``).
    ``deg_hit_rate``
        Share of changed genes called changed with the correct sign.
    ``three_class_balanced_acc``
        Mean of three recalls: true-up DEGs predicted ``> tau``, true-down DEGs predicted ``< -tau``,
        unchanged genes predicted ``|.| <= tau``. NaN unless all three classes are non-empty.
    """
    delta_pred = np.asarray(delta_pred, dtype=np.float64)
    delta_true = np.asarray(delta_true, dtype=np.float64)
    deg_mask = np.asarray(deg_mask, dtype=bool)
    unchanged = np.asarray(unchanged, dtype=bool)
    scored = np.ones(delta_pred.shape, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    if not (delta_pred.shape == delta_true.shape == deg_mask.shape == unchanged.shape == scored.shape):
        raise ValueError("delta_pred, delta_true, deg_mask, unchanged and scored must have the same shape")

    changed = deg_mask & scored & (delta_true != 0)
    unch = unchanged & scored & ~deg_mask
    grey = scored & ~changed & ~unch
    out: dict[str, float] = {
        "n_deg": int(changed.sum()),
        "n_unchanged": int(unch.sum()),
        "n_grey": int(grey.sum()),
    }

    have = np.isfinite(delta_pred)  # a gene without a prediction has no score
    changed, unch, grey = changed & have, unch & have, grey & have
    score = np.abs(delta_pred)
    called = score > tau
    out["detect_auroc"] = _rank_auroc(score[changed], score[unch])
    out["detect_auroc_all"] = _rank_auroc(score[changed], score[unch | grey])
    out["abs_pred_deg_median"] = float(np.median(score[changed])) if changed.any() else np.nan
    out["abs_pred_unchanged_median"] = float(np.median(score[unch])) if unch.any() else np.nan
    out["unchanged_call_rate"] = float(called[unch].mean()) if unch.any() else np.nan
    out["deg_call_rate"] = float(called[changed].mean()) if changed.any() else np.nan
    hit = called & (np.sign(delta_pred) == np.sign(delta_true))
    out["deg_hit_rate"] = float(hit[changed].mean()) if changed.any() else np.nan
    up, down = changed & (delta_true > 0), changed & (delta_true < 0)
    if up.any() and down.any() and unch.any():
        recalls = [(delta_pred[up] > tau).mean(), (delta_pred[down] < -tau).mean(), (~called[unch]).mean()]
        out["three_class_balanced_acc"] = float(np.mean(recalls))
    else:
        out["three_class_balanced_acc"] = np.nan
    return out


def evaluate_detection(
    delta_pred: pd.DataFrame,
    delta_true: pd.DataFrame,
    deg_mask: pd.DataFrame,
    unchanged: pd.DataFrame,
    exclude_genes: dict[str, list[str]] | None = None,
    tau: float = 0.1,
) -> pd.DataFrame:
    """Per-perturbation :func:`detection_metrics`.

    All frames are indexed by perturbation (rows) and gene (columns). Only perturbations and genes present
    in all four frames are scored. ``exclude_genes[p]`` lists genes left out for perturbation ``p`` (e.g. the
    CRISPRi target itself), exactly as in :func:`evaluate`.
    """
    frames = (delta_pred, delta_true, deg_mask, unchanged)
    perts, genes = frames[0].index, frames[0].columns
    for f in frames[1:]:
        perts, genes = perts.intersection(f.index), genes.intersection(f.columns)
    rows = []
    for p in perts:
        scored = ~genes.isin(exclude_genes[p]) if exclude_genes and p in exclude_genes else None
        m = detection_metrics(
            delta_pred.loc[p, genes].to_numpy(),
            delta_true.loc[p, genes].to_numpy(),
            deg_mask.loc[p, genes].to_numpy(dtype=bool),
            unchanged.loc[p, genes].to_numpy(dtype=bool),
            tau=tau,
            scored=scored,
        )
        rows.append({"perturbation": p, **m})
    columns = ["perturbation", *DETECTION_COUNTS]
    return (pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)).set_index("perturbation")


def summarize_detection(per_pert: pd.DataFrame, min_degs: int = 5, min_unchanged: int = 20) -> pd.Series:
    """Median (and IQR) across perturbations with >= ``min_degs`` DEGs and >= ``min_unchanged`` unchanged genes.

    Same shape as :func:`summarize`: ``n_perturbations``, ``n_perturbations_total`` and ``<metric>_median`` /
    ``<metric>_iqr`` for every column of :func:`detection_metrics` except the gene counts.
    """
    kept = per_pert[(per_pert["n_deg"] >= min_degs) & (per_pert["n_unchanged"] >= min_unchanged)]
    out = {"n_perturbations": len(kept), "n_perturbations_total": len(per_pert)}
    for c in (c for c in kept.columns if c not in DETECTION_COUNTS):
        q1, med, q3 = kept[c].astype(float).quantile([0.25, 0.5, 0.75])
        out[f"{c}_median"] = med
        out[f"{c}_iqr"] = q3 - q1
    return pd.Series(out)
