"""Learned cross-context baseline for perturbation effects: kNN transfer.

It predicts a perturbation's effect in a target context from its effects in other contexts, fitted on
perturbations that were observed in the target context during training.

Inputs are effect matrices (perturbation x gene, log1p(CP10k) differences vs control), one per context.
Only ``train_perts`` (target-context training perturbations) are used for fitting; ``test_perts`` are
only predicted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def stack_others(effects: dict[str, pd.DataFrame], others: list[str], perts, genes) -> np.ndarray:
    """(n_perts, n_genes, n_others) array of effects in the other contexts; NaN where unobserved."""
    return np.stack([effects[c].reindex(index=perts, columns=genes).to_numpy() for c in others], axis=-1)


def fill_missing_contexts(X: np.ndarray) -> np.ndarray:
    """Replace a context's missing effect by the mean over the contexts where the perturbation was seen.

    Rows unseen in all contexts stay NaN.
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN mean, intended
        mean = np.nanmean(X, axis=-1, keepdims=True)
    return np.where(np.isnan(X), mean, X)


def _kfold(n: int, k: int, seed: int):
    if n < 2 * k:
        raise ValueError(f"need at least {2 * k} target-context training perturbations for {k}-fold CV, got {n}")
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, min(k, n))


def _corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of A and every row of B."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


def _knn_predict(S: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """Similarity-weighted mean of the k most similar rows of Y (negative similarities get weight 0)."""
    k = min(k, S.shape[1])
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(S, nn, axis=1), 0, None)
    w_sum = w.sum(1, keepdims=True)
    w = np.where(w_sum > 0, w / np.where(w_sum > 0, w_sum, 1), 1.0 / k)  # all-negative -> plain mean
    return np.einsum("nk,nkg->ng", w, Y[nn])


def knn_transfer(
    effects: dict[str, pd.DataFrame],
    target: str,
    others: list[str],
    train_perts: list[str],
    test_perts: list[str],
    genes: list[str],
    ks=(5, 10, 20, 50),
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Nearest target-context training perturbations by correlation of their other-context effects."""
    emb_tr = fill_missing_contexts(stack_others(effects, others, train_perts, genes)).mean(-1)
    keep = ~np.isnan(emb_tr).all(axis=1)
    emb_tr = np.nan_to_num(emb_tr[keep])
    Y_tr = effects[target].reindex(index=train_perts, columns=genes).to_numpy()[keep]
    folds = _kfold(len(emb_tr), n_folds, seed)
    cv = {}
    for k in ks:
        err = 0.0
        for f in folds:
            m = np.ones(len(emb_tr), dtype=bool)
            m[f] = False
            err += float(((_knn_predict(_corr_rows(emb_tr[f], emb_tr[m]), Y_tr[m], k) - Y_tr[f]) ** 2).sum())
        cv[k] = err / Y_tr.size
    best = min(cv, key=cv.get)

    emb_te = fill_missing_contexts(stack_others(effects, others, test_perts, genes)).mean(-1)
    seen = ~np.isnan(emb_te).all(axis=1)
    pred = np.full((len(test_perts), len(genes)), np.nan)
    pred[seen] = _knn_predict(_corr_rows(np.nan_to_num(emb_te[seen]), emb_tr), Y_tr, best)
    out = pd.DataFrame(pred, index=test_perts, columns=genes).dropna(how="all")
    return out, {"k": best, "cv_mse": cv, "n_train": int(len(emb_tr))}
