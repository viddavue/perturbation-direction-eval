import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from metrics import (
    derange,
    direction_metrics,
    evaluate,
    log1p_cp10k,
    shuffled_prediction,
    summarize,
)


def test_perfect_and_inverted_predictions():
    true = np.array([1.0, -2.0, 0.5, -0.1, 3.0])
    mask = np.ones(5, dtype=bool)
    m = direction_metrics(true * 0.3, true, mask, top_k=(2,))
    assert m["sign_acc"] == 1.0 and m["balanced_acc"] == 1.0 and m["auroc"] == 1.0
    assert m["sign_acc_top2"] == 1.0
    m = direction_metrics(-true, true, mask, top_k=(2,))
    assert m["sign_acc"] == 0.0 and m["auroc"] == 0.0


def test_balanced_accuracy_penalizes_constant_prediction():
    # 4 up, 1 down; predicting "always up" gets 80% accuracy but 50% balanced accuracy
    true = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
    m = direction_metrics(np.ones(5), true, np.ones(5, dtype=bool))
    assert m["sign_acc"] == pytest.approx(0.8)
    assert m["balanced_acc"] == pytest.approx(0.5)
    assert m["n_up"] == 4 and m["n_down"] == 1


def test_mask_ties_and_no_calls():
    true = np.array([1.0, -1.0, 0.0, 2.0, -2.0])
    pred = np.array([0.5, 0.0, 9.0, -0.01, -1.0])
    mask = np.array([True, True, True, True, False])
    m = direction_metrics(pred, true, mask, eps=0.05)
    # gene 2 has zero true effect -> dropped; gene 4 not in mask -> dropped
    assert m["n_deg"] == 3
    # genes 1 (pred 0) and 3 (|pred| < eps) are no-calls; only gene 0 is called (correct)
    assert m["no_call_rate"] == pytest.approx(2 / 3)
    assert m["sign_acc"] == 1.0
    # without eps, an exact zero prediction is called and counted wrong
    m0 = direction_metrics(pred, true, mask, eps=0.0)
    assert m0["no_call_rate"] == pytest.approx(1 / 3)
    assert m0["sign_acc"] == pytest.approx(0.5)


def test_top_k_uses_largest_true_effects():
    true = np.array([5.0, -4.0, 0.1, -0.1])
    pred = np.array([1.0, -1.0, -1.0, 1.0])  # right on the two big ones, wrong on the small ones
    m = direction_metrics(pred, true, np.ones(4, dtype=bool), top_k=(2, 10))
    assert m["sign_acc_top2"] == 1.0
    assert np.isnan(m["sign_acc_top10"])  # fewer than k DEGs
    assert m["sign_acc"] == 0.5


def test_empty_deg_set_is_nan():
    m = direction_metrics(np.ones(3), np.ones(3), np.zeros(3, dtype=bool))
    assert m["n_deg"] == 0
    assert np.isnan(m["sign_acc"]) and np.isnan(m["auroc"]) and np.isnan(m["balanced_acc"])


def test_evaluate_excludes_target_gene_and_aligns_frames():
    genes = ["A", "B", "C", "D"]
    true = pd.DataFrame([[-3.0, 1.0, -1.0, 0.5], [1.0, 1.0, -1.0, -1.0]], index=["A", "X"], columns=genes)
    # prediction has columns in a different order and an extra perturbation
    pred = pd.DataFrame(
        [[0.2, -1.0, 1.0, -1.0], [1, 1, 1, 1], [-1.0, -1.0, 1.0, 1.0]], index=["A", "Z", "X"], columns=genes[::-1]
    )
    degs = pd.DataFrame(True, index=true.index, columns=genes)
    res = evaluate(pred, true, degs, exclude_genes={"A": ["A"]})
    assert list(res.index) == ["A", "X"]
    assert res.loc["A", "n_deg"] == 3
    assert res.loc["A", "sign_acc"] == 1.0  # target gene A excluded, rest correct
    assert res.loc["X", "sign_acc"] == 1.0


def test_shuffle_has_no_fixed_points():
    idx = pd.Index([f"p{i}" for i in range(20)])
    perm = derange(idx, seed=1)
    assert sorted(perm) == sorted(idx) and not np.any(perm == idx)
    pred = pd.DataFrame(np.arange(20)[:, None] * np.ones((1, 3)), index=idx)
    sh = shuffled_prediction(pred, seed=1)
    assert list(sh.index) == list(idx) and not np.any(sh.iloc[:, 0].to_numpy() == np.arange(20))


def test_summarize_filters_weak_perturbations():
    per = pd.DataFrame(
        {
            "n_deg": [10, 2, 30],
            "n_up": [5, 1, 15],
            "n_down": [5, 1, 15],
            "no_call_rate": [0.0, 0.0, 0.0],
            "sign_acc": [0.6, 0.0, 0.9],
        },
        index=["a", "b", "c"],
    )
    s = summarize(per, min_degs=5)
    assert s["n_perturbations"] == 2 and s["n_perturbations_total"] == 3
    assert s["sign_acc_median"] == pytest.approx(0.75)
    assert s["sign_acc_pooled"] == pytest.approx((6 + 27) / 40)


def test_log1p_cp10k_dense_and_sparse_agree():
    X = np.array([[1, 0, 3], [0, 0, 0], [2, 2, 0]], dtype=float)
    d = log1p_cp10k(X)
    s = log1p_cp10k(sparse.csr_matrix(X))
    np.testing.assert_allclose(d, s)
    np.testing.assert_allclose(np.expm1(d[0]).sum(), 1e4)
    assert np.all(d[1] == 0)


def test_remove_shared_projects_out_direction_and_strength():
    from metrics import remove_shared

    ref = pd.Series([1.0, 1.0, 0.0, 0.0], index=list("abcd"))
    specific = np.array([0.0, 0.0, 2.0, -1.0])
    delta = pd.DataFrame(
        [3 * ref.to_numpy() + specific, -0.5 * ref.to_numpy(), ref.to_numpy() + [1.0, -1.0, 0.0, 0.0]],
        index=["strong", "opposite", "tilted"],
        columns=list("abcd"),
    )
    resid, beta = remove_shared(delta, ref)
    np.testing.assert_allclose(beta, [3.0, -0.5, 1.0])
    np.testing.assert_allclose(resid.loc["strong"], specific, atol=1e-12)  # strength removed, not just the mean
    np.testing.assert_allclose(resid.loc["opposite"], 0, atol=1e-12)
    np.testing.assert_allclose(resid.loc["tilted"], [1.0, -1.0, 0.0, 0.0], atol=1e-12)  # orthogonal part kept
    np.testing.assert_allclose(resid.to_numpy() @ ref.to_numpy(), 0, atol=1e-12)
