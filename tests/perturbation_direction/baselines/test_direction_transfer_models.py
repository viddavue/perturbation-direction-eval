import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.baselines.transfer_models import fill_missing_contexts, knn_transfer

GENES = [f"g{i}" for i in range(20)]


def frame(a, perts):
    return pd.DataFrame(a, index=perts, columns=GENES)


def sign_acc(pred: pd.DataFrame, true: pd.DataFrame) -> float:
    p, t = pred.loc[true.index].to_numpy(), true.to_numpy()
    return float((np.sign(p) == np.sign(t)).mean())


def test_fill_missing_contexts_uses_mean_of_observed():
    X = np.array([[[1.0, np.nan, 3.0]], [[np.nan, np.nan, np.nan]]])
    out = fill_missing_contexts(X)
    np.testing.assert_allclose(out[0, 0], [1.0, 2.0, 3.0])
    assert np.isnan(out[1]).all()


def test_knn_recovers_cluster_specific_target_response():
    rng = np.random.default_rng(2)
    n_clusters, per = 6, 40
    perts = [f"p{i}" for i in range(n_clusters * per)]
    cluster = np.repeat(np.arange(n_clusters), per)
    other_sig = rng.normal(size=(n_clusters, 20))  # how a cluster looks in other lines
    target_sig = rng.normal(size=(n_clusters, 20))  # unrelated, cluster-specific response in the target line
    noise = lambda: 0.2 * rng.normal(size=(len(perts), 20))
    effects = {"a": frame(other_sig[cluster] + noise(), perts), "T": frame(target_sig[cluster] + noise(), perts)}
    test_idx = np.arange(len(perts)) % per < 5
    train = [p for p, t in zip(perts, test_idx, strict=True) if not t]
    test = [p for p, t in zip(perts, test_idx, strict=True) if t]
    pred, info = knn_transfer(effects, "T", ["a"], train, test, GENES)
    true = frame(target_sig[cluster[test_idx]], test)
    assert sign_acc(pred, true) > 0.95
    assert info["k"] in (5, 10, 20)  # 35 training perturbations per cluster; 50 would mix clusters


def test_unseen_perturbations_get_no_prediction():
    rng = np.random.default_rng(3)
    perts = [f"p{i}" for i in range(9)]
    vals = rng.normal(size=(9, 20))
    eff = {"a": frame(vals, perts).drop("p8"), "T": frame(vals, perts)}
    train = perts[:8]
    pred, _ = knn_transfer(eff, "T", ["a"], train, ["p8"], GENES, ks=(1,), n_folds=2)
    assert pred.empty


def test_too_few_training_perturbations_raise():
    eff = {"a": frame(np.ones((3, 20)), ["p0", "p1", "p2"]), "T": frame(np.ones((3, 20)), ["p0", "p1", "p2"])}
    with pytest.raises(ValueError, match="need at least"):
        knn_transfer(eff, "T", ["a"], ["p0", "p1"], ["p2"], GENES)
