import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from transfer_models import calibrated_transfer, fill_missing_contexts, knn_transfer

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


def test_calibration_learns_per_gene_sign_flips_that_plain_transfer_gets_wrong():
    rng = np.random.default_rng(0)
    perts = [f"p{i}" for i in range(300)]
    base = rng.normal(size=(300, 20))
    flip = np.where(np.arange(20) < 10, -1.0, 1.0)  # first 10 genes respond oppositely in the target line
    effects = {
        "a": frame(base + 0.05 * rng.normal(size=base.shape), perts),
        "b": frame(base + 0.05 * rng.normal(size=base.shape), perts),
        "T": frame(base * flip * 2 + 0.05 * rng.normal(size=base.shape), perts),
    }
    train, test = perts[:250], perts[250:]
    pred, info = calibrated_transfer(effects, "T", ["a", "b"], train, test, GENES)
    true = effects["T"].loc[test]
    assert sign_acc(pred, true) > 0.97
    plain = (effects["a"].loc[test] + effects["b"].loc[test]) / 2
    assert 0.4 < sign_acc(plain, true) < 0.6  # half the genes flipped -> plain transfer ~50%
    assert info["n_train"] == 250


def test_calibration_never_uses_test_perturbations_in_target():
    rng = np.random.default_rng(1)
    perts = [f"p{i}" for i in range(100)]
    base = rng.normal(size=(100, 20))
    eff = {"a": frame(base, perts), "T": frame(base, perts)}
    poisoned = {**eff, "T": eff["T"].copy()}
    poisoned["T"].loc[perts[80:]] = 1e6  # target effects of test perturbations must be ignored
    p1, _ = calibrated_transfer(eff, "T", ["a"], perts[:80], perts[80:], GENES)
    p2, _ = calibrated_transfer(poisoned, "T", ["a"], perts[:80], perts[80:], GENES)
    pd.testing.assert_frame_equal(p1, p2)


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
    pred, _ = calibrated_transfer(eff, "T", ["a"], train, ["p8"], GENES, n_folds=2)
    assert pred.empty


def test_too_few_training_perturbations_raise():
    eff = {"a": frame(np.ones((3, 20)), ["p0", "p1", "p2"]), "T": frame(np.ones((3, 20)), ["p0", "p1", "p2"])}
    with pytest.raises(ValueError, match="need at least"):
        calibrated_transfer(eff, "T", ["a"], ["p0", "p1"], ["p2"], GENES)
