"""The log2-ratio target must be stable on absent genes and differ from the log1p difference."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.baselines.target_variants import ceiling_rows, log2_effect, pseudobulk_cp10k
from scipy import sparse

GENES = ["g0", "g1", "g2"]


def make_adata(rows):
    """rows: list of (label, counts-per-gene)."""
    labels = [r[0] for r in rows]
    X = sparse.csr_matrix(np.asarray([r[1] for r in rows], dtype=np.float64))
    obs = pd.DataFrame({"gene": labels}, index=[f"c{i}" for i in range(len(rows))])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES))
    adata.layers["counts"] = X.copy()
    return adata


def test_pseudobulk_cp10k_sums_to_ten_thousand():
    adata = make_adata([("ctrl", [10, 10, 0]), ("ctrl", [10, 10, 0])])
    means = pseudobulk_cp10k(adata, "gene", prior=0.0)
    assert means.loc["ctrl"].sum() == pytest.approx(1e4)


def test_gene_absent_from_both_arms_has_zero_effect():
    adata = make_adata([("ctrl", [10, 10, 0]), ("ctrl", [10, 10, 0]), ("p", [10, 10, 0]), ("p", [10, 10, 0])])
    means = pseudobulk_cp10k(adata, "gene", prior=0.5)
    delta = log2_effect(means, ["p"], "ctrl")
    assert delta.loc["p", "g2"] == 0.0  # not an exploding -inf


def test_gene_absent_from_one_arm_stays_finite():
    adata = make_adata([("ctrl", [10, 10, 1]), ("p", [10, 10, 0])])
    means = pseudobulk_cp10k(adata, "gene", prior=0.5)
    delta = log2_effect(means, ["p"], "ctrl")
    assert np.isfinite(delta.loc["p"]).all()
    assert delta.loc["p", "g2"] < 0  # lost expression reads as down


def test_fold_change_is_log2_of_the_cp10k_ratio():
    # p has 4x the relative frequency of g0 versus ctrl
    adata = make_adata([("ctrl", [10, 30, 0]), ("p", [40, 10, 0])])
    means = pseudobulk_cp10k(adata, "gene", prior=0.0)
    delta = log2_effect(means, ["p"], "ctrl")
    expected = np.log2((40 / 50) / (10 / 40))
    assert delta.loc["p", "g0"] == pytest.approx(expected)


def test_log1p_difference_and_log2_ratio_disagree_on_sparse_genes():
    # g0 goes from 1 to 2 counts while g1 stays fixed, so the share of the library doubles:
    # the log1p difference is ~0.69, the log2 ratio is 1.0.
    adata = make_adata([("ctrl", [1, 100, 0]), ("p", [2, 100, 0])])
    means = pseudobulk_cp10k(adata, "gene", prior=0.0)
    ratio = log2_effect(means, ["p"], "ctrl").loc["p", "g0"]
    expected = np.log2((2 / 102) / (1 / 101))
    log1p_diff = np.log1p(2) - np.log1p(1)
    assert ratio == pytest.approx(expected, abs=1e-6)
    assert abs(ratio - log1p_diff) > 0.2


def test_direction_metrics_accept_the_variant_geometry():
    perts = ["p1", "p2"]
    up = pd.DataFrame([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], index=perts, columns=GENES)
    deg = pd.DataFrame([[True, True, False], [True, True, False]], index=perts, columns=GENES)
    ceiling = ceiling_rows(up, up, deg, perts)
    assert (ceiling["sign_acc"] == 1.0).all()
    assert ceiling["n_deg"].eq(2).all()


def test_ceiling_is_chance_when_prediction_is_anti_correlated():
    perts = ["p1"]
    a = pd.DataFrame([[1.0, 1.0, 0.0]], index=perts, columns=GENES)
    b = pd.DataFrame([[-1.0, -1.0, 0.0]], index=perts, columns=GENES)
    deg = pd.DataFrame([[True, True, False]], index=perts, columns=GENES)
    assert ceiling_rows(a, b, deg, perts)["sign_acc"].iloc[0] == 0.0
