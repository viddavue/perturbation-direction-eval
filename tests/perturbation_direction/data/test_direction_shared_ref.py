"""The shared response must be the mean effect of training conditions, control-relative."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.data.make_shared_reference import mean_effect
from scipy import sparse


def cells(groups, lib=1000):
    """groups: label -> per-gene counts for every cell of that group."""
    labels, rows = [], []
    for label, counts in groups.items():
        labels.append(label)
        rows.append(counts)
    adata = ad.AnnData(
        sparse.csr_matrix(np.asarray(rows, dtype=float)),
        obs={"cytokine": labels},
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    return adata


def lognorm(adata):
    """Same transform ground_truth.lognorm applies; must set layers AFTER copying."""
    out = adata.copy()
    out.layers["counts"] = out.X.copy()
    lib = np.asarray(out.X.sum(1)).ravel()
    normalized = out.X.multiply((1e4 / lib)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    out.X = normalized
    return out


def test_effect_is_relative_to_the_control():
    a = lognorm(cells({"PBS": [10, 10, 10], "IL-2": [40, 5, 5]}))
    eff = mean_effect(a, "cytokine", ["IL-2"], "PBS")
    assert eff.loc["IL-2", "a"] > 0 and eff.loc["IL-2", "b"] < 0


def test_identical_conditions_give_zero_reference():
    base = [10, 20, 30]
    a = lognorm(cells({"PBS": base, "T1": base, "T2": base}))
    eff = mean_effect(a, "cytokine", ["T1", "T2"], "PBS")
    np.testing.assert_allclose(eff.to_numpy(), 0, atol=1e-12)


def test_missing_condition_is_an_error():
    a = lognorm(cells({"PBS": [10, 10, 10], "T1": [20, 10, 5]}))
    with pytest.raises(KeyError, match="absent"):
        mean_effect(a, "cytokine", ["T1", "NOPE"], "PBS")


def test_missing_control_is_an_error():
    a = lognorm(cells({"T1": [20, 10, 5]}))
    with pytest.raises(KeyError, match="control"):
        mean_effect(a, "cytokine", ["T1"], "PBS")


def test_reference_is_the_mean_across_conditions():
    a = lognorm(cells({"PBS": [10, 10, 10], "T1": [40, 5, 5], "T2": [5, 40, 5]}))
    eff = mean_effect(a, "cytokine", ["T1", "T2"], "PBS")
    np.testing.assert_allclose(eff.mean(axis=0).to_numpy(), eff.to_numpy().mean(axis=0))
