"""Real halves must be disjoint, including controls, and drop groups that are too small."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from make_real_halves import split_halves


def adata(groups):
    n = sum(groups.values())
    labels = [g for g, c in groups.items() for _ in range(c)]
    return ad.AnnData(sparse.csr_matrix(np.zeros((n, 2))),
                      obs={"gene": labels}, var={"gene": ["a", "b"]})


def test_halves_are_disjoint_and_balanced():
    a, b, info = split_halves(adata({"c": 20, "p": 10}), "gene", seed=0)
    assert not set(a.obs_names) & set(b.obs_names)
    assert a.obs.gene.value_counts().to_dict() == {"c": 10, "p": 5}
    assert b.obs.gene.value_counts().to_dict() == {"c": 10, "p": 5}
    assert all(not v["dropped"] for v in info.values())


def test_small_groups_are_dropped_not_half_used():
    a, b, info = split_halves(adata({"c": 20, "p": 4}), "gene", seed=0)
    assert info["p"]["dropped"] is True
    assert "p" not in set(a.obs.gene) | set(b.obs.gene)


def test_split_is_deterministic():
    x = adata({"c": 20, "p": 12})
    a1, b1, _ = split_halves(x, "gene", seed=3)
    a2, b2, _ = split_halves(x, "gene", seed=3)
    assert list(a1.obs_names) == list(a2.obs_names)
    assert list(b1.obs_names) == list(b2.obs_names)


def test_nothing_splittable_is_an_error():
    with pytest.raises(ValueError, match="large enough"):
        split_halves(adata({"p": 3}), "gene", seed=0)
