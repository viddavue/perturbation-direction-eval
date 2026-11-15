"""The validation split must be disjoint, complete, stratified, and reproducible."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.data.make_val_split import build_split, quota


def deg_from_counts(counts):
    """One row per perturbation with exactly ``count`` DEGs set, so deg.sum(1) == counts."""
    width = max(counts)
    arr = np.zeros((len(counts), width), dtype=bool)
    for i, c in enumerate(counts):
        arr[i, :c] = True
    return pd.DataFrame(arr, index=[f"p{i}" for i in range(len(counts))])


def test_quota_reserves_a_member_for_every_nonempty_stratum():
    picks = quota(6, [40, 2, 30, 10, 0])
    assert sum(picks) == 6
    assert picks[1] >= 1 and picks[4] == 0


def test_quota_rejects_impossible_requests():
    with pytest.raises(ValueError, match="every stratum"):
        quota(2, [50, 20, 20, 10, 5])


def test_split_is_disjoint_complete_and_reproducible():
    # spread perturbations across all five strength bins
    sizes = [6, 10, 12, 8, 4]
    counts = [0, 5, 20, 100, 500]
    rows = []
    for size, c in zip(sizes, counts, strict=True):
        rows.extend([c] * size)
    deg = deg_from_counts(rows)
    val1, test1, _ = build_split(deg, 12, seed=3)
    val2, test2, _ = build_split(deg, 12, seed=3)
    assert (val1, test1) == (val2, test2)
    assert not set(val1) & set(test1)
    assert set(val1) | set(test1) == set(deg.index)
    assert len(val1) == 12


def test_split_covers_weak_and_strong_perturbations():
    rows = [0] * 6 + [8] * 10 + [50] * 12 + [200] * 8 + [900] * 4
    deg = deg_from_counts(rows)
    val, _, _ = build_split(deg, 10, seed=1)
    counts = deg.sum(1)
    assert counts[val].min() < 20 and counts[val].max() > 100


def test_too_large_validation_set_is_rejected():
    deg = pd.DataFrame(np.ones((5, 2), dtype=bool), index=list("abcde"))
    with pytest.raises(ValueError):
        build_split(deg, 5, seed=0)
