"""Uncertainty and paired tests must be paired, deterministic, and corrected within a family."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from stats import bootstrap_ci, holm, paired_difference, pairwise_tests, summarize, usable


def frame(values, n_deg=20):
    return pd.DataFrame({"balanced_acc": values, "n_deg": n_deg},
                        index=[f"p{i}" for i in range(len(values))])


def test_usable_filters_on_degs_and_missing_values():
    f = pd.DataFrame({"balanced_acc": [0.9, np.nan, 0.8], "n_deg": [20, 20, 3]},
                     index=["a", "b", "c"])
    assert list(usable(f).index) == ["a"]


def test_bootstrap_ci_brackets_median_and_is_deterministic():
    values = np.linspace(0.5, 1.0, 30)
    lo1, hi1 = bootstrap_ci(values, seed=7)
    lo2, hi2 = bootstrap_ci(values, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= np.median(values) <= hi1


def test_bootstrap_of_constant_data_is_degenerate():
    lo, hi = bootstrap_ci(np.full(10, 0.5))
    assert lo == hi == 0.5


def test_paired_difference_uses_intersection_only():
    a = pd.Series([0.9, 0.8, 0.7], index=["x", "y", "z"])
    b = pd.Series([0.5, 0.5], index=["x", "y"])
    result = paired_difference(a, b, seed=0)
    assert result["n"] == 2
    assert result["median_difference"] == pytest.approx(0.35)


def test_a_consistent_shift_is_detected_and_a_null_is_not():
    rng = np.random.default_rng(0)
    base = rng.normal(0.8, 0.05, 40)
    shifted = pd.Series(base + 0.1, index=[f"p{i}" for i in range(40)])
    same = pd.Series(base, index=[f"p{i}" for i in range(40)])
    assert paired_difference(shifted, same, seed=0)["p_raw"] < 1e-3
    assert paired_difference(same, same.copy(), seed=0)["p_raw"] == 1.0


def test_holm_is_monotone_and_bounded():
    raw = pd.Series({"a": 0.001, "b": 0.02, "c": 0.5})
    adj = holm(raw)
    assert (adj >= raw).all() and (adj <= 1).all()
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_pairwise_tests_declare_one_family_and_compare_against_reference():
    rng = np.random.default_rng(1)
    frames = {
        "scLDM_w1": frame(rng.normal(0.9, 0.05, 30)),
        "knn": frame(rng.normal(0.93, 0.05, 30)),
        "generic": frame(rng.normal(0.86, 0.05, 30)),
    }
    tests = pairwise_tests(frames, reference="scLDM_w1", n_boot=200, seed=0)
    assert set(tests.left) == {"scLDM_w1"}
    assert set(tests.right) == {"knn", "generic"}
    assert set(tests.family) == {"scLDM_w1 vs others"}
    assert "p_holm" in tests


def test_summarize_reports_n_and_interval_per_method():
    frames = {"a": frame(np.linspace(0.5, 1.0, 20)), "b": frame(np.linspace(0.4, 0.6, 20))}
    table = summarize(frames, n_boot=200)
    assert table.loc["a", "median"] > table.loc["b", "median"]
    assert table.loc["a", "n_perturbations"] == 20
    assert table.loc["a", "ci_low"] <= table.loc["a", "median"] <= table.loc["a", "ci_high"]
