"""Reproducibility replicate comparison must detect drift and refuse incomparable inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.evaluation.replicate_check import compare


def frame():
    return pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["p1", "p2"], columns=["g1", "g2"])


def test_identical_runs_report_zero_difference():
    per_condition, summary = compare(frame(), frame())
    assert summary["max_abs_diff"] == 0.0
    assert summary["fraction_exactly_equal"] == 1.0
    assert summary["n_conditions_exactly_equal"] == 2
    assert per_condition.max_abs_diff.eq(0).all()


def test_drift_is_measured_not_hidden():
    drift = frame()
    drift.loc["p2", "g1"] = 3.5
    _, summary = compare(frame(), drift)
    assert summary["max_abs_diff"] == pytest.approx(0.5)
    assert summary["n_conditions_exactly_equal"] == 1


def test_row_and_column_order_do_not_matter():
    other = frame().iloc[::-1, ::-1]
    _, summary = compare(frame(), other)
    assert summary["max_abs_diff"] == 0.0


def test_incomparable_inputs_are_rejected():
    with pytest.raises(ValueError, match="not comparable"):
        compare(frame(), frame().rename(index={"p2": "p3"}))


def test_constant_row_has_undefined_correlation():
    a = pd.DataFrame([[1.0, 1.0]], index=["p"], columns=["g1", "g2"])
    per_condition, summary = compare(a, a)
    assert np.isnan(per_condition.pearson.iloc[0])
    assert summary["min_pearson"] is None
