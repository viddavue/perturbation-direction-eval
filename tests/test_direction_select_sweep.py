"""Sweep selection must score validation only, tie-break deterministically, and stay auditable."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from select_sweep import choose, delta_from_profiles, parse_name, score_config


def test_delta_subtracts_and_drops_control():
    prof = pd.DataFrame([[1.0, 1.0], [3.0, 2.0]], index=["non-targeting", "p1"], columns=["a", "b"])
    delta = delta_from_profiles(prof)
    assert list(delta.index) == ["p1"]
    np.testing.assert_allclose(delta.loc["p1"], [2.0, 1.0])


def test_missing_control_is_rejected():
    with pytest.raises(KeyError, match="control"):
        delta_from_profiles(pd.DataFrame([[1.0]], index=["p1"], columns=["a"]))


def test_config_names_parse():
    assert parse_name("w1") == (1.0, False)
    assert parse_name("w2_ema") == (2.0, True)
    assert parse_name("w4") == (4.0, False)


def test_ties_prefer_lower_guidance_then_raw():
    table = pd.DataFrame({
        "balanced_acc_median": [0.90, 0.901, 0.902],
        "guidance": [1.0, 1.0, 4.0],
        "ema": [True, False, False],
    }, index=["w1_ema", "w1", "w4"])
    assert choose(table, tie_tolerance=0.005) == "w1"
    tight = table.copy()
    tight["balanced_acc_median"] = [0.90, 0.902, 0.95]
    assert choose(tight, tie_tolerance=0.005) == "w4"


def test_scoring_uses_only_the_requested_perturbations():
    # six DEGs (three up, three down) so the row clears the >=5-DEG threshold the scorer applies
    row = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    truth = pd.DataFrame([row, row], index=["p1", "p2"], columns=list("abcdef"))
    deg = pd.DataFrame(True, index=truth.index, columns=truth.columns)
    pred = truth.copy()
    scores = score_config(pred, truth, deg, ["p1"])
    assert scores["n_scored"] == 1
    assert scores["balanced_acc_median"] == pytest.approx(1.0)


def test_scoring_rejects_a_prediction_without_validation_rows():
    truth = pd.DataFrame([[1.0]], index=["p1"], columns=["a"])
    deg = pd.DataFrame([[True]], index=["p1"], columns=["a"])
    with pytest.raises(ValueError, match="no validation perturbations"):
        score_config(pd.DataFrame([[1.0]], index=["other"], columns=["a"]), truth, deg, ["p1"])
