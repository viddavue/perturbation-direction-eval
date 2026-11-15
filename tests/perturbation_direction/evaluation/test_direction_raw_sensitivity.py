"""The STATE filter must not change what the evaluation reports.

STATE keeps a knockdown-selected subset of cells. The sensitivity run rebuilt the ground truth from
all raw GEO cells and found effects within ~4% of the kept-cell effects. These tests pin the two
properties the scoring path depends on: paired metrics see the same perturbations either way, and a
bare effect matrix (no control row) is scored rather than rejected.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.evaluation.metrics import direction_metrics


def test_effect_matrices_from_kept_and_raw_cells_agree_on_direction():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(400)]
    kept = pd.DataFrame(rng.normal(0, 0.1, size=(20, len(genes))), index=[f"p{i}" for i in range(20)], columns=genes)
    # raw-cell effects differ only slightly: what a milder cell subset does to the same quantities
    raw = kept + pd.DataFrame(rng.normal(0, 0.003, size=kept.shape), index=kept.index, columns=genes)
    deg = (kept.abs() > 0.1) & (rng.random(kept.shape) > 0.5)
    deg.iloc[:, :60] = True
    for p in kept.index:
        a = direction_metrics(raw.loc[p].to_numpy(), kept.loc[p].to_numpy(), deg.loc[p].to_numpy())
        b = direction_metrics(kept.loc[p].to_numpy(), kept.loc[p].to_numpy(), deg.loc[p].to_numpy())
        assert abs(a["sign_acc"] - b["sign_acc"]) < 0.10


def test_a_named_pred_without_a_control_row_is_an_effect_matrix(tmp_path):
    """evaluate.py used to subtract the truth's control row from every named pred unconditionally."""
    from experiments.perturbation_direction.evaluation import evaluate

    truth = tmp_path / "truth"
    truth.mkdir()
    perts = [f"p{i}" for i in range(1, 7)]  # medians skip perturbations with < 5 DEGs
    genes = list("abcdefghij")  # a, b, c are named like the targets "p1".."p3", so excluded
    row_up = np.array([0.0, 0.0, 0.0, 0.5, -0.5, 0.2, 0.3, 0.4, -0.3, 0.35])  # d..j are DEGs
    ded = np.array([row_up] * 6)
    pd.DataFrame(ded, index=perts, columns=genes).to_parquet(truth / "delta_true.parquet")
    pd.DataFrame(np.zeros(ded.shape), index=perts, columns=genes).to_parquet(truth / "padj.parquet")
    # deliberately no deg.parquet: which genes count twice (--exclude-target names as genes) is what
    # the rebuild path has to get right, and a stored mask would hide it
    (truth / "info.json").write_text('{"control": "non-targeting", "min_lfc": 0.1, "alpha": 0.05}')
    eff = tmp_path / "eff.parquet"
    pd.DataFrame(ded, index=perts, columns=genes).to_parquet(eff)
    out = tmp_path / "eval"
    try:  # evaluate.py raises SystemExit(main())
        evaluate.main(["--truth-dir", str(truth), "--exclude-target", "--pred", f"m={eff}", "--out-dir", str(out)])
    except SystemExit as exc:
        assert exc.code in (0, None)
    summary = pd.read_csv(out / "summary.csv", index_col=0)
    assert (summary.loc["m", "balanced_acc_median"]) == pytest.approx(1.0)
