"""The third report, changed vs unchanged genes: does |predicted effect| separate DEGs from unchanged genes?

metrics.py (unchanged_mask, detection_metrics, evaluate_detection, summarize_detection) on hand-computed cases,
evaluate.py end to end on a small synthetic truth dir, and side_by_side.py's third report on synthetic
evaluate.py output dirs.

The synthetic truth (8 perturbations G00..G07 x 60 genes G00..G59; min_lfc 0.1, alpha 0.05):

  G00..G07   a perturbation's own gene is its CRISPRi target: a strong DEG, excluded with --exclude-target
  G10.., G30..  k up-DEGs and k down-DEGs, k = 0, 2, 3, 4, 5, 6, 7, 8 for G00..G07
                (target excluded: 0, 4, 6, ... 16 DEGs -> G00 and G01 are below --min-degs 5)
  G49        padj is NaN: never unchanged
  G50..G59   grey zone: not significant (padj 0.3) and |delta| = 0.07, not clearly unchanged
  the rest   clearly unchanged: |delta| <= 0.03, padj >= 0.6; G07 has only 6 of them (the others: padj 0.3)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

import dashboard_data as dd
import evaluate as evaluate_script
import freeze_results
import side_by_side as sbs
import stats
from metrics import (
    DETECTION_COUNTS,
    detection_metrics,
    evaluate,
    evaluate_detection,
    shuffled_prediction,
    summarize,
    summarize_detection,
    unchanged_mask,
)

CEILING = "noise ceiling (split-half)"
METRICS = (
    "detect_auroc",
    "detect_auroc_all",
    "abs_pred_deg_median",
    "abs_pred_unchanged_median",
    "unchanged_call_rate",
    "deg_call_rate",
    "deg_hit_rate",
    "three_class_balanced_acc",
)

# ---------------------------------------------------------------------------
# metrics: the unchanged mask
# ---------------------------------------------------------------------------


def test_unchanged_mask_definition_and_boundaries():
    #                 clear   |d|=max  padj=min  padj<min  DEG     NaN padj  NaN delta  neg. small
    delta = np.array([0.01, 0.05, 0.04, 0.01, 0.01, 0.01, np.nan, -0.049])
    padj = np.array([0.90, 0.90, 0.50, 0.49, 0.90, np.nan, 0.90, 1.0])
    deg = np.array([False, False, False, False, True, False, False, False])
    got = unchanged_mask(delta, padj, deg, max_effect=0.05, min_padj=0.5)
    # |delta| < max_effect is strict, padj >= min_padj is inclusive; a DEG or a NaN is never unchanged
    assert got.tolist() == [True, False, True, False, False, False, False, True]
    assert got.dtype == bool
    assert unchanged_mask(delta, padj, deg, max_effect=0.06, min_padj=0.95).tolist() == [False] * 7 + [True]
    with pytest.raises(ValueError, match="same shape"):
        unchanged_mask(delta, padj[:3], deg)


def test_unchanged_mask_aligns_frames_to_the_effects():
    genes = ["a", "b", "c"]
    delta = pd.DataFrame([[0.0, 0.01, 0.5], [0.02, 0.0, 0.0]], index=["p", "q"], columns=genes)
    padj = pd.DataFrame([[0.9, 0.9], [0.9, 0.9]], index=["q", "p"], columns=["b", "a"])  # no gene c; other order
    deg = pd.DataFrame([[True]], index=["q"], columns=["a"])  # only (q, a) is a DEG
    got = unchanged_mask(delta, padj, deg)
    assert got.index.tolist() == ["p", "q"] and got.columns.tolist() == genes
    # (q, a) is a DEG; gene c has no padj, so it is never unchanged, whatever its effect
    assert got.to_numpy().tolist() == [[True, True, False], [False, True, False]]


# ---------------------------------------------------------------------------
# metrics: per-perturbation numbers, by hand
# ---------------------------------------------------------------------------
#                 --------- changed (DEGs) ---------  ----------- unchanged ----------  --- grey ---
TRUE = np.array([1.0, -0.5, 0.4, -0.3, 0.01, -0.02, 0.0, 0.03, 0.08, -0.07])
PRED = np.array([0.30, -0.20, -0.15, 0.05, 0.20, 0.05, -0.02, 0.0, 0.25, 0.01])
DEG = np.array([True] * 4 + [False] * 6)
UNCH = np.array([False] * 4 + [True] * 4 + [False] * 2)


def test_detection_metrics_hand_computed():
    m = detection_metrics(PRED, TRUE, DEG, UNCH, tau=0.1)
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (4, 4, 2)
    # |pred| changed: .30 .20 .15 .05; unchanged: .20 .05 .02 .00. Pairs won (ties count half):
    # .30 -> 4, .20 -> 3.5 (ties .20), .15 -> 3, .05 -> 2.5 (ties .05): 13 of 16
    assert m["detect_auroc"] == pytest.approx(13 / 16)
    # negatives + grey (.25, .01): .30 -> 6, .20 -> 4.5, .15 -> 4, .05 -> 3.5: 18 of 24
    assert m["detect_auroc_all"] == pytest.approx(18 / 24)
    assert m["abs_pred_deg_median"] == pytest.approx(0.175)
    assert m["abs_pred_unchanged_median"] == pytest.approx(0.035)
    assert m["unchanged_call_rate"] == pytest.approx(1 / 4)  # only the .20 exceeds tau
    assert m["deg_call_rate"] == pytest.approx(3 / 4)  # .05 is not called
    assert m["deg_hit_rate"] == pytest.approx(2 / 4)  # -.15 is called, but the gene goes up
    # up-DEGs (1.0, 0.4): 1 of 2 predicted > tau; down-DEGs (-0.5, -0.3): 1 of 2 < -tau; unchanged: 3 of 4 within
    assert m["three_class_balanced_acc"] == pytest.approx((0.5 + 0.5 + 0.75) / 3)
    assert list(m) == [*DETECTION_COUNTS, *METRICS]
    # the rank formula is sklearn's AUROC
    both = DEG | UNCH
    assert m["detect_auroc"] == pytest.approx(roc_auc_score(DEG[both], np.abs(PRED)[both]))
    assert m["detect_auroc_all"] == pytest.approx(roc_auc_score(DEG, np.abs(PRED)))
    # only |pred| enters the AUROCs and the call rates: flipping every sign keeps them and zeroes the hits
    flipped = detection_metrics(-PRED, TRUE, DEG, UNCH, tau=0.1)
    for k in ("detect_auroc", "detect_auroc_all", "deg_call_rate", "unchanged_call_rate", "abs_pred_deg_median"):
        assert flipped[k] == m[k]
    assert flipped["deg_hit_rate"] == pytest.approx(1 / 4)  # only the wrongly signed -.15 is now right


def test_constant_and_perfect_predictions():
    # a constant prediction ranks nothing: every pair is a tie, half credit -> exactly 0.5 (not NaN)
    for value in (0.0, 0.3, -0.3):
        m = detection_metrics(np.full(10, value), TRUE, DEG, UNCH, tau=0.1)
        assert m["detect_auroc"] == 0.5 and m["detect_auroc_all"] == 0.5
        called = float(abs(value) > 0.1)
        assert m["deg_call_rate"] == called and m["unchanged_call_rate"] == called
    assert detection_metrics(np.zeros(10), TRUE, DEG, UNCH)["three_class_balanced_acc"] == pytest.approx(1 / 3)
    assert detection_metrics(np.full(10, 0.3), TRUE, DEG, UNCH)["three_class_balanced_acc"] == pytest.approx(1 / 3)
    assert detection_metrics(np.full(10, 0.3), TRUE, DEG, UNCH)["deg_hit_rate"] == pytest.approx(0.5)  # the up-DEGs
    # the truth itself: perfect separation, every DEG hit, no unchanged gene called
    m = detection_metrics(TRUE, TRUE, DEG, UNCH, tau=0.1)
    assert m["detect_auroc"] == 1.0 and m["three_class_balanced_acc"] == 1.0
    assert m["deg_hit_rate"] == 1.0 and m["unchanged_call_rate"] == 0.0
    assert m["detect_auroc_all"] == 1.0
    # predicting large effects exactly where nothing changes: 0
    inverse = np.where(DEG, 0.0, 1.0)
    assert detection_metrics(inverse, TRUE, DEG, UNCH)["detect_auroc"] == 0.0


def test_empty_classes_are_nan():
    none = np.zeros(10, dtype=bool)
    m = detection_metrics(PRED, TRUE, none, UNCH)  # no DEG: the four former DEGs are grey
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (0, 4, 6)
    for k in ("detect_auroc", "detect_auroc_all", "abs_pred_deg_median", "deg_call_rate", "deg_hit_rate"):
        assert np.isnan(m[k]), k
    assert np.isnan(m["three_class_balanced_acc"])
    assert m["unchanged_call_rate"] == pytest.approx(0.25) and m["abs_pred_unchanged_median"] == pytest.approx(0.035)

    m = detection_metrics(PRED, TRUE, DEG, none)  # no unchanged gene: the headline is undefined
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (4, 0, 6)
    assert np.isnan(m["detect_auroc"]) and np.isnan(m["unchanged_call_rate"])
    assert np.isnan(m["abs_pred_unchanged_median"]) and np.isnan(m["three_class_balanced_acc"])
    assert m["detect_auroc_all"] == pytest.approx(roc_auc_score(DEG, np.abs(PRED)))  # vs the six grey genes
    assert m["deg_call_rate"] == pytest.approx(0.75)

    only_up = DEG & (TRUE > 0)  # no down-DEG: the three-class accuracy needs all three classes
    m = detection_metrics(PRED, TRUE, only_up, UNCH)
    assert np.isnan(m["three_class_balanced_acc"]) and not np.isnan(m["detect_auroc"])
    assert not np.isnan(m["deg_hit_rate"])

    m = detection_metrics(PRED[:4], TRUE[:4], DEG[:4], UNCH[:4])  # only DEGs scored: nothing to rank against
    assert np.isnan(m["detect_auroc"]) and np.isnan(m["detect_auroc_all"]) and m["n_grey"] == 0


def test_tau_boundary_is_not_a_call():
    true = np.array([1.0, -1.0, 0.0, 0.0])
    deg = np.array([True, True, False, False])
    unch = ~deg
    at_tau = np.array([0.1, -0.1, 0.1, -0.1])  # |pred| == tau exactly
    m = detection_metrics(at_tau, true, deg, unch, tau=0.1)
    assert m["deg_call_rate"] == 0.0 and m["deg_hit_rate"] == 0.0 and m["unchanged_call_rate"] == 0.0
    assert m["three_class_balanced_acc"] == pytest.approx(1 / 3)  # both DEGs missed, both unchanged genes kept
    above = np.nextafter(at_tau, np.sign(at_tau) * np.inf)  # one ulp further from zero: a call
    m = detection_metrics(above, true, deg, unch, tau=0.1)
    assert m["deg_call_rate"] == 1.0 and m["deg_hit_rate"] == 1.0 and m["unchanged_call_rate"] == 1.0
    assert m["three_class_balanced_acc"] == pytest.approx(2 / 3)
    # a larger tau turns calls into no-calls; the rank-based headline does not move
    lo, hi = detection_metrics(PRED, TRUE, DEG, UNCH, tau=0.1), detection_metrics(PRED, TRUE, DEG, UNCH, tau=0.22)
    assert hi["deg_call_rate"] == pytest.approx(1 / 4) and hi["unchanged_call_rate"] == 0.0
    assert hi["detect_auroc"] == lo["detect_auroc"] and hi["detect_auroc_all"] == lo["detect_auroc_all"]


def test_scored_mask_excludes_genes_from_every_class():
    # leaving out the strongest DEG (index 0), the false positive (index 4) and one grey gene (index 8)
    scored = np.ones(10, dtype=bool)
    scored[[0, 4, 8]] = False
    m = detection_metrics(PRED, TRUE, DEG, UNCH, tau=0.1, scored=scored)
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (3, 3, 1)
    keep = np.flatnonzero(scored)
    same = detection_metrics(PRED[keep], TRUE[keep], DEG[keep], UNCH[keep], tau=0.1)
    assert m == same
    # changed .20 .15 .05 vs unchanged .05 .02 .00: 3 + 3 + 2.5 of 9
    assert m["detect_auroc"] == pytest.approx(8.5 / 9)
    assert m["unchanged_call_rate"] == 0.0


def test_nan_predictions_zero_true_effects_and_overlapping_masks():
    pred = PRED.copy()
    pred[[1, 5]] = np.nan  # no prediction for one DEG and one unchanged gene
    m = detection_metrics(pred, TRUE, DEG, UNCH, tau=0.1)
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (4, 4, 2)  # the counts describe the truth
    have = np.isfinite(pred)
    rest = detection_metrics(pred[have], TRUE[have], DEG[have], UNCH[have], tau=0.1)
    for k in METRICS:  # the metrics use the genes that have a prediction
        assert m[k] == rest[k], k
    assert m["deg_call_rate"] == pytest.approx(2 / 3)

    # a "DEG" without a true effect has no direction: as in direction_metrics it is not a changed gene (grey)
    true = TRUE.copy()
    true[3] = 0.0
    m = detection_metrics(PRED, true, DEG, UNCH)
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (3, 4, 3)
    # a gene flagged as both is a DEG
    m = detection_metrics(PRED, TRUE, DEG, np.ones(10, dtype=bool))
    assert (m["n_deg"], m["n_unchanged"], m["n_grey"]) == (4, 6, 0)
    with pytest.raises(ValueError, match="same shape"):
        detection_metrics(PRED, TRUE, DEG, UNCH[:5])


def test_evaluate_detection_excludes_target_and_aligns_frames():
    genes = ["A", "B", "C", "D", "E", "F"]
    true = pd.DataFrame(
        [[-3.0, 1.0, -1.0, 0.0, 0.01, 0.2], [0.0, 1.0, -1.0, 0.02, 0.0, 0.01]], index=["A", "X"], columns=genes
    )
    deg = pd.DataFrame(
        [[True, True, True, False, False, False], [False, True, True, False, False, False]],
        index=["A", "X"],
        columns=genes,
    )
    unch = pd.DataFrame(
        [[False, False, False, True, True, False], [True, False, False, True, True, True]],
        index=["A", "X"],
        columns=genes,
    )
    # columns in another order, an extra perturbation Z; A: perfect except for its target
    pred = pd.DataFrame(
        [[0.0, 0.5, -0.5, 0.0, 0.02, 0.3], [9, 9, 9, 9, 9, 9], [0.3, 0.05, -0.4, 0.0, 0.0, 0.2]],
        index=["A", "Z", "X"],
        columns=genes,
    )[genes[::-1]]
    res = evaluate_detection(pred, true, deg, unch, exclude_genes={"A": ["A"], "X": ["X"]}, tau=0.1)
    assert list(res.index) == ["A", "X"]
    # A: target A (a DEG predicted 0) is excluded -> 2 DEGs, 2 unchanged, 1 grey (F), separated perfectly
    assert res.loc["A", ["n_deg", "n_unchanged", "n_grey"]].tolist() == [2, 2, 1]
    assert res.loc["A", "detect_auroc"] == 1.0 and res.loc["A", "deg_hit_rate"] == 1.0
    assert res.loc["A", "detect_auroc_all"] == 1.0  # grey F (.3) is below both DEGs (.5)
    # X is no gene of its own row's exclusion list in the columns: nothing dropped; unchanged A, D, E, F
    assert res.loc["X", ["n_deg", "n_unchanged", "n_grey"]].tolist() == [2, 4, 0]
    # changed .05 .40 vs unchanged .30 .00 .00 .20: .05 beats two, .40 beats four -> 6 of 8
    assert res.loc["X", "detect_auroc"] == pytest.approx(6 / 8)
    assert res.loc["X", "unchanged_call_rate"] == pytest.approx(2 / 4)
    with_target = evaluate_detection(pred, true, deg, unch, tau=0.1)
    assert with_target.loc["A", "n_deg"] == 3 and with_target.loc["A", "deg_call_rate"] == pytest.approx(2 / 3)
    # an unchanged target is excluded from the unchanged genes in the same way
    res = evaluate_detection(pred, true, deg, unch, exclude_genes={"X": ["A"]})
    assert res.loc["X", "n_unchanged"] == 3 and res.loc["A", "n_deg"] == 3
    # no common perturbation: an empty table with the count columns, not a crash
    empty = evaluate_detection(pred.loc[["Z"]], true, deg, unch)
    assert len(empty) == 0 and {"n_deg", "n_unchanged"} <= set(empty.columns)


def test_summarize_detection_applies_both_panel_rules():
    per = pd.DataFrame(
        {
            "n_deg": [10, 2, 30, 8, 50],
            "n_unchanged": [100, 100, 5, 20, 400],
            "n_grey": [1, 1, 1, 1, 1],
            "detect_auroc": [0.6, 0.0, 0.1, 0.8, 0.9],
            "deg_call_rate": [0.5, 1.0, 1.0, np.nan, 0.7],
        },
        index=list("abcde"),
    )
    s = summarize_detection(per, min_degs=5, min_unchanged=20)  # b: too few DEGs; c: too few unchanged genes
    assert s["n_perturbations"] == 3 and s["n_perturbations_total"] == 5
    assert s["detect_auroc_median"] == pytest.approx(0.8)
    assert s["detect_auroc_iqr"] == pytest.approx(0.85 - 0.7)
    assert s["deg_call_rate_median"] == pytest.approx(0.6)  # NaN skipped
    # the shape of metrics.summarize: n, total, then <metric>_median / _iqr; the gene counts are not metrics
    assert list(s.index) == [
        "n_perturbations",
        "n_perturbations_total",
        "detect_auroc_median",
        "detect_auroc_iqr",
        "deg_call_rate_median",
        "deg_call_rate_iqr",
    ]
    assert summarize_detection(per, min_degs=5, min_unchanged=21)["n_perturbations"] == 2  # d has exactly 20
    assert summarize_detection(per, min_degs=0, min_unchanged=0)["n_perturbations"] == 5
    none = summarize_detection(per, min_degs=99, min_unchanged=0)
    assert none["n_perturbations"] == 0 and np.isnan(none["detect_auroc_median"])


# ---------------------------------------------------------------------------
# evaluate.py end to end on a synthetic truth dir
# ---------------------------------------------------------------------------
GENES = [f"G{i:02d}" for i in range(60)]
PERTS = GENES[:8]
N_DEG_PAIRS = [0, 2, 3, 4, 5, 6, 7, 8]  # k up + k down DEGs per perturbation
CTRL = "non-targeting"
EXISTING = (  # evaluate.py outputs that predate the report and must not change
    "summary.csv",
    "summary_specific.csv",
    "by_strength.csv",
    "by_strength_specific.csv",
)


def synthetic_truth(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(delta_true, padj) as described in the module docstring."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-0.03, 0.03, (len(PERTS), len(GENES)))
    padj = rng.uniform(0.6, 1.0, (len(PERTS), len(GENES)))
    for i, k in enumerate(N_DEG_PAIRS):
        delta[i, 10 : 10 + k] = rng.uniform(0.3, 1.0, k)
        delta[i, 30 : 30 + k] = -rng.uniform(0.3, 1.0, k)
        padj[i, 10 : 10 + k] = padj[i, 30 : 30 + k] = 0.001
        delta[i, i], padj[i, i] = -1.5, 1e-6  # the CRISPRi target
    delta[:, 50:] = 0.07 * rng.choice([-1, 1], (len(PERTS), 10))
    padj[:, 50:] = 0.3
    padj[:, 49] = np.nan
    weak = [j for j in range(8, 49) if abs(delta[7, j]) < 0.05][6:]  # G07: all but 6 unchanged genes -> grey
    padj[7, weak] = 0.3
    padj[7, :7] = 0.3
    frame = lambda x: pd.DataFrame(x, index=PERTS, columns=GENES)
    return frame(delta), frame(padj)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """A truth dir (with halves and half A's padj), a model prediction and three baselines."""
    d = tmp_path_factory.mktemp("unchanged")
    rng = np.random.default_rng(1)
    delta, padj = synthetic_truth()
    deg = (padj < 0.05) & (delta.abs() > 0.1)
    truth = d / "truth"
    truth.mkdir()
    delta.to_parquet(truth / "delta_true.parquet")
    padj.to_parquet(truth / "padj.parquet")
    deg.to_parquet(truth / "deg.parquet")
    # halves: the same effects plus sampling noise; half A's test is a little weaker
    half_a = delta + rng.normal(0, 0.02, delta.shape)
    half_b = delta + rng.normal(0, 0.02, delta.shape)
    padj_a = padj.where(padj > 0.05, 0.01)
    padj_a.iloc[2, 10] = 0.2  # one of G02's DEGs is not significant in half A
    deg_a = (padj_a < 0.05) & (half_a.abs() > 0.1)
    half_a.to_parquet(truth / "delta_half_a.parquet")
    half_b.to_parquet(truth / "delta_half_b.parquet")
    deg_a.to_parquet(truth / "deg_half_a.parquet")
    padj_a.to_parquet(truth / "padj_half_a.parquet")
    evaluate(half_b, half_a, deg_a, exclude_genes={p: [p] for p in PERTS}).to_csv(truth / "ceiling.csv")
    (truth / "info.json").write_text(json.dumps({"control": CTRL, "alpha": 0.05, "min_lfc": 0.1}))

    model = delta + rng.normal(0, 0.04, delta.shape)
    profiles = pd.concat([pd.DataFrame(2.0, index=[CTRL], columns=GENES), 2.0 + model])
    profiles.to_parquet(d / "model.parquet")
    (delta * 0.5 + rng.normal(0, 0.05, delta.shape)).to_parquet(d / "weak.parquet")
    generic = pd.DataFrame(np.tile(delta.mean().to_numpy(), (len(PERTS), 1)), index=PERTS, columns=GENES)
    generic.to_parquet(d / "generic.parquet")
    (np.sign(delta) * rng.uniform(0.3, 0.5, delta.shape)).to_parquet(d / "conf.parquet")  # P(up) - 0.5: no size
    return d


def run_evaluate(inputs: Path, out: Path, *extra: str, truth: Path | None = None) -> Path:
    evaluate_script.main(
        [
            "--truth-dir",
            str(truth or inputs / "truth"),
            "--pred",
            f"model={inputs / 'model.parquet'}",
            "--baseline",
            f"weak={inputs / 'weak.parquet'}",
            f"conf={inputs / 'conf.parquet'}",
            f"generic={inputs / 'generic.parquet'}",
            "--specific-ref",
            str(inputs / "generic.parquet"),
            "--exclude-target",
            "--out-dir",
            str(out),
            *extra,
        ]
    )
    return out


def expected_panel(min_degs: int = 5, min_unchanged: int = 20, max_effect: float = 0.05, min_padj: float = 0.5):
    """The report's panel and masks, recomputed with plain numpy (target excluded)."""
    delta, padj = synthetic_truth()
    d, q = delta.to_numpy(), padj.to_numpy()
    deg = (q < 0.05) & (np.abs(d) > 0.1)
    unch = ~deg & (np.abs(d) < max_effect) & (q >= min_padj)
    not_target = ~np.eye(len(PERTS), len(GENES), dtype=bool)
    n_deg, n_unch = (deg & not_target).sum(1), (unch & not_target).sum(1)
    panel = [p for p, a, b in zip(PERTS, n_deg, n_unch, strict=True) if a >= min_degs and b >= min_unchanged]
    return panel, deg, unch, not_target, n_deg, n_unch


def test_fixture_has_the_cases_the_docstring_promises():
    panel, deg, unch, not_target, n_deg, n_unch = expected_panel()
    assert n_deg.tolist() == [2 * k for k in N_DEG_PAIRS]  # the target is not counted
    assert n_unch[7] == 6 and (n_unch[:7] >= 20).all()
    assert panel == ["G02", "G03", "G04", "G05", "G06"]  # G00, G01: < 5 DEGs; G07: < 20 unchanged genes
    assert not unch[:, 49].any() and not unch[:, 50:].any()  # NaN padj and the grey zone


def test_evaluate_writes_the_report(inputs, tmp_path, capsys):
    out = run_evaluate(inputs, tmp_path / "eval", "--direction-only", "conf")
    log = capsys.readouterr().out
    assert "left out as direction-only (a direction confidence, not an effect size): conf" in log
    assert "panel: 5 of 8 perturbations with >= 5 DEGs and >= 20 unchanged genes" in log

    summary = pd.read_csv(out / "summary_unchanged.csv", index_col=0)
    assert list(summary.index) == ["model", "model (shuffled)", "weak", "generic", CEILING]  # no conf
    assert list(summary.columns[:3]) == ["n_perturbations", "detect_auroc_median", "detect_auroc_all_median"]
    assert set(summary.columns) == {
        "n_perturbations",
        "n_perturbations_total",
        *(f"{m}_{s}" for m in METRICS for s in ("median", "iqr")),
    }
    assert (summary["n_perturbations"] == 5).all() and (summary["n_perturbations_total"] == 8).all()
    files = {p.name for p in (out / "per_perturbation_unchanged").glob("*.csv")}
    assert files == {"model.csv", "model_(shuffled).csv", "weak.csv", "generic.csv", "noise_ceiling_(split-half).csv"}

    # per perturbation: the same numbers from the plain definition
    panel, deg, unch, not_target, n_deg, n_unch = expected_panel()
    delta, _ = synthetic_truth()
    profiles = pd.read_parquet(inputs / "model.parquet")
    pred = (profiles.loc[PERTS] - profiles.loc[CTRL]).to_numpy()
    per = pd.read_csv(out / "per_perturbation_unchanged" / "model.csv", index_col=0)
    assert list(per.index) == PERTS and list(per.columns) == [*DETECTION_COUNTS, *METRICS]
    assert per["n_deg"].tolist() == n_deg.tolist() and per["n_unchanged"].tolist() == n_unch.tolist()
    assert (per["n_deg"] + per["n_unchanged"] + per["n_grey"] == len(GENES) - 1).all()  # every gene but the target
    for i, p in enumerate(PERTS):
        c, u = deg[i] & not_target[i], unch[i] & not_target[i]
        score = np.abs(pred[i])
        if c.any():
            auroc = roc_auc_score(np.r_[np.ones(c.sum()), np.zeros(u.sum())], np.r_[score[c], score[u]])
            assert per.loc[p, "detect_auroc"] == pytest.approx(auroc)
            assert per.loc[p, "detect_auroc_all"] == pytest.approx(
                roc_auc_score(c[not_target[i]], score[not_target[i]])
            )
            hit = (score[c] > 0.1) & (np.sign(pred[i][c]) == np.sign(delta.to_numpy()[i][c]))
            assert per.loc[p, "deg_hit_rate"] == pytest.approx(hit.mean())
        else:
            assert np.isnan(per.loc[p, "detect_auroc"]) and np.isnan(per.loc[p, "deg_call_rate"])
        assert per.loc[p, "unchanged_call_rate"] == pytest.approx((score[u] > 0.1).mean())
        assert per.loc[p, "abs_pred_unchanged_median"] == pytest.approx(np.median(score[u]))

    # summary = median and IQR over the panel
    for m in METRICS:
        values = per.loc[panel, m]
        assert summary.loc["model", f"{m}_median"] == pytest.approx(values.median())
        q1, q3 = values.quantile([0.25, 0.75])
        assert summary.loc["model", f"{m}_iqr"] == pytest.approx(q3 - q1)
    # the model sees the effects (noise 0.04); the shared response alone and a shuffled model see less
    assert summary.loc["model", "detect_auroc_median"] > 0.95
    assert summary.loc["model", "detect_auroc_median"] > summary.loc["model (shuffled)", "detect_auroc_median"]
    assert summary.loc["model", "detect_auroc_median"] > summary.loc["generic", "detect_auroc_median"]

    manifest = json.loads((out / "methods.json").read_text())
    assert manifest["direction_only"] == ["conf"]
    assert manifest["unchanged"] == {
        "max_effect": 0.05,
        "min_padj": 0.5,
        "min_unchanged": 20,
        "tau": 0.1,
        "methods": ["model", "model (shuffled)", "weak", "generic"],
        "direction_only_excluded": ["conf"],
        "ceiling": True,
    }


def test_shuffled_control_and_ceiling_use_the_documented_inputs(inputs, tmp_path):
    out = run_evaluate(inputs, tmp_path / "eval", "--seed", "3")
    truth = inputs / "truth"
    delta, padj = synthetic_truth()
    deg = pd.read_parquet(truth / "deg.parquet")
    exclude = {p: [p] for p in PERTS}
    profiles = pd.read_parquet(inputs / "model.parquet")
    pred = profiles.loc[PERTS] - profiles.loc[CTRL]
    # the same derangement (seed) as the full report's shuffled control
    shuffled = evaluate_detection(
        shuffled_prediction(pred, seed=3), delta, deg, unchanged_mask(delta, padj, deg), exclude_genes=exclude
    )
    got = pd.read_csv(out / "per_perturbation_unchanged" / "model_(shuffled).csv", index_col=0)
    pd.testing.assert_frame_equal(got, shuffled, check_dtype=False, check_names=False)
    full = pd.read_csv(out / "per_perturbation" / "model_(shuffled).csv", index_col=0)
    again = evaluate(shuffled_prediction(pred, seed=3), delta, deg, exclude_genes=exclude)
    pd.testing.assert_frame_equal(full, again, check_dtype=False, check_names=False)

    # ceiling: labels and signs from half A (its own DEGs and padj), |delta| of half B
    half_a, half_b = (pd.read_parquet(truth / f"delta_half_{h}.parquet") for h in "ab")
    deg_a, padj_a = pd.read_parquet(truth / "deg_half_a.parquet"), pd.read_parquet(truth / "padj_half_a.parquet")
    expected = evaluate_detection(half_b, half_a, deg_a, unchanged_mask(half_a, padj_a, deg_a), exclude_genes=exclude)
    ceiling = pd.read_csv(out / "per_perturbation_unchanged" / "noise_ceiling_(split-half).csv", index_col=0)
    pd.testing.assert_frame_equal(ceiling, expected, check_dtype=False, check_names=False)
    assert ceiling.loc["G02", "n_deg"] == 5  # one of its 6 DEGs is not significant in half A
    # ... but the ceiling's summary row follows the panel of the full data (G02 has 6 DEGs there), like the others
    summary = pd.read_csv(out / "summary_unchanged.csv", index_col=0)
    panel = expected_panel()[0]
    assert summary.loc[CEILING, "n_perturbations"] == len(panel) == 5
    assert summary.loc[CEILING, "detect_auroc_median"] == pytest.approx(ceiling.loc[panel, "detect_auroc"].median())
    # without --direction-only the confidence baseline is scored like any other method
    assert "conf" in summary.index


def test_report_without_half_a_padj_has_no_ceiling(inputs, tmp_path, capsys):
    """Truth dirs written before ground_truth.py saved padj_half_a.parquet: the report is written, without a ceiling."""
    old = tmp_path / "truth_old"
    old.mkdir()
    for f in (inputs / "truth").iterdir():
        if f.name != "padj_half_a.parquet":
            (old / f.name).write_bytes(f.read_bytes())
    with_ceiling = run_evaluate(inputs, tmp_path / "new", "--direction-only", "conf")
    capsys.readouterr()
    out = run_evaluate(inputs, tmp_path / "old", "--direction-only", "conf", truth=old)
    log = capsys.readouterr().out
    assert "no split-half ceiling for this report" in log and "padj_half_a.parquet" in log
    summary = pd.read_csv(out / "summary_unchanged.csv", index_col=0)
    assert list(summary.index) == ["model", "model (shuffled)", "weak", "generic"]
    assert not (out / "per_perturbation_unchanged" / "noise_ceiling_(split-half).csv").exists()
    assert json.loads((out / "methods.json").read_text())["unchanged"]["ceiling"] is False
    # every other row is what the complete truth dir gives; the direction reports keep their ceilings
    complete = pd.read_csv(with_ceiling / "summary_unchanged.csv", index_col=0)
    pd.testing.assert_frame_equal(summary, complete.drop(index=CEILING))
    for name in EXISTING:
        assert (out / name).read_bytes() == (with_ceiling / name).read_bytes(), name
    assert CEILING in pd.read_csv(out / "summary.csv", index_col=0).index
    assert CEILING in pd.read_csv(out / "summary_specific.csv", index_col=0).index


def test_report_needs_padj_and_can_be_switched_off(inputs, tmp_path, capsys):
    no_padj = tmp_path / "truth_no_padj"
    no_padj.mkdir()
    for f in (inputs / "truth").iterdir():
        if not f.name.startswith("padj"):
            (no_padj / f.name).write_bytes(f.read_bytes())
    for out in (
        run_evaluate(inputs, tmp_path / "a", truth=no_padj),
        run_evaluate(inputs, tmp_path / "b", "--no-unchanged-report"),
    ):
        assert not (out / "summary_unchanged.csv").exists() and not (out / "per_perturbation_unchanged").exists()
        manifest = json.loads((out / "methods.json").read_text())
        assert manifest["unchanged"] is None and manifest["direction_only"] == []
        assert (out / "summary.csv").exists() and (out / "summary_specific.csv").exists()
    assert "the changed-vs-unchanged report is not written" in capsys.readouterr().out


def test_thresholds_and_panel_options(inputs, tmp_path):
    # G07 has 6 unchanged genes: --min-unchanged 6 keeps it, 7 drops it
    for min_unchanged, n in ((6, 6), (7, 5), (1000, 0)):
        out = run_evaluate(inputs, tmp_path / f"m{min_unchanged}", "--min-unchanged", str(min_unchanged))
        summary = pd.read_csv(out / "summary_unchanged.csv", index_col=0)
        assert (summary["n_perturbations"] == n).all(), min_unchanged
        assert len(expected_panel(min_unchanged=min_unchanged)[0]) == n
    assert summary["detect_auroc_median"].isna().all()  # an empty panel: NaN, not a crash
    assert json.loads((out / "methods.json").read_text())["unchanged"]["min_unchanged"] == 1000
    # the same rule as the full report for the DEGs: --min-degs 4 adds G01 (4 DEGs without its target)
    out = run_evaluate(inputs, tmp_path / "d4", "--min-degs", "4")
    assert pd.read_csv(out / "summary_unchanged.csv", index_col=0).loc["model", "n_perturbations"] == 6
    assert pd.read_csv(out / "summary.csv", index_col=0).loc["model", "n_perturbations"] == 7  # incl. G07

    # the thresholds define the unchanged genes: a wider effect window and a lower padj take in the grey zone
    out = run_evaluate(inputs, tmp_path / "wide", "--unchanged-max-effect", "0.08", "--unchanged-min-padj", "0.25")
    per = pd.read_csv(out / "per_perturbation_unchanged" / "model.csv", index_col=0)
    n_unch = expected_panel(max_effect=0.08, min_padj=0.25)[5]
    assert per["n_unchanged"].tolist() == n_unch.tolist()
    default = pd.read_csv(run_evaluate(inputs, tmp_path / "default") / "per_perturbation_unchanged" / "model.csv")
    assert (per["n_unchanged"].to_numpy() > default["n_unchanged"].to_numpy()).all()
    assert (per["n_deg"].to_numpy() == default["n_deg"].to_numpy()).all()  # the DEG call does not move
    manifest = json.loads((out / "methods.json").read_text())
    assert manifest["unchanged"]["max_effect"] == 0.08 and manifest["unchanged"]["min_padj"] == 0.25


def test_direction_only_names(inputs, tmp_path, capsys):
    # a prediction declared direction-only takes its shuffled control with it; an unknown name is reported
    out = run_evaluate(inputs, tmp_path / "eval", "--direction-only", "model", "conf", "nobody")
    log = capsys.readouterr().out
    assert "direction-only (a direction confidence, not an effect size): model, model (shuffled), conf" in log
    assert "--direction-only names not scored here: nobody" in log
    summary = pd.read_csv(out / "summary_unchanged.csv", index_col=0)
    assert list(summary.index) == ["weak", "generic", CEILING]
    manifest = json.loads((out / "methods.json").read_text())
    assert manifest["direction_only"] == ["model", "conf", "nobody"]
    assert manifest["unchanged"]["direction_only_excluded"] == ["model", "model (shuffled)", "conf"]
    # the direction reports still score every method
    assert {"model", "model (shuffled)", "conf"} <= set(pd.read_csv(out / "summary.csv", index_col=0).index)


def test_rerun_into_the_same_dir_leaves_no_stale_tables(inputs, tmp_path, capsys):
    """side_by_side.py reads every table of the folder: a method dropped by a rerun must not linger there."""
    out = run_evaluate(inputs, tmp_path / "eval")  # conf scored, ceiling present
    folder = out / "per_perturbation_unchanged"
    assert (folder / "conf.csv").exists() and (folder / "noise_ceiling_(split-half).csv").exists()
    old = tmp_path / "truth_old"
    old.mkdir()
    for f in (inputs / "truth").iterdir():
        if f.name != "padj_half_a.parquet":
            (old / f.name).write_bytes(f.read_bytes())
    capsys.readouterr()
    run_evaluate(inputs, out, "--direction-only", "conf", truth=old)  # now: no conf, no ceiling
    assert (
        "removed 2 stale table(s) of an earlier run: conf.csv, noise_ceiling_(split-half).csv"
        in capsys.readouterr().out
    )
    summary = pd.read_csv(out / "summary_unchanged.csv", index_col=0)
    assert {p.stem for p in folder.glob("*.csv")} == {n.replace(" ", "_") for n in summary.index}
    assert "conf" not in sbs.read_eval(out)[0]["unchanged"]
    assert (out / "per_perturbation" / "conf.csv").exists()  # the direction reports' folders are not touched


def test_existing_outputs_do_not_change(inputs, tmp_path):
    """The report is additive: every earlier evaluate.py output is the same with it, without it, however it is set."""
    plain = run_evaluate(inputs, tmp_path / "plain", "--no-unchanged-report")
    with_report = run_evaluate(inputs, tmp_path / "with")
    other = run_evaluate(
        inputs,
        tmp_path / "other",
        "--direction-only",
        "conf",
        "--unchanged-max-effect",
        "0.02",
        "--unchanged-min-padj",
        "0.9",
        "--min-unchanged",
        "3",
    )
    names = [*EXISTING]
    for folder in ("per_perturbation", "per_perturbation_specific"):
        tables = sorted(p.name for p in (plain / folder).glob("*.csv"))
        assert tables and tables == sorted(p.name for p in (with_report / folder).glob("*.csv"))
        names += [f"{folder}/{t}" for t in tables]
    for name in names:
        ref = (plain / name).read_bytes()
        assert (with_report / name).read_bytes() == ref and (other / name).read_bytes() == ref, name
    assert {p.name for p in plain.iterdir()} | {"summary_unchanged.csv", "per_perturbation_unchanged"} == {
        p.name for p in with_report.iterdir()
    }
    old_keys = ("truth_dir", "specific_ref", "control", "exclude_target", "eps", "min_degs", "methods")
    a, b = (json.loads((d / "methods.json").read_text()) for d in (plain, with_report))
    assert list(a)[: len(old_keys)] == list(old_keys) and all(a[k] == b[k] for k in old_keys)
    # and the full report is still what the (untouched) direction metrics give
    delta, _ = synthetic_truth()
    deg = pd.read_parquet(inputs / "truth" / "deg.parquet")
    weak = evaluate(pd.read_parquet(inputs / "weak.parquet"), delta, deg, exclude_genes={p: [p] for p in PERTS})
    got = pd.read_csv(with_report / "per_perturbation" / "weak.csv", index_col=0)
    pd.testing.assert_frame_equal(got, weak, check_dtype=False, check_names=False)
    summary = pd.read_csv(with_report / "summary.csv", index_col=0)
    assert summary.loc["weak", "balanced_acc_median"] == pytest.approx(summarize(weak)["balanced_acc_median"])


def test_other_consumers_ignore_the_new_files(inputs, tmp_path):
    """dashboard_data.py and freeze_results.py read the two direction reports; the extra files do not disturb them."""
    out = run_evaluate(inputs, tmp_path / "root" / "eval", "--direction-only", "conf")
    assert dd.find_runs(tmp_path / "root") == [out]
    run = dd.load_run(out)
    assert set(run.summary) == {"full", "specific"} and set(run.per_pert) == {"full", "specific"}
    assert run.methods == list(pd.read_csv(out / "summary.csv", index_col=0).index)
    assert "conf" in run.per_pert["full"] and run.manifest["unchanged"]["ceiling"] is True
    rc = freeze_results.main(
        ["--eval-dir", str(out), "--tag", "t", "--results-dir", str(tmp_path / "res"), "--reference", "model"]
        + ["--n-boot", "50"]
    )
    assert rc == 0
    assert {p.name for p in (tmp_path / "res").glob("t_*.csv")} == {
        f"t_summary{s}{k}.csv" for s in ("", "_specific") for k in ("", "_paired", "_raw")
    }


# ---------------------------------------------------------------------------
# side_by_side.py: the third report
# ---------------------------------------------------------------------------
SBS_PERTS = [f"P{i:02d}" for i in range(30)]
N_BOOT = 200
SHARED = SBS_PERTS[3:]  # see counts()


def counts(version: str) -> pd.DataFrame:
    """n_deg / n_unchanged per perturbation.

    filtered  P00: 3 DEGs (out); P01: 10 unchanged genes (out)           -> own panel P02..P29 (28)
    all       P02: 19 unchanged genes (out); P01 has 400                 -> own panel P00, P01, P03..P29 (29)
    shared    P03..P29 (27)
    """
    c = pd.DataFrame({"n_deg": 10, "n_unchanged": 300, "n_grey": 900}, index=SBS_PERTS)
    if version == "filtered":
        c.loc["P00", "n_deg"] = 3
        c.loc["P01", "n_unchanged"] = 10
    else:
        c.loc["P02", "n_unchanged"] = 19
        c.loc["P01", "n_unchanged"] = 400
    return c


def detection_frame(c: pd.DataFrame, auroc: np.ndarray, rng) -> pd.DataFrame:
    f = c.copy()
    f["detect_auroc"] = np.clip(auroc, 0, 1)
    f["detect_auroc_all"] = np.clip(auroc - 0.05, 0, 1)
    for i, col in enumerate(sbs.SECONDARY[:-1]):
        f[col] = np.clip(0.2 + 0.1 * i + rng.normal(0, 0.02, len(c)), 0, 1)
    return f.rename_axis("perturbation")


def eval_frames(version: str, seed: int = 0, third: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
    """{report: {stem: frame}}: full and specific (balanced_acc), and the changed-vs-unchanged report.

    The third report has no mlp (direction-only); its ceiling has too few unchanged genes in half A for every
    4th perturbation and too few DEGs for P05.
    """
    rng = np.random.default_rng(seed + (0 if version == "filtered" else 100))
    c = counts(version)
    out = {}
    for report, shift in (("full", 0.0), ("specific", -0.15)):
        base = 0.9 + shift + rng.normal(0, 0.03, len(c))
        methods = {"scLDM_s0": base, "knn": base + 0.01, "mlp": base - 0.03, "scLDM_s0_(shuffled)": base - 0.2}
        out[report] = {
            m: pd.DataFrame({"n_deg": c["n_deg"], "balanced_acc": np.clip(v, 0, 1)}).rename_axis("perturbation")
            for m, v in methods.items()
        }
    if third:
        base = 0.85 + rng.normal(0, 0.04, len(c))
        methods = {
            "scLDM_s0": base,
            "knn": base + 0.03 + rng.normal(0, 0.02, len(c)),
            "generic": base - 0.02 + rng.normal(0, 0.02, len(c)),
            "scLDM_s0_(shuffled)": base - 0.15 + rng.normal(0, 0.03, len(c)),
        }
        frames = {m: detection_frame(c, v, rng) for m, v in methods.items()}
        half = c.copy()
        half.iloc[::4, half.columns.get_loc("n_unchanged")] = 15
        half.loc["P05", "n_deg"] = 4
        frames["noise_ceiling_(split-half)"] = detection_frame(half, base + 0.1, rng)
        out["unchanged"] = frames
    return out


def build(evals: dict, **kw) -> dict:
    kw = {"reference": "scLDM_s0", "n_boot": N_BOOT, "seed": 0, **kw}
    return sbs.side_by_side(evals, {v: [] for v in evals}, **kw)


def cell(result, report, version, method, col):
    long = result["long"]
    row = long[(long.report == report) & (long.version == version) & (long.method == method)]
    assert len(row) == 1, (report, version, method)
    return row.iloc[0][col]


def direction_rows(result: dict, report: str) -> pd.DataFrame:
    """A direction report's rows of the methods that have one (generic and the ceiling: third report only)."""
    long = result["long"]
    rows = long[(long.report == report) & long.method.isin(["scLDM_s0", "knn", "mlp", "scLDM_s0_(shuffled)"])]
    return rows.sort_values("method").reset_index(drop=True)


def test_third_report_panel_needs_degs_and_unchanged_genes():
    evals = {"filtered": eval_frames("filtered"), "all": eval_frames("all")}
    res = build(evals)
    pan = res["panels"]["unchanged"]
    assert pan["shared"] == 27 and pan["own"] == {"filtered": 28, "all": 29}
    assert pan["only_in"] == {"filtered": ["P02"], "all": ["P00", "P01"]}
    assert pan["versions_without_report"] == [] and pan["n_deg_inconsistent"] == {}
    # the direction reports keep the n_deg rule alone: P01 and P02 stay in, whatever their unchanged genes
    for report in ("full", "specific"):
        assert res["panels"][report]["shared"] == 29 and res["panels"][report]["own"] == {"filtered": 29, "all": 30}
    panel = res["panel"]
    assert list(panel.columns) == [
        "report",
        "perturbation",
        "n_deg_filtered",
        "panel_filtered",
        "n_deg_all",
        "panel_all",
        "shared",
        "n_unchanged_filtered",
        "n_unchanged_all",
    ]
    third = panel[panel.report == "unchanged"].set_index("perturbation")
    assert set(third.index[third.shared]) == set(SHARED)
    assert third.loc["P01", "n_unchanged_filtered"] == 10 and not third.loc["P01", "panel_filtered"]
    assert third.loc["P01", "panel_all"] and third.loc["P00", "n_deg_filtered"] == 3
    assert panel[panel.report == "full"]["n_unchanged_all"].isna().all()  # a column of the third report only
    # a looser rule moves the panel: with --min-unchanged 10 P01 (filtered) and P02 (all) are in
    loose = build(evals, min_unchanged=10)["panels"]["unchanged"]
    assert loose["shared"] == 29 and loose["own"] == {"filtered": 29, "all": 30}


def test_third_report_statistics_are_stats_py_on_the_shared_panel():
    evals = {"filtered": eval_frames("filtered"), "all": eval_frames("all")}
    res = build(evals)
    assert res["metrics"] == {"full": "balanced_acc", "specific": "balanced_acc", "unchanged": "detect_auroc"}
    for v in ("filtered", "all"):
        frames = evals[v]["unchanged"]
        s0 = frames["scLDM_s0"]["detect_auroc"].loc[SHARED]
        assert cell(res, "unchanged", v, "scLDM_s0", "n") == 27
        assert cell(res, "unchanged", v, "scLDM_s0", "median") == pytest.approx(np.median(s0))
        lo, hi = stats.bootstrap_ci(s0.to_numpy(), n_boot=N_BOOT, seed=0)
        assert (cell(res, "unchanged", v, "scLDM_s0", "ci_low"), cell(res, "unchanged", v, "scLDM_s0", "ci_high")) == (
            lo,
            hi,
        )
        knn = frames["knn"]["detect_auroc"].loc[SHARED]
        ref = stats.paired_difference(s0, knn, n_boot=N_BOOT, seed=0)
        assert cell(res, "unchanged", v, "knn", "n_paired") == 27
        assert cell(res, "unchanged", v, "knn", "diff") == pytest.approx(ref["median_difference"])
        assert cell(res, "unchanged", v, "knn", "p_raw") == pytest.approx(ref["p_raw"])
        # own panel: this version's own rule
        own = frames["scLDM_s0"]
        own = own[(own.n_deg >= 5) & (own.n_unchanged >= 20)]["detect_auroc"]
        assert cell(res, "unchanged", v, "scLDM_s0", "n_own") == len(own) == res["panels"]["unchanged"]["own"][v]
        assert cell(res, "unchanged", v, "scLDM_s0", "median_own") == pytest.approx(np.median(own))
        # Holm within the (report, version) column
        long = res["long"]
        col = long[(long.report == "unchanged") & (long.version == v) & ~long.reference].dropna(subset=["p_raw"])
        assert len(col) == 4
        np.testing.assert_allclose(col.p_holm.to_numpy(), stats.holm(col.p_raw).to_numpy())
        # the ceiling needs both counts in its own half too
        ceiling = frames["noise_ceiling_(split-half)"].loc[SHARED]
        ceiling = ceiling[(ceiling.n_deg >= 5) & (ceiling.n_unchanged >= 20)]
        assert cell(res, "unchanged", v, "noise_ceiling_(split-half)", "n") == len(ceiling) < 27
        assert cell(res, "unchanged", v, "noise_ceiling_(split-half)", "median") == pytest.approx(
            ceiling["detect_auroc"].median()
        )
        # direction-only mlp: in the direction reports, missing here, and named
        assert cell(res, "full", v, "mlp", "status") == "ok"
        assert cell(res, "unchanged", v, "mlp", "status") == "missing"
        assert res["notes"]["unchanged"][v]["missing_methods"] == ["mlp"]
        assert cell(res, "full", v, "generic", "status") == "missing"  # this fixture has it in the third report only
    # the direction reports are the same numbers with or without the third report in the eval dirs
    without = build({v: {r: f for r, f in e.items() if r != "unchanged"} for v, e in evals.items()})
    for report in ("full", "specific"):
        pd.testing.assert_frame_equal(direction_rows(res, report), direction_rows(without, report))
    # --unchanged-metric switches the third report's headline only
    alt = build(evals, unchanged_metric="detect_auroc_all")
    assert cell(alt, "unchanged", "filtered", "scLDM_s0", "median") == pytest.approx(
        cell(res, "unchanged", "filtered", "scLDM_s0", "median") - 0.05
    )
    assert cell(alt, "full", "filtered", "scLDM_s0", "median") == cell(res, "full", "filtered", "scLDM_s0", "median")


def test_third_report_secondary_medians():
    evals = {"filtered": eval_frames("filtered"), "all": eval_frames("all")}
    res = build(evals)
    assert set(res["secondary"]) == {"filtered", "all"}
    for v in ("filtered", "all"):
        table = res["secondary"][v]
        assert list(table.columns) == ["n", *sbs.SECONDARY] and "mlp" not in table.index
        knn = evals[v]["unchanged"]["knn"].loc[SHARED]
        assert table.loc["knn", "n"] == 27
        for c in sbs.SECONDARY:
            assert table.loc["knn", c] == pytest.approx(knn[c].median()), c
        ceiling = evals[v]["unchanged"]["noise_ceiling_(split-half)"].loc[SHARED]
        ceiling = ceiling[(ceiling.n_deg >= 5) & (ceiling.n_unchanged >= 20)]
        assert table.loc["noise_ceiling_(split-half)", "n"] == len(ceiling)
        assert table.loc["noise_ceiling_(split-half)", "deg_hit_rate"] == pytest.approx(
            ceiling["deg_hit_rate"].median()
        )
    md = sbs.markdown(res, primary="filtered", title="t", min_degs=5, n_boot=N_BOOT, seed=0)
    section = md[md.index("## Changed vs unchanged genes") :]
    assert "Shared panel: **27** perturbations (filtered own 28, all own 29)" in section
    assert ">= 5 scored DEGs **and** >= 20 unchanged genes" in section
    header = "| method | n | " + " | ".join(f"`{c}`" for c in sbs.SECONDARY) + " |"
    assert section.count(header) == 2 and "filtered (primary):" in section
    knn = res["secondary"]["all"].loc["knn"]
    assert f"| knn | 27 | {knn['three_class_balanced_acc']:.3f} | {knn['deg_call_rate']:.3f} |" in section
    assert "- not scored in filtered: mlp" in section and "- not scored in all: mlp" in section
    assert "## Changed vs unchanged genes" not in md[: md.index("## Perturbation-specific")]  # the third section
    assert "no split-half ceiling" not in md and "not a DEG, with a small |Δ|" in md  # no recorded settings given
    # with the eval dirs' recorded settings: the thresholds, tau, what was left out, a missing ceiling
    recorded = {"filtered": {**SETTINGS, "ceiling": False}, "all": SETTINGS}
    md = sbs.markdown(res, primary="filtered", title="t", min_degs=5, n_boot=N_BOOT, seed=0, unchanged=recorded)
    assert "not a DEG, |Δ| < 0.05, padj ≥ 0.5" in md and "(τ = 0.1)" in md and "not an effect size (mlp)" in md
    assert md.count("no split-half ceiling") == 1 and "- filtered: no split-half ceiling" in md


def test_a_version_without_the_third_report_is_marked_missing():
    evals = {"filtered": eval_frames("filtered"), "all": eval_frames("all", third=False)}  # "all": an older dir
    res = build(evals)
    pan = res["panels"]["unchanged"]
    # the shared panel is taken over the versions that have the report
    assert pan["versions_without_report"] == ["all"] and pan["own"] == {"filtered": 28} and pan["shared"] == 28
    long = res["long"]
    rows = long[(long.report == "unchanged") & (long.version == "all")]
    assert len(rows) and (rows.status == "missing").all() and rows["median"].isna().all()
    assert res["notes"]["unchanged"]["all"]["report_missing"] and "all" not in res["secondary"]
    assert cell(res, "unchanged", "filtered", "scLDM_s0", "status") == "ok"
    assert cell(res, "unchanged", "filtered", "scLDM_s0", "n") == 28
    assert not np.isnan(cell(res, "unchanged", "filtered", "knn", "p_holm"))
    third = res["panel"][res["panel"].report == "unchanged"]
    assert third["n_deg_all"].isna().all() and not third["panel_all"].any()
    # the direction reports are untouched by the missing report
    both = build({"filtered": eval_frames("filtered"), "all": eval_frames("all")})
    for report in ("full", "specific"):
        pd.testing.assert_frame_equal(direction_rows(res, report), direction_rows(both, report))
    md = sbs.markdown(res, primary="filtered", title="t", min_degs=5, n_boot=N_BOOT, seed=0)
    section = md[md.index("## Changed vs unchanged genes") :]
    assert "Shared panel: **28** perturbations (filtered own 28, all: no report)" in section
    assert f"- all: {sbs.REPORT_MISSING}" in section and "- not scored in all" not in section
    assert "| scLDM_s0 | 28 |" in section  # the own-panel table: filtered scored, all "—"
    wide = sbs.wide(res["long"], "unchanged", res["versions"])
    assert (wide["all_status"] == "missing").all() and wide.loc["scLDM_s0", "filtered_status"] == "ok"

    # no version has it (two older eval dirs): rows marked missing, a section that says so
    res = build({"filtered": eval_frames("filtered", third=False), "all": eval_frames("all", third=False)})
    assert res["panels"]["unchanged"]["shared"] == 0 and res["panels"]["unchanged"]["own"] == {}
    rows = res["long"][res["long"].report == "unchanged"]
    assert len(rows) == 2 * len(res["order"]) and (rows.status == "missing").all()
    assert res["panel"][res["panel"].report == "unchanged"].empty and res["secondary"] == {}
    md = sbs.markdown(res, primary="filtered", title="t", min_degs=5, n_boot=N_BOOT, seed=0)
    assert f"Not available: {sbs.REPORT_MISSING}." in md[md.index("## Changed vs unchanged genes") :]


def write_eval(d: Path, frames: dict, unchanged: dict | None | str = "absent") -> Path:
    """An evaluate.py output dir with the given reports; ``unchanged``: the methods.json block ("absent": no key)."""
    for report, suffix in sbs.REPORTS.items():
        if report not in frames:
            continue
        (d / f"per_perturbation{suffix}").mkdir(parents=True, exist_ok=True)
        for stem, f in frames[report].items():
            f.to_csv(d / f"per_perturbation{suffix}" / f"{stem}.csv")
        pd.DataFrame({"n_perturbations": [1]}, index=["x"]).to_csv(d / f"summary{suffix}.csv")
    methods = [
        {"name": "scLDM_s0", "kind": "pred", "path": str(d / "s0.parquet")},
        {"name": "scLDM_s0 (shuffled)", "kind": "shuffled", "source": "scLDM_s0", "seed": 0},
        *({"name": n, "kind": "baseline", "path": str(d / f"{n}.parquet")} for n in ("knn", "mlp", "generic")),
    ]
    manifest = {"control": "non-targeting", "exclude_target": True, "eps": 0.0, "min_degs": 5, "methods": methods}
    if unchanged != "absent":
        manifest["unchanged"] = unchanged
    (d / "methods.json").write_text(json.dumps(manifest))
    return d


SETTINGS = {
    "max_effect": 0.05,
    "min_padj": 0.5,
    "min_unchanged": 20,
    "tau": 0.1,
    "methods": ["scLDM_s0", "scLDM_s0 (shuffled)", "knn", "generic"],
    "direction_only_excluded": ["mlp"],
    "ceiling": True,
}


def table_args(dirs: dict[str, Path], out: Path, *extra: str) -> list[str]:
    args = ["table", "--primary", "filtered", "--tag", "t", "--out-dir", str(out), "--n-boot", str(N_BOOT)]
    for v, d in dirs.items():
        args += ["--eval", f"{v}={d}"]
    return [*args, *extra]


def test_table_cli_third_report(tmp_path):
    dirs = {v: write_eval(tmp_path / v, eval_frames(v), SETTINGS) for v in ("filtered", "all")}
    out = tmp_path / "results"
    assert sbs.main(table_args(dirs, out)) == 0
    assert {p.name for p in out.iterdir()} == {
        "t_side_by_side.csv",
        "t_side_by_side_specific.csv",
        "t_side_by_side_unchanged.csv",
        "t_side_by_side_long.csv",
        "t_side_by_side_panel.csv",
        "t_side_by_side.md",
        "t.provenance.json",
    }
    wide = pd.read_csv(out / "t_side_by_side_unchanged.csv", index_col=0)
    # every method of the table, in the table's order; the direction-only one is marked missing
    assert list(wide.index) == [
        "scLDM_s0",
        "knn",
        "mlp",
        "generic",
        "scLDM_s0_(shuffled)",
        "noise_ceiling_(split-half)",
    ]
    assert wide.loc["mlp", "filtered_status"] == "missing" and wide.loc["mlp", "all_status"] == "missing"
    assert wide.loc["knn", "filtered_n"] == 27 and 0 < wide.loc["knn", "filtered_p_holm"] <= 1
    full = pd.read_csv(out / "t_side_by_side.csv", index_col=0)
    assert list(full.columns) == list(wide.columns) and full.loc["mlp", "filtered_status"] == "ok"
    long = pd.read_csv(out / "t_side_by_side_long.csv")
    assert set(long.report) == {"full", "specific", "unchanged"}
    assert set(pd.read_csv(out / "t_side_by_side_panel.csv").report) == {"full", "specific", "unchanged"}
    md = (out / "t_side_by_side.md").read_text()
    assert md.index("## Full effect") < md.index("## Perturbation-specific") < md.index("## Changed vs unchanged")
    assert "not a DEG, |Δ| < 0.05, padj ≥ 0.5" in md and "τ = 0.1" in md
    assert "not an effect size (mlp)" in md and "**scLDM_s0** (reference)" in md
    prov = json.loads((out / "t.provenance.json").read_text())
    assert prov["metric"] == "balanced_acc" and prov["metrics"]["unchanged"] == "detect_auroc"
    assert prov["min_degs"] == 5 and prov["min_unchanged"] == 20
    assert prov["unchanged"] == {"filtered": SETTINGS, "all": SETTINGS}
    assert prov["panels"]["unchanged"]["shared"] == 27 and prov["missing_methods"]["unchanged"]["all"] == ["mlp"]
    table = dirs["all"] / "per_perturbation_unchanged" / "knn.csv"
    assert str(table) in prov["inputs"] and str(dirs["all"] / "summary_unchanged.csv") in prov["inputs"]
    assert "t_side_by_side_unchanged.csv" in prov["tables"]

    # the options of the third report
    out2 = tmp_path / "results2"
    assert sbs.main(table_args(dirs, out2, "--unchanged-metric", "detect_auroc_all", "--min-unchanged", "10")) == 0
    wide2 = pd.read_csv(out2 / "t_side_by_side_unchanged.csv", index_col=0)
    assert (
        wide2.loc["knn", "filtered_n"] == 29
        and wide2.loc["knn", "filtered_median"] < wide.loc["knn", "filtered_median"]
    )
    pd.testing.assert_frame_equal(pd.read_csv(out2 / "t_side_by_side.csv", index_col=0), full)  # --metric's reports
    prov2 = json.loads((out2 / "t.provenance.json").read_text())
    assert prov2["metrics"]["unchanged"] == "detect_auroc_all" and prov2["min_unchanged"] == 10

    # the frozen comparison stays full|specific
    frozen = tmp_path / "frozen.csv"
    pd.DataFrame({"n_perturbations": [1.0], "median": [0.5]}, index=["scLDM_s0"]).to_csv(frozen)
    with pytest.raises(SystemExit, match="full"):
        sbs.main(table_args(dirs, tmp_path / "r3", "--frozen-summary", f"unchanged={frozen}"))


def test_table_cli_with_older_eval_dirs(tmp_path):
    """An eval dir without the report (no tables, no methods.json key) and one whose tables are stale."""
    new = write_eval(tmp_path / "filtered", eval_frames("filtered"), SETTINGS)
    old = write_eval(tmp_path / "all", eval_frames("all", third=False))
    out = tmp_path / "results"
    assert sbs.main(table_args({"filtered": new, "all": old}, out)) == 0
    wide = pd.read_csv(out / "t_side_by_side_unchanged.csv", index_col=0)
    assert (wide["all_status"] == "missing").all() and wide.loc["knn", "filtered_status"] == "ok"
    assert wide.loc["knn", "filtered_n"] == 28  # the filtered own panel: nothing to share it with
    prov = json.loads((out / "t.provenance.json").read_text())
    assert prov["unchanged"] == {"filtered": SETTINGS, "all": None}
    assert prov["panels"]["unchanged"]["versions_without_report"] == ["all"]
    assert "all: no report" in (out / "t_side_by_side.md").read_text()
    # both older: the tables of the direction reports are what they were, the third is all missing
    older = write_eval(tmp_path / "filtered_old", eval_frames("filtered", third=False))
    assert sbs.main(table_args({"filtered": older, "all": old}, tmp_path / "r2")) == 0
    assert (pd.read_csv(tmp_path / "r2" / "t_side_by_side_unchanged.csv")["filtered_status"] == "missing").all()
    for name in ("t_side_by_side.csv", "t_side_by_side_specific.csv"):
        a, b = pd.read_csv(tmp_path / "r2" / name, index_col=0), pd.read_csv(out / name, index_col=0)
        pd.testing.assert_frame_equal(a, b.drop(index="noise_ceiling_(split-half)"))  # a third-report row

    # a rerun of evaluate.py without the report (methods.json: null) leaves stale tables behind: not read
    stale = write_eval(tmp_path / "stale", eval_frames("all"), None)
    assert (stale / "per_perturbation_unchanged" / "knn.csv").exists()
    frames, _ = sbs.read_eval(stale)
    assert set(frames) == {"full", "specific"}
    assert set(sbs.read_eval(new)[0]) == {"full", "specific", "unchanged"}
    assert set(sbs.read_eval(write_eval(tmp_path / "nokey", eval_frames("all")))[0]) == set(sbs.REPORTS)


def test_table_refuses_versions_with_different_unchanged_settings(tmp_path):
    f = write_eval(tmp_path / "filtered", eval_frames("filtered"), SETTINGS)
    for key, value in (("max_effect", 0.1), ("min_padj", 0.3), ("min_unchanged", 5), ("tau", 0.2)):
        a = write_eval(tmp_path / f"all_{key}", eval_frames("all"), {**SETTINGS, key: value})
        with pytest.raises(SystemExit, match=f"changed-vs-unchanged settings differ.*{key}"):
            sbs.main(table_args({"filtered": f, "all": a}, tmp_path / "out"))
    # what a version left out as direction-only is not a threshold: reported per version, not refused
    a = write_eval(tmp_path / "all_ok", eval_frames("all"), {**SETTINGS, "direction_only_excluded": []})
    assert sbs.check_unchanged_settings({"filtered": f, "all": a})["all"]["direction_only_excluded"] == []
