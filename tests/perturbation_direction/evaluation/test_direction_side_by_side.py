"""The raw Replogle evaluation stage: side_by_side.py (shared panel, stats, provenance) and its runner.

Unit tests use synthetic evaluate.py output dirs (per-perturbation CSVs + methods.json). Panels, per report:

  filtered  P00 has 3 DEGs (out), P01..P29 have 10           -> own panel P01..P29 (29)
  all       P00 has 8 DEGs (in), P01 has 2 (out), P02 absent -> own panel P00, P03..P29 (28)
  shared    P03..P29 (27)

These synthetic dirs hold the two direction reports only, as an eval dir written before evaluate.py had the
changed-vs-unchanged report does: that report is then missing for every version, and nothing else changes.
test_direction_unchanged.py covers the third report itself.

The end-to-end test runs run_replogle_raw_eval.sh on tiny synthetic prepared files (3 lines, 40 genes, 8 test
perturbations) with fast settings. The resume test runs the same pipeline once more, from a copy of the pipeline
directory, so that it can edit a step's script.
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.evaluation import side_by_side as sbs
from experiments.perturbation_direction.evaluation import stats
from omegaconf import OmegaConf
from scipy import sparse

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "experiments" / "perturbation_direction" / "runners" / "run_replogle_raw_eval.sh"
EVAL_CONFIG = REPO / "experiments" / "configs" / "perturbation_direction" / "replogle_raw_eval.yaml"
PERTS = [f"P{i:02d}" for i in range(30)]
N_BOOT = 200
DIRECTION_REPORTS = ("full", "specific")  # what method_frames() builds; "unchanged" is tested in its own file


# ---------------------------------------------------------------------------
# synthetic evaluate.py output dirs
# ---------------------------------------------------------------------------
def n_degs(version: str) -> pd.Series:
    n = pd.Series(10, index=PERTS)
    if version == "filtered":
        n["P00"] = 3
    else:
        n["P00"], n["P01"] = 8, 2
        n = n.drop("P02")
    return n


def frame(values: np.ndarray, n_deg: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"n_deg": n_deg.to_numpy(), "balanced_acc": values}, index=n_deg.index).rename_axis(
        "perturbation"
    )


def method_frames(version: str, seed: int = 0, drop: tuple[str, ...] = ()) -> dict[str, dict[str, pd.DataFrame]]:
    """{report: {stem: frame}}: two seeds, knn, mlp, generic (full only), a shuffled control, the ceiling."""
    rng = np.random.default_rng(seed + (0 if version == "filtered" else 100))
    n = n_degs(version)
    out = {}
    for report, shift in (("full", 0.0), ("specific", -0.15)):
        base = 0.9 + shift + rng.normal(0, 0.03, len(n))
        methods = {
            "scLDM_s0": base,
            "scLDM_s1": base + rng.normal(0, 0.01, len(n)),
            "knn": base + 0.01 + rng.normal(0, 0.02, len(n)),
            "mlp": base - 0.03 + rng.normal(0, 0.02, len(n)),
            "scLDM_s0_(shuffled)": base - 0.2 + rng.normal(0, 0.05, len(n)),
        }
        if report == "full":
            methods["generic"] = base - 0.06 + rng.normal(0, 0.02, len(n))
        frames = {m: frame(np.clip(v, 0, 1), n) for m, v in methods.items() if m not in drop}
        ceiling_n = n.copy()
        ceiling_n.iloc[::4] = 4  # every 4th perturbation has too few DEGs in its half
        frames["noise_ceiling_(split-half)"] = frame(np.clip(base + 0.04, 0, 1), ceiling_n)
        out[report] = frames
    return out


def write_eval(d: Path, frames: dict[str, dict[str, pd.DataFrame]], files: Path | None = None) -> Path:
    """An evaluate.py output dir: per_perturbation[_specific|_unchanged]/<stem>.csv, summary csvs, methods.json.

    Only the reports in ``frames`` are written (the changed-vs-unchanged report is optional in an eval dir).
    """
    for report, suffix in sbs.REPORTS.items():
        if report not in frames:
            continue
        (d / f"per_perturbation{suffix}").mkdir(parents=True, exist_ok=True)
        for stem, f in frames[report].items():
            f.to_csv(d / f"per_perturbation{suffix}" / f"{stem}.csv")
        pd.DataFrame({"n_perturbations": [1]}, index=["x"]).to_csv(d / f"summary{suffix}.csv")
    files = files or d / "inputs"
    files.mkdir(parents=True, exist_ok=True)
    (files / "truth").mkdir(exist_ok=True)
    (files / "truth" / "delta_true.parquet").write_bytes(b"truth-" + d.name.encode())

    def entry(name):
        p = files / f"{name}.parquet"
        p.write_bytes(f"{d.name}-{name}".encode())
        return {"path": str(p), "abs_path": str(p.resolve())}

    stems = set().union(*(set(f) for f in frames.values()))
    methods = []
    for name in ("scLDM_s0", "scLDM_s1"):
        if name in stems:
            methods.append({"name": name, "kind": "pred", **entry(name)})
        if f"{name}_(shuffled)" in stems:
            methods.append({"name": f"{name} (shuffled)", "kind": "shuffled", "source": name, "seed": 0})
    for name in ("generic", "knn", "mlp"):  # deliberately not the table order
        if name in stems:
            methods.append({"name": name, "kind": "baseline", **entry(name)})
    manifest = {
        "truth_dir": {"path": str(files / "truth"), "abs_path": str((files / "truth").resolve())},
        "specific_ref": entry("generic_ref"),
        "control": "non-targeting",
        "exclude_target": True,
        "methods": methods,
    }
    (d / "methods.json").write_text(json.dumps(manifest))
    return d


def build(evals: dict[str, dict], **kw) -> dict:
    manifests = {v: [] for v in evals}
    kw = {"reference": "scLDM_s0", "n_boot": N_BOOT, "seed": 0, **kw}
    return sbs.side_by_side(evals, manifests, **kw)


def cell(result, report, version, method, col):
    long = result["long"]
    row = long[(long.report == report) & (long.version == version) & (long.method == method)]
    assert len(row) == 1, (report, version, method)
    return row.iloc[0][col]


SHARED = set(PERTS[3:])


# ---------------------------------------------------------------------------
# panels and statistics
# ---------------------------------------------------------------------------
def test_shared_panel_and_own_panels():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all")}
    res = build(evals)
    assert set(res["panels"]) == set(sbs.REPORTS) == {*DIRECTION_REPORTS, "unchanged"}
    for report in DIRECTION_REPORTS:
        pan = res["panels"][report]
        assert pan["shared"] == 27 and pan["own"] == {"filtered": 29, "all": 28}
        assert pan["versions_without_report"] == []
        assert pan["only_in"] == {"filtered": ["P01", "P02"], "all": ["P00"]}
        assert pan["absent_from"] == {"filtered": [], "all": ["P02"]}
        for v in ("filtered", "all"):
            values = evals[v][report]["scLDM_s0"]["balanced_acc"]
            assert cell(res, report, v, "scLDM_s0", "n") == 27
            assert cell(res, report, v, "scLDM_s0", "median") == pytest.approx(np.median(values.loc[sorted(SHARED)]))
            own = values[evals[v][report]["scLDM_s0"]["n_deg"] >= 5]
            assert cell(res, report, v, "scLDM_s0", "n_own") == len(own)
            assert cell(res, report, v, "scLDM_s0", "median_own") == pytest.approx(np.median(own))
    panel = res["panel"]
    full = panel[panel.report == "full"].set_index("perturbation")
    assert full.loc["P02", "panel_filtered"] and not full.loc["P02", "panel_all"]
    assert np.isnan(full.loc["P02", "n_deg_all"]) and full.loc["P00", "n_deg_all"] == 8
    assert set(full.index[full.shared]) == SHARED


def test_ci_and_paired_tests_reuse_stats_on_the_shared_panel():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all")}
    res = build(evals)
    for report in DIRECTION_REPORTS:
        for v in ("filtered", "all"):
            frames = evals[v][report]
            s0 = frames["scLDM_s0"]["balanced_acc"].loc[sorted(SHARED)]
            lo, hi = stats.bootstrap_ci(s0.to_numpy(), n_boot=N_BOOT, seed=0)
            assert (cell(res, report, v, "scLDM_s0", "ci_low"), cell(res, report, v, "scLDM_s0", "ci_high")) == (lo, hi)
            knn = frames["knn"]["balanced_acc"].loc[sorted(SHARED)]
            ref = stats.paired_difference(s0, knn, n_boot=N_BOOT, seed=0)
            assert cell(res, report, v, "knn", "n_paired") == 27
            assert cell(res, report, v, "knn", "diff") == pytest.approx(ref["median_difference"])
            assert cell(res, report, v, "knn", "diff_ci_low") == pytest.approx(ref["ci_low"])
            assert cell(res, report, v, "knn", "p_raw") == pytest.approx(ref["p_raw"])
            assert np.isnan(cell(res, report, v, "scLDM_s0", "p_raw"))  # the reference has no test of its own


def test_holm_is_applied_within_each_report_and_version_column():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all")}
    res = build(evals)
    long = res["long"]
    for _, col in long[~long.reference & long.report.isin(DIRECTION_REPORTS)].groupby(["report", "version"]):
        col = col.dropna(subset=["p_raw"])
        assert len(col) >= 4
        np.testing.assert_allclose(col.p_holm.to_numpy(), stats.holm(col.p_raw).to_numpy())
    # changing the 'all' version leaves every filtered number untouched
    evals2 = {"filtered": method_frames("filtered"), "all": method_frames("all", seed=7)}
    res2 = build(evals2)
    a = res["long"][res["long"].version == "filtered"].reset_index(drop=True)
    b = res2["long"][res2["long"].version == "filtered"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    c = res["long"][res["long"].version == "all"].p_holm.to_numpy()
    d = res2["long"][res2["long"].version == "all"].p_holm.to_numpy()
    assert not np.allclose(np.nan_to_num(c), np.nan_to_num(d))


def test_ceiling_is_restricted_to_the_panel_and_its_own_degs():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all")}
    res = build(evals)
    for v in ("filtered", "all"):
        ceiling = evals[v]["full"]["noise_ceiling_(split-half)"]
        expected = ceiling.loc[sorted(SHARED)]
        expected = expected[expected.n_deg >= 5]
        assert cell(res, "full", v, "noise_ceiling_(split-half)", "n") == len(expected) < 27
        assert cell(res, "full", v, "noise_ceiling_(split-half)", "n_own") == int((ceiling.n_deg >= 5).sum())
    # the ceiling never defines the panel, even with DEG counts the methods do not have
    assert res["panels"]["full"]["n_deg_inconsistent"] == {}


def test_a_method_missing_from_one_version_is_reported_not_dropped():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all", drop=("mlp",))}
    res = build(evals)
    for report in DIRECTION_REPORTS:
        assert cell(res, report, "all", "mlp", "status") == "missing"
        assert np.isnan(cell(res, report, "all", "mlp", "median"))
        assert cell(res, report, "filtered", "mlp", "status") == "ok"
        assert "mlp" in res["notes"][report]["all"]["missing_methods"]
        # the 'all' Holm family has one comparison fewer
        long = res["long"]
        n_f = long[(long.report == report) & (long.version == "filtered")].p_raw.notna().sum()
        n_a = long[(long.report == report) & (long.version == "all")].p_raw.notna().sum()
        assert n_f == n_a + 1
    # generic has no perturbation-specific part in either version: a row, missing in both
    assert cell(res, "specific", "filtered", "generic", "status") == "missing"
    assert cell(res, "specific", "all", "generic", "status") == "missing"
    md = sbs.markdown(res, primary="filtered", title="t", min_degs=5, n_boot=N_BOOT, seed=0)
    # without methods.json the order is by name: full misses mlp, specific misses generic and mlp
    assert "- not scored in all: mlp\n" in md and "- not scored in all: generic, mlp\n" in md
    assert "| mlp | 27 |" in md


def test_missing_reference_and_unscored_method_do_not_crash():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all", drop=("scLDM_s0",))}
    res = build(evals)
    assert np.isnan(cell(res, "full", "all", "knn", "p_raw"))
    assert any("reference scLDM_s0 missing" in n for n in res["notes"]["full"]["all"]["notes"])
    assert not np.isnan(cell(res, "full", "filtered", "knn", "p_raw"))
    # a method that scores none of the shared perturbations: n 0, no test, kept out of the Holm family
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all")}
    evals["all"]["full"]["knn"] = evals["all"]["full"]["knn"].loc[["P00"]]
    res = build(evals)
    assert cell(res, "full", "all", "knn", "n") == 0 and np.isnan(cell(res, "full", "all", "knn", "p_raw"))
    col = res["long"][(res["long"].report == "full") & (res["long"].version == "all")].dropna(subset=["p_raw"])
    np.testing.assert_allclose(col.p_holm.to_numpy(), stats.holm(col.p_raw).to_numpy())


def test_inconsistent_deg_counts_are_reported():
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all")}
    evals["filtered"]["full"]["knn"] = evals["filtered"]["full"]["knn"].assign(
        n_deg=lambda f: f.n_deg.where(f.index != "P05", 9)
    )
    res = build(evals)
    assert res["panels"]["full"]["n_deg_inconsistent"] == {"filtered": ["P05"]}


def test_method_order_follows_the_table_convention(tmp_path):
    frames = {v: method_frames(v) for v in ("filtered", "all")}
    for v in frames:
        frames[v]["full"]["transfer"] = frames[v]["full"]["knn"]
        frames[v]["full"]["zzz_extra"] = frames[v]["full"]["knn"]
    manifests = {}
    for v in frames:
        _, manifests[v] = sbs.read_eval(write_eval(tmp_path / v, frames[v]))
    manifests["all"].append({"name": "transfer", "kind": "baseline", "path": "x"})
    order, kinds = sbs.method_order(frames, manifests)
    assert order == [
        "scLDM_s0",
        "scLDM_s1",
        "knn",
        "mlp",
        "generic",
        "transfer",
        "scLDM_s0_(shuffled)",
        "noise_ceiling_(split-half)",
        "zzz_extra",
    ]
    assert kinds["noise_ceiling_(split-half)"] == "ceiling" and kinds["zzz_extra"] == "other"
    assert sbs.label("scLDM_s0_(shuffled)") == "scLDM_s0 (shuffled)"
    assert sbs.label("noise_ceiling_(split-half)") == "noise ceiling (split-half)"


# ---------------------------------------------------------------------------
# CLI: files, markdown, provenance
# ---------------------------------------------------------------------------
def test_table_cli_writes_tables_markdown_and_provenance(tmp_path):
    evals = {"filtered": method_frames("filtered"), "all": method_frames("all", drop=("mlp",))}
    dirs = {v: write_eval(tmp_path / f"eval_{v}", f) for v, f in evals.items()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"versions": ["filtered", "all"]}')
    frozen = tmp_path / "frozen_summary.csv"
    pd.DataFrame(
        {"n_perturbations": [29.0, 5.0], "median": [0.5, 0.9], "ci_low": [0.4, 0.8], "ci_high": [0.6, 1.0]},
        index=["scLDM_s0", "calibrated"],
    ).to_csv(frozen)
    out = tmp_path / "results"
    rc = sbs.main(
        [
            "table",
            "--eval",
            f"all={dirs['all']}",  # order on the command line does not matter: the primary goes first
            "--eval",
            f"filtered={dirs['filtered']}",
            "--primary",
            "filtered",
            "--tag",
            "t",
            "--out-dir",
            str(out),
            "--n-boot",
            str(N_BOOT),
            "--timestamp",
            "2026-11-10T12:00:00Z",
            "--input",
            f"manifest={manifest}",
            "--frozen-summary",
            f"full={frozen}",
            "--config",
            str(EVAL_CONFIG),
            "--override",
            "prepared_dir=/tmp/somewhere",
        ]
    )
    assert rc == 0
    names = {p.name for p in out.iterdir()}
    assert names == {
        "t_side_by_side.csv",
        "t_side_by_side_specific.csv",
        "t_side_by_side_unchanged.csv",
        "t_side_by_side_long.csv",
        "t_side_by_side_panel.csv",
        "t_side_by_side_vs_frozen.csv",
        "t_side_by_side.md",
        "t.provenance.json",
    }
    wide = pd.read_csv(out / "t_side_by_side.csv", index_col=0)
    assert list(wide.columns[:3]) == ["label", "filtered_status", "filtered_n"]
    assert wide.loc["mlp", "all_status"] == "missing" and wide.loc["knn", "all_n"] == 27
    assert wide.index[0] == "scLDM_s0" and wide.index[-1] == "noise_ceiling_(split-half)"
    md = (out / "t_side_by_side.md").read_text()
    assert "filtered (primary) n" in md and md.index("filtered (primary)") < md.index("| all n")
    assert "Shared panel: **27** perturbations (filtered own 29, all own 28)" in md
    assert "**scLDM_s0** (reference)" in md and "calibrated (full, frozen only)" in md
    vs = pd.read_csv(out / "t_side_by_side_vs_frozen.csv").set_index("method")
    assert vs.loc["scLDM_s0", "own_n"] == 29 and vs.loc["scLDM_s0", "frozen_median"] == 0.5

    prov = json.loads((out / "t.provenance.json").read_text())
    assert prov["timestamp"] == "2026-11-10T12:00:00Z"
    assert set(prov["git"]) == {"commit", "dirty", "changed_paths", "ignored_output_paths"} and isinstance(
        prov["git"]["dirty"], bool
    )
    assert prov["versions"] == ["filtered", "all"] and prov["primary"] == "filtered"
    pred = dirs["filtered"] / "inputs" / "scLDM_s0.parquet"
    assert prov["inputs"][str(pred)] == hashlib.sha256(pred.read_bytes()).hexdigest()
    truth = dirs["all"] / "inputs" / "truth" / "delta_true.parquet"
    assert prov["inputs"][str(truth)] == hashlib.sha256(truth.read_bytes()).hexdigest()
    table = dirs["all"] / "per_perturbation_specific" / "knn.csv"
    assert prov["inputs"][str(table)] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert prov["inputs"][str(frozen)] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert prov["extra_inputs"]["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert prov["config"]["resolved"]["out_dir"] == "/tmp/somewhere/eval"
    assert prov["config"]["sha256"] == hashlib.sha256(EVAL_CONFIG.read_bytes()).hexdigest()
    assert prov["code"]["side_by_side.py"] and prov["code"]["stats.py"]
    assert prov["panels"]["full"]["shared"] == 27
    assert prov["missing_methods"]["full"]["all"] == ["mlp"]
    assert set(prov["tables"]) == names


def test_table_cli_refuses_unknown_primary(tmp_path):
    d = write_eval(tmp_path / "e", method_frames("filtered"))
    with pytest.raises(SystemExit):
        sbs.main(["table", "--eval", f"filtered={d}", "--primary", "all", "--tag", "t", "--out-dir", str(tmp_path)])


@pytest.mark.parametrize(
    "change, message",
    [({"min_degs": 3}, "differ"), ({"eps": 0.1}, "differ"), ({"exclude_target": False}, "differ")],
)
def test_table_refuses_versions_scored_with_different_settings(tmp_path, change, message):
    """n_deg (and so the shared panel) depends on evaluate.py's settings; they must agree across versions."""
    f = write_eval(tmp_path / "f", method_frames("filtered"))
    a = write_eval(tmp_path / "a", method_frames("all"))
    for d in (f, a):
        m = json.loads((d / "methods.json").read_text())
        m.update({"eps": 0.0, "min_degs": 5})
        (d / "methods.json").write_text(json.dumps(m))
    assert set(sbs.check_eval_settings({"filtered": f, "all": a})) == {"filtered", "all"}
    m = json.loads((a / "methods.json").read_text())
    m.update(change)
    (a / "methods.json").write_text(json.dumps(m))
    args = ["table", "--eval", f"filtered={f}", "--eval", f"all={a}", "--primary", "filtered", "--tag", "t"]
    with pytest.raises(SystemExit, match=message):
        sbs.main([*args, "--out-dir", str(tmp_path / "out")])


def test_table_refuses_versions_not_scored_with_exclude_target(tmp_path):
    dirs = {}
    for v in ("filtered", "all"):
        dirs[v] = write_eval(tmp_path / v, method_frames(v))
        m = json.loads((dirs[v] / "methods.json").read_text())
        m["exclude_target"] = False
        (dirs[v] / "methods.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="exclude-target"):
        sbs.check_eval_settings(dirs)


# ---------------------------------------------------------------------------
# selection check and config
# ---------------------------------------------------------------------------
def selection(chosen: str, w2_ema: float = 0.93) -> dict:
    table = [
        {"config": "w1", "n_scored": 60, "balanced_acc_median": 0.92},
        {"config": "w2_ema", "n_scored": 60, "balanced_acc_median": w2_ema},
    ]
    return {"chosen": chosen, "table": table}


def test_check_selection(tmp_path):
    rep = sbs.check_selection(selection("w2_ema"), "w2_ema", selection("w2_ema"))
    assert rep["pass"] and rep["table_matches_frozen"] and rep["max_abs_diff"] == 0
    rep = sbs.check_selection(selection("w2_ema", 0.9301), "w2_ema", selection("w2_ema"))
    assert rep["pass"] and not rep["table_matches_frozen"]  # the table is reported, the choice is the gate
    assert not sbs.check_selection(selection("w1"), "w2_ema", selection("w2_ema"))["pass"]
    assert not sbs.check_selection(selection("w2_ema"), "w2_ema", selection("w1"))["pass"]
    assert sbs.check_selection(selection("w2_ema"), "w2_ema")["pass"]
    (tmp_path / "r.json").write_text(json.dumps(selection("w1")))
    args = ["check-selection", "--rescored", str(tmp_path / "r.json"), "--expected", "w2_ema"]
    assert sbs.main([*args, "--out", str(tmp_path / "c.json")]) == 1
    assert not json.loads((tmp_path / "c.json").read_text())["pass"]


def parse_shell(text: str) -> tuple[dict, dict]:
    values, named = {}, {"SWEEP": {}, "PRED": {}, "CONTROL": {}}
    for line in text.splitlines():
        if line.startswith("CFG_"):
            k, v = line.split("=", 1)
            values[k] = "".join(shlex.split(v))
        else:
            kind, name, path = line.split("\t")
            named[kind][name] = path
    return values, named


def test_config_dump_of_the_shipped_config(capsys):
    assert sbs.main(["config", "--config", str(EVAL_CONFIG), "generation_dir=/gen", "prepared_dir=/p"]) == 0
    values, named = parse_shell(capsys.readouterr().out)
    assert values["CFG_OUT_DIR"] == "/p/eval" and values["CFG_VERSIONS"] == "filtered all"
    assert values["CFG_REQUIRE_DATA_DONE"] == "1" and values["CFG_GT_MIN_CELLS"] == "15"
    assert values["CFG_REFERENCE"] == "scLDM_s0" and values["CFG_SEL_EXPECTED"] == "w2_ema"
    assert values["CFG_A5_TOL"] == "1e-09" and values["CFG_L_DEVICE"] == "auto"
    assert values["CFG_UNCH_MAX_EFFECT"] == "0.05" and values["CFG_UNCH_MIN_PADJ"] == "0.5"
    assert values["CFG_MIN_UNCHANGED"] == "20" and values["CFG_DIRECTION_ONLY"] == "mlp prior"
    assert values["CFG_METRIC"] == "balanced_acc" and values["CFG_UNCH_METRIC"] == "detect_auroc"
    assert named["PRED"] == {f"scLDM_s{k}": f"/gen/pred/test312_w2_ema_seed{k}.parquet" for k in range(3)}
    assert set(named["SWEEP"]) == {"w1", "w1_ema", "w2", "w2_ema", "w4", "w4_ema"}
    assert named["SWEEP"]["w2_ema"] == "/gen/sweep/w2_ema.parquet" and len(named["CONTROL"]) == 3
    assert sbs.main(["config", "--config", str(EVAL_CONFIG), "controls=null", "primary_version=all"]) == 0
    values, named = parse_shell(capsys.readouterr().out)
    assert named["CONTROL"] == {} and values["CFG_VERSIONS"] == "all filtered"
    assert sbs.main(["config", "--config", str(EVAL_CONFIG), "evaluate.direction_only=[]"]) == 0
    assert parse_shell(capsys.readouterr().out)[0]["CFG_DIRECTION_ONLY"] == ""  # nothing is left out


@pytest.mark.parametrize(
    "override, message",
    [
        ("reference=scLDM_s9", "reference"),
        ("selection.expected=w8", "selection.expected"),
        ("versions=[raw]", "unknown"),
        ("evaluate.direction_only=[a b]", "direction_only"),
    ],
)
def test_config_dump_validates(override, message):
    with pytest.raises(ValueError, match=message):
        sbs.main(["config", "--config", str(EVAL_CONFIG), override])


# ---------------------------------------------------------------------------
# runner end to end on tiny synthetic prepared files
# ---------------------------------------------------------------------------
GENES = [f"G{i:02d}" for i in range(40)]
LINES = ("hepg2", "k562", "rpe1")
TRAIN_PERTS = GENES[:12]  # HepG2 training perturbations (also perturbed in the other lines)
TEST_PERTS = GENES[12:20]  # HepG2 test perturbations, perturbed in the other lines in train
VAL, TEST = TEST_PERTS[:3], TEST_PERTS[3:]
CTRL = "non-targeting"


def make_cells(rng, line: str, base: np.ndarray, effects: dict, design: dict, kd: bool) -> list[tuple]:
    cells = []
    for label, n in design.items():
        e = effects[label] if (kd and label != CTRL) else np.zeros(len(GENES))
        mu = base * np.exp(e) * rng.uniform(0.8, 1.2, (n, 1))
        cells += [(line, label, row, kd or label == CTRL) for row in rng.poisson(mu)]
    return cells


def write_prepared(path: Path, cells: list[tuple], start: int = 0) -> None:
    X = sparse.csr_matrix(np.array([c[2] for c in cells], dtype=np.int32))
    ids = [f"BC{start + i:05d}" for i in range(len(cells))]
    obs = pd.DataFrame(
        {
            "cell_id": ids,
            "gene": pd.Categorical([c[1] for c in cells]),
            "cell_line": pd.Categorical([c[0] for c in cells]),
            "screen": [c[0] for c in cells],
            "gem_group": np.ones(len(cells), dtype=np.int64),
            "UMI_count": np.asarray(X.sum(1)).ravel().astype(np.float32),
            "library_full": np.asarray(X.sum(1)).ravel().astype(np.int64),
            "kd_keep": [c[3] for c in cells],
        },
        index=[f"{c[0]}:{i}" for c, i in zip(cells, ids, strict=True)],
    )
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)).write_h5ad(path)


def build_prepared(d: Path) -> None:
    """replogle_{train,test}_raw_{filtered,all}.h5ad; 'all' = 'filtered' + 6 unperturbed cells per perturbation."""
    rng = np.random.default_rng(0)
    shared = np.zeros(len(GENES))
    shared[20:25], shared[25:30] = 0.9, -0.9
    effects = {}
    for p in TRAIN_PERTS + TEST_PERTS:
        e = rng.uniform(0.6, 1.4) * shared
        idx = rng.choice(np.arange(30, 40), 6, replace=False)
        e[idx] += rng.choice([-1.2, 1.2], 6)
        e[GENES.index(p)] = np.log(0.2)  # CRISPRi: the target goes down
        effects[p] = e
    train, test, extra_train, extra_test = [], [], [], []
    for line in LINES:
        base = rng.uniform(8, 30, len(GENES))
        perts = TRAIN_PERTS if line == "hepg2" else TRAIN_PERTS + TEST_PERTS
        train += make_cells(rng, line, base, effects, {CTRL: 60, **dict.fromkeys(perts, 24)}, kd=True)
        extra_train += make_cells(rng, line, base, effects, dict.fromkeys(perts, 6), kd=False)
        if line == "hepg2":
            test += make_cells(rng, line, base, effects, dict.fromkeys(TEST_PERTS, 30), kd=True)
            extra_test += make_cells(rng, line, base, effects, dict.fromkeys(TEST_PERTS, 6), kd=False)
    d.mkdir(parents=True)
    write_prepared(d / "replogle_train_raw_filtered.h5ad", train)
    write_prepared(d / "replogle_test_raw_filtered.h5ad", test, start=50_000)
    write_prepared(d / "replogle_train_raw_all.h5ad", train + extra_train)
    write_prepared(d / "replogle_test_raw_all.h5ad", test + extra_test, start=50_000)
    (d / "manifest.json").write_text(json.dumps({"versions": ["filtered", "all"], "synthetic": True}))
    (d / "DONE_raw_data").write_text("finished synthetic\n")


def profiles(delta: pd.DataFrame, rows: list[str], noise: float, seed: int, flip: float = 0.0) -> pd.DataFrame:
    """A generate.py-like output: control row + control + predicted effect for every perturbation."""
    rng = np.random.default_rng(seed)
    eff = delta.loc[rows].to_numpy() + rng.normal(0, noise, (len(rows), delta.shape[1]))
    eff *= np.where(rng.random(eff.shape) < flip, -1, 1)
    ctrl = np.full(delta.shape[1], 2.0)
    return pd.DataFrame(np.vstack([ctrl, ctrl + eff]), index=[CTRL, *rows], columns=delta.columns)


@pytest.fixture(scope="module")
def eval_inputs(tmp_path_factory):
    from experiments.perturbation_direction.evaluation import ground_truth, subset_truth
    from experiments.perturbation_direction.generation import select_sweep

    d = tmp_path_factory.mktemp("raweval")
    build_prepared(d / "prepared")
    meta = d / "train.json"
    meta.write_text(json.dumps({"genes": GENES, "labels": {"cell_line": list(LINES), "gene": [*GENES, CTRL]}}))
    split = d / "split.json"
    split.write_text(json.dumps({"val": VAL, "test": TEST}))
    # the frozen "STATE" truth: the same computation on the same (filtered) cells
    ground_truth.main(
        [
            "--pert-h5ad",
            str(d / "prepared" / "replogle_test_raw_filtered.h5ad"),
            "--ctrl-h5ad",
            str(d / "prepared" / "replogle_train_raw_filtered.h5ad"),
            "--pert-key",
            "gene",
            "--control",
            CTRL,
            "--context",
            "cell_line=hepg2",
            "--metadata-json",
            str(meta),
            "--min-cells",
            "15",
            "--out-dir",
            str(d / "frozen" / "truth"),
        ]
    )
    subset_truth.main(
        [
            "--truth-dir",
            str(d / "frozen" / "truth"),
            "--split-json",
            str(split),
            "--out-dir",
            str(d / "frozen" / "test"),
        ]
    )
    delta = pd.read_parquet(d / "frozen" / "truth" / "delta_true.parquet")
    gen = d / "gen"
    (gen / "sweep").mkdir(parents=True)
    (gen / "pred").mkdir()
    for name in ("w1", "w1_ema", "w2", "w2_ema", "w4", "w4_ema"):  # only w2_ema is right on every gene
        profiles(delta, VAL, 0.0, seed=1, flip=0.0 if name == "w2_ema" else 0.4).to_parquet(
            gen / "sweep" / f"{name}.parquet"
        )
    for k in range(3):
        profiles(delta, TEST, 0.1, seed=10 + k).to_parquet(gen / "pred" / f"test312_w2_ema_seed{k}.parquet")
    profiles(delta, TEST, 0.3, seed=20).to_parquet(gen / "pred" / "test312_w1.parquet")  # cells128, w0: absent
    select_sweep.main(
        [
            "--sweep-dir",
            str(gen / "sweep"),
            "--truth-dir",
            str(d / "frozen" / "truth"),
            "--val-split",
            str(split),
            "--out",
            str(d / "frozen" / "selection.json"),
        ]
    )
    assert json.loads((d / "frozen" / "selection.json").read_text())["chosen"] == "w2_ema"
    # like the real sweep dir, a *_sample file sits next to the configs; select_sweep.py would read it as a
    # perfect w0 config (and choose it) if the runner pointed it at the directory
    profiles(delta, VAL, 0.0, seed=2, flip=0.0).to_parquet(gen / "sweep" / "w0_sample.parquet")

    cfg = OmegaConf.load(EVAL_CONFIG)
    cfg.prepared_dir = str(d / "prepared")
    cfg.results_dir = str(d / "results")
    cfg.train_json = str(meta)
    cfg.split_json = str(split)
    cfg.generation_dir = str(gen)
    cfg.frozen.truth = str(d / "frozen" / "truth")
    cfg.frozen.truth_test = str(d / "frozen" / "test")
    cfg.frozen.summary = None
    cfg.frozen.summary_specific = None
    cfg.selection.frozen_json = str(d / "frozen" / "selection.json")
    cfg.learned.update({"embedding_dim": 8, "epochs": 1, "batch_size": 512, "device": "cpu"})
    # 40 genes and 30 cells per perturbation: few genes are clearly unchanged at the shipped thresholds
    cfg.evaluate.update({"unchanged_max_effect": 0.2, "unchanged_min_padj": 0.2, "min_unchanged": 3})
    cfg.stats.n_boot = N_BOOT
    OmegaConf.save(cfg, d / "eval.yaml")
    return d


def run_runner(cfg: Path, *overrides: str, job_root: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHON=sys.executable, CONFIG=str(cfg), OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore")
    if job_root is not None:
        env["JOB_ROOT"] = str(job_root)
    return subprocess.run(["bash", str(RUNNER), *overrides], env=env, capture_output=True, text=True, timeout=900)


STEPS = {
    "truth_filtered",
    "subset_val_filtered",
    "subset_test_filtered",
    "a5_truth",
    "a5_truth_test",
    "stage_sweep",
    "select_sweep",
    "select_check",
    "truth_all",
    "subset_val_all",
    "subset_test_all",
    "tierb_all",
    "tierb_test_all",
    "baselines_filtered",
    "learned_filtered",
    "baselines_all",
    "learned_all",
    "eval_filtered",
    "eval_all",
    "controls_filtered",
    "controls_all",
    "side_by_side",
    "side_by_side_controls",
}


def test_runner_end_to_end_resumes_and_stops_at_a5(eval_inputs):
    d = eval_inputs
    cfg = d / "eval.yaml"
    r = run_runner(cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    out = d / "prepared" / "eval"
    assert "RAW_EVAL_DONE" in r.stdout and (out / "DONE_raw_eval").exists()
    done = (out / "DONE_raw_eval").read_text()
    assert "chosen w2_ema" in done and "versions filtered all" in done
    assert {p.stem for p in (out / "logs").glob("*.done")} == STEPS
    assert "control scLDM_w0 skipped" in r.stdout and "control scLDM_w2_ema_cells128 skipped" in r.stdout

    for v in ("filtered", "all"):
        for sub in ("", "_val", "_test"):
            assert (out / f"truth_{v}{sub}" / "delta_true.parquet").exists()
        assert set(pd.read_parquet(out / f"truth_{v}_test" / "delta_true.parquet").index) == set(TEST)
        for f in ("delta_generic", "delta_transfer", "delta_knn"):
            assert (out / f"baselines_{v}" / f"{f}.parquet").exists()
        for f in ("delta_prior", "delta_mlp"):
            assert (out / f"learned_{v}" / f"{f}.parquet").exists()
        per = {p.stem for p in (out / f"eval_{v}" / "per_perturbation").glob("*.csv")}
        assert per == {
            "scLDM_s0",
            "scLDM_s1",
            "scLDM_s2",
            "scLDM_s0_(shuffled)",
            "scLDM_s1_(shuffled)",
            "scLDM_s2_(shuffled)",
            "knn",
            "mlp",
            "generic",
            "transfer",
            "prior",
            "noise_ceiling_(split-half)",
        }
        methods = json.loads((out / f"eval_{v}" / "methods.json").read_text())
        assert methods["exclude_target"] and methods["specific_ref"]["path"].endswith(
            f"baselines_{v}/delta_generic.parquet"
        )
        # the third report: the config's thresholds reach evaluate.py, mlp and prior (direction-only) are left
        # out, and its split-half ceiling exists because the truth (and its test subset) now carry half A's padj
        assert (out / f"truth_{v}" / "padj_half_a.parquet").exists()
        assert set(pd.read_parquet(out / f"truth_{v}_test" / "padj_half_a.parquet").index) == set(TEST)
        unchanged = {p.stem for p in (out / f"eval_{v}" / "per_perturbation_unchanged").glob("*.csv")}
        assert unchanged == per - {"mlp", "prior"}
        assert methods["direction_only"] == ["mlp", "prior"]
        assert methods["unchanged"]["direction_only_excluded"] == ["mlp", "prior"] and methods["unchanged"]["ceiling"]
        assert [methods["unchanged"][k] for k in ("max_effect", "min_padj", "min_unchanged", "tau")] == [
            0.2,
            0.2,
            3,
            0.1,
        ]
        summary = pd.read_csv(out / f"eval_{v}" / "summary_unchanged.csv", index_col=0)
        assert "mlp" not in summary.index and {"scLDM_s0", "knn", "noise ceiling (split-half)"} <= set(summary.index)
        assert (out / f"eval_controls_{v}" / "summary_unchanged.csv").exists()
        assert {p.stem for p in (out / f"eval_controls_{v}" / "per_perturbation").glob("*.csv")} >= {
            "scLDM_s0",
            "scLDM_w1",
        }
    a5 = json.loads((out / "verify" / "a5_truth.json").read_text())
    assert a5["pass"] and json.loads((out / "verify" / "a5_truth_test.json").read_text())["pass"]
    tierb = json.loads((out / "verify" / "filtered_vs_all" / "filtered_vs_all.json").read_text())
    assert tierb["perturbations"]["both"] == len(TEST_PERTS)
    assert (out / "verify" / "filtered_vs_all_test" / "filtered_vs_all.csv").exists()
    sel = json.loads((out / "selection" / "selection_filtered.json").read_text())
    assert sel["chosen"] == "w2_ema" and len(sel["table"]) == 6  # the decoy *_sample file was not staged
    check = json.loads((out / "selection" / "check.json").read_text())
    assert check["pass"] and check["table_matches_frozen"]

    res = d / "results"
    tag = "replogle_raw_test312_w2ema_3seeds"
    for t in (tag, f"{tag}_controls"):
        for suffix in (
            "_side_by_side.csv",
            "_side_by_side_specific.csv",
            "_side_by_side_unchanged.csv",
            "_side_by_side_long.csv",
            "_side_by_side_panel.csv",
            "_side_by_side.md",
            ".provenance.json",
        ):
            assert (res / f"{t}{suffix}").exists(), f"{t}{suffix}"
    wide = pd.read_csv(res / f"{tag}_side_by_side.csv", index_col=0)
    assert list(wide.index[:3]) == ["scLDM_s0", "scLDM_s1", "scLDM_s2"]
    assert set(wide.index) == {
        "scLDM_s0", "scLDM_s1", "scLDM_s2", "knn", "mlp", "generic", "transfer", "prior",
        "scLDM_s0_(shuffled)", "scLDM_s1_(shuffled)", "scLDM_s2_(shuffled)", "noise_ceiling_(split-half)",
    }  # fmt: skip
    assert (wide.filtered_status == "ok").all() and (wide.all_status == "ok").all()
    assert wide.loc["scLDM_s0", "filtered_n"] > 0 and wide.loc["knn", "filtered_p_holm"] >= 0
    third = pd.read_csv(res / f"{tag}_side_by_side_unchanged.csv", index_col=0)
    assert list(third.index) == list(wide.index)  # the same rows; the direction-only methods are missing
    for v in ("filtered", "all"):
        assert set(third.index[third[f"{v}_status"] == "missing"]) == {"mlp", "prior"}
    assert third.loc["scLDM_s0", "filtered_n"] > 0 and 0 <= third.loc["scLDM_s0", "filtered_median"] <= 1
    md = (res / f"{tag}_side_by_side.md").read_text()
    assert "## Changed vs unchanged genes" in md and "- not scored in filtered: mlp, prior" in md
    assert "|Δ| < 0.2, padj ≥ 0.2" in md and ">= 3 unchanged genes" in md and "`three_class_balanced_acc`" in md
    prov = json.loads((res / f"{tag}.provenance.json").read_text())
    assert prov["metrics"] == {"full": "balanced_acc", "specific": "balanced_acc", "unchanged": "detect_auroc"}
    assert prov["min_unchanged"] == 3 and prov["unchanged"]["all"]["max_effect"] == 0.2
    hashed = json.loads((out / "logs" / "side_by_side.hash.json").read_text())["files"]
    assert {str(out / f"eval_{v}" / "summary_unchanged.csv") for v in ("filtered", "all")} <= set(hashed)
    manifest = d / "prepared" / "manifest.json"
    assert prov["extra_inputs"]["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    for k in range(3):
        pred = d / "gen" / "pred" / f"test312_w2_ema_seed{k}.parquet"
        assert prov["inputs"][str(pred)] == hashlib.sha256(pred.read_bytes()).hexdigest()
    assert prov["extra_inputs"]["sweep_w2_ema"]["sha256"]
    assert prov["extra_inputs"]["a5_truth"]["sha256"] and prov["extra_inputs"]["selection_check"]["sha256"]
    assert prov["config"]["path"] == str(cfg) and prov["timestamp"].endswith("Z")
    assert prov["config"]["resolved"]["learned"]["epochs"] == 1

    # resumable: a rerun skips every step
    r = run_runner(cfg)
    assert r.returncode == 0 and "skip truth_filtered" in r.stdout and "skip side_by_side" in r.stdout
    assert "start " not in r.stdout

    # A5: a frozen truth that differs by 1e-3 in one entry stops the run before anything is fitted
    bad = d / "frozen_bad"
    shutil.copytree(d / "frozen" / "truth", bad)
    delta = pd.read_parquet(bad / "delta_true.parquet")
    delta.iloc[0, 0] += 1e-3
    delta.to_parquet(bad / "delta_true.parquet")
    out2 = d / "out_bad"
    r = run_runner(cfg, f"frozen.truth={bad}", f"out_dir={out2}", f"results_dir={d / 'results_bad'}")
    assert r.returncode == 1 and "FAIL  a5_truth" in r.stderr
    assert not (out2 / "DONE_raw_eval").exists()
    assert not json.loads((out2 / "verify" / "a5_truth.json").read_text())["pass"]
    assert {p.stem for p in (out2 / "logs").glob("*.done")} == {
        "truth_filtered",
        "subset_val_filtered",
        "subset_test_filtered",
    }
    assert not (out2 / "baselines_filtered").exists() and not (d / "results_bad").exists()

    # changed inputs (here: the prepared manifest, after a data-stage rerun) invalidate every step
    manifest.write_text(json.dumps({"versions": ["filtered", "all"], "synthetic": "rebuilt"}))
    r = run_runner(cfg, f"frozen.truth={bad}")  # rerun from scratch; it then stops at A5 again
    assert r.returncode == 1 and "inputs changed since the last run" in r.stdout
    assert str(manifest) in r.stdout and "start truth_filtered" in r.stdout
    assert not (out / "DONE_raw_eval").exists() and not (out / "logs" / "side_by_side.done").exists()

    # missing inputs are listed before anything runs
    r = run_runner(cfg, f"prepared_dir={d / 'nowhere'}")
    assert r.returncode == 2 and "DONE_raw_data" in r.stderr and not (d / "nowhere").exists()


def test_step_hash_masks_a_volatile_argv_value():
    """A value the runners name with --mask-argv (the table timestamp) never reruns a step."""
    from experiments.perturbation_direction.data import step_hash

    def digest(stamp: str, mask: tuple[str, ...] = ()) -> str:
        return step_hash.step_hash(config_digest="d", argv=["t.py", "--timestamp", stamp], mask=mask)[0]

    a, b = "2026-11-10T12:00:00Z", "2026-11-11T09:00:00Z"
    assert digest(a) != digest(b)
    assert digest(a, (a,)) == digest(b, (b,))
    _, comp = step_hash.step_hash(config_digest="d", argv=["t.py", "--timestamp", a], mask=[a])
    assert comp["argv"] == ["t.py", "--timestamp", step_hash.ARGV_MASK]


def test_runner_resume_reruns_a_step_whose_hash_changed(eval_inputs, tmp_path):
    """A .done marker is valid only for the step hash it recorded (step_hash.py, as in run_replogle_raw_data.sh).

    A deleted marker, an edited step script and a missing inputs stamp rerun that step and every step downstream
    of it; an unchanged rerun skips everything. The pipeline runs from a copy of its directory, so the test can
    edit a step's script without touching the repository.
    """
    d = eval_inputs
    cfg = d / "eval.yaml"
    job = tmp_path / "job"
    shutil.copytree(
        REPO / "experiments" / "perturbation_direction",
        job / "experiments" / "perturbation_direction",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    out = d / "prepared" / "eval_resume"
    logs = out / "logs"
    over = (f"out_dir={out}", f"results_dir={d / 'results_resume'}")

    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0, r.stdout + r.stderr
    assert {p.stem for p in logs.glob("*.done")} == STEPS and (out / "DONE_raw_eval").exists()
    for name in STEPS:  # every marker starts with the hash of the inputs, code and upstream markers it ran with
        first = (logs / f"{name}.done").read_text().splitlines()[0]
        assert first.startswith("hash=") and len(first) == len("hash=") + 64, name

    # unchanged inputs, code and markers: every step stays done
    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0 and "start " not in r.stdout
    assert r.stdout.count("skip ") == len(STEPS) and (out / "DONE_raw_eval").exists()

    # a deleted marker reruns that step, and everything downstream of it, and nothing else
    (logs / "select_sweep.done").unlink()
    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0
    assert "start select_sweep (not done)" in r.stdout and "start select_check " in r.stdout
    assert "start side_by_side " in r.stdout
    assert "invalidated side_by_side (downstream of select_sweep)" in r.stdout
    for name in ("truth_filtered", "subset_test_filtered", "learned_all", "eval_all", "side_by_side_controls"):
        assert f"skip {name} " in r.stdout
    assert {p.stem for p in logs.glob("*.done")} == STEPS and (out / "DONE_raw_eval").exists()

    # an edited step script reruns it: the code closure is part of the hash
    script = job / "experiments" / "perturbation_direction" / "generation" / "select_sweep.py"
    script.write_text(script.read_text() + "\n# edited by the resume test\n")
    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0 and "start select_sweep (hash changed: code: select_sweep.py)" in r.stdout
    assert "start select_check " in r.stdout and "skip truth_all " in r.stdout

    # the tables depend on the content of each eval dir's changed-vs-unchanged summary
    with (out / "eval_all" / "summary_unchanged.csv").open("a") as fh:
        fh.write("\n")
    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0 and "start side_by_side (hash changed: files: " in r.stdout
    assert "eval_all/summary_unchanged.csv" in r.stdout
    assert "skip eval_all " in r.stdout and "skip side_by_side_controls " in r.stdout

    # a marker without a hash (an earlier runner wrote a timestamp) is stale
    (logs / "select_check.done").write_text("2026-11-10T12:00:00\n")
    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0
    assert "start select_check (marker without a step hash" in r.stdout and "start side_by_side " in r.stdout
    assert "skip select_sweep " in r.stdout

    # a missing inputs stamp: the markers cannot be shown to match the inputs, so every step reruns
    (logs / "inputs.sha256").unlink()
    r = run_runner(cfg, *over, job_root=job)
    assert r.returncode == 0 and "is missing while" in r.stdout
    assert "start truth_filtered (not done)" in r.stdout and "start side_by_side " in r.stdout
    assert (logs / "inputs.sha256").exists() and (out / "DONE_raw_eval").exists()


def test_git_state_ignores_the_tables_own_output_dir(tmp_path):
    """A regenerated table must not mark its own provenance dirty; any other change still does."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "results" / "tag").mkdir(parents=True)
    (repo / "code.py").write_text("x = 1\n")
    (repo / "results" / "tag" / "old.csv").write_text("a\n")
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"], check=True)
    out = repo / "results" / "tag"
    (out / "old.csv").write_text("b\n")  # rewritten table
    (out / "new.provenance.json").write_text("{}")  # new file in the output dir
    state = sbs.git_state(repo, exclude=out)
    assert state["dirty"] is False and state["changed_paths"] == 0 and state["ignored_output_paths"] == 2
    assert sbs.git_state(repo)["dirty"] is True  # without exclude the same tree is dirty
    (repo / "code.py").write_text("x = 2\n")  # a real change outside the output dir
    state = sbs.git_state(repo, exclude=out)
    assert state["dirty"] is True and state["changed_paths"] == 1
    assert sbs.git_state(repo, exclude=tmp_path / "elsewhere")["changed_paths"] == 3  # out dir outside the repo
