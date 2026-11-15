"""Edge cases the synthetic happy-path fixture hides.

Each case was reproduced against real `outputs/paper` runs before the fix:

1. a folder with a `summary.csv` but no per-knockdown tables was offered as a run and crashed
   every view (the metric picker hit `next(iter({}.values()))`);
2. a method with no prediction for the selected knockdown crashed the Perturbation view
   (scDEBART lacks 51 knockdowns; the per-line transfers lack 54-87);
3. choosing no method and unticking the ceiling crashed Overview and Strength.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dashboard_data import build_run

APP = str(Path(__file__).resolve().parents[3] / "experiments" / "perturbation_direction" / "dashboard" / "dashboard.py")
from experiments.perturbation_direction.dashboard import dashboard_data as dd  # noqa: E402


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    eval_dir = build_run(tmp_path)
    monkeypatch.setenv("DASHBOARD_ROOT", str(eval_dir.parent))
    return eval_dir


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_incomplete_folder_is_not_offered_as_a_run(run_root):
    """A summary.csv without per_perturbation/ is not an evaluate.py output."""
    partial = run_root.parent / "partial_run"
    partial.mkdir()
    pd.read_csv(run_root / "summary.csv").to_csv(partial / "summary.csv", index=False)

    listed = [p.name for p in dd.find_runs(run_root.parent)]
    assert "partial_run" not in listed
    assert run_root.name in listed

    at = fresh()
    options = at.sidebar.selectbox[0].options
    assert not any("partial_run" in o for o in options)


def test_run_with_empty_per_perturbation_folder_is_not_listed(run_root):
    empty = run_root.parent / "empty_tables"
    (empty / "per_perturbation").mkdir(parents=True)
    pd.read_csv(run_root / "summary.csv").to_csv(empty / "summary.csv", index=False)
    assert "empty_tables" not in [p.name for p in dd.find_runs(run_root.parent)]


def test_method_missing_a_knockdown_does_not_crash_the_perturbation_view(run_root):
    """The view must drop the method for that knockdown rather than ask for a missing column.

    Effects come from the prediction parquet named in methods.json, not from the scored
    per-knockdown CSV, so that parquet is the file that has to be missing a row.
    """
    manifest = json.loads((run_root / "methods.json").read_text())
    entry = next(m for m in manifest["methods"] if m["kind"] in ("baseline", "pred"))
    path = Path(entry["path"]) if Path(entry["path"]).is_absolute() else run_root / entry["path"]
    frame = pd.read_parquet(path)
    dropped = frame.index[-1]
    frame.drop(index=dropped).to_parquet(path)
    name = entry["name"]

    at = fresh()
    at.sidebar.radio(key="view").set_value("Perturbation").run()
    at.selectbox(key="pert").set_value(dropped).run()
    at.sidebar.multiselect(key="methods_full").set_value([name]).run()

    assert not at.exception, at.exception
    # the method had no row for this knockdown, so nothing is left to plot
    assert any("No prediction" in c.value for c in at.caption)


def test_no_method_selected_does_not_crash_overview_or_strength(run_root):
    for view in ("Overview", "Strength & distributions"):
        at = fresh()
        at.sidebar.multiselect(key="methods_full").set_value([]).run()
        at.sidebar.checkbox[0].set_value(False).run()
        at.sidebar.radio(key="view").set_value(view).run()
        assert not at.exception, (view, at.exception)
        assert at.info, f"{view} should explain that nothing is selected"
