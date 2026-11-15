"""Headless run of the Streamlit dashboard on a synthetic evaluation (all views, both reports, click hand-off)."""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dashboard_data import build_run

APP = str(Path(__file__).resolve().parents[3] / "experiments" / "perturbation_direction" / "dashboard" / "dashboard.py")
VIEWS = ["Perturbation", "Overview", "Strength & distributions", "Head-to-head"]


@pytest.fixture(scope="module")
def app_root(tmp_path_factory):
    run_dir = build_run(tmp_path_factory.mktemp("dash_app"))
    os.environ["DASHBOARD_ROOT"] = str(run_dir.parent)
    yield run_dir
    os.environ.pop("DASHBOARD_ROOT", None)


def fresh(app_root) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("report", ["full", "specific"])
@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders(app_root, view, report):
    at = fresh(app_root)
    at.sidebar.radio(key="report").set_value(report).run()
    at.sidebar.radio(key="view").set_value(view).run()
    assert not at.exception, at.exception
    assert at.header[0].value == view
    if view == "Perturbation":
        assert len(at.dataframe) == 1 and len(at.metric) == 4
    if view != "Head-to-head":
        assert len(at.get("plotly_chart")) >= 1


def test_click_handoff_opens_perturbation(app_root):
    at = fresh(app_root)  # opens on the Perturbation view with the default perturbation selected
    options = at.selectbox(key="pert").options
    default = at.selectbox(key="pert").value
    target = next(o.split()[0] for o in options if not o.startswith(default))
    at.sidebar.radio(key="view").set_value("Head-to-head").run()
    at.session_state["_goto"] = target  # what a chart click sets before st.rerun()
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.radio(key="view").value == "Perturbation"
    assert at.selectbox(key="pert").value == target
