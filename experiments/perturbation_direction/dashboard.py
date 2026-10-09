"""Interactive explorer for perturbation-direction evaluations (evaluate.py outputs).

    streamlit run experiments/perturbation_direction/dashboard.py -- --root outputs/perturbation_direction

Views: a single perturbation's predicted vs true effects per gene, the method overview, accuracy by perturbation
strength and per-perturbation distributions, and head-to-head comparisons. Clicking a perturbation in any chart
opens it in the Perturbation view. Runs are any directories under --root containing summary.csv; the gene-level
views need the methods.json manifest that evaluate.py writes.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dashboard_data as dd
from metrics import remove_shared

# reference categorical palette (fixed slot order) + status and neutral tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GOOD, BAD, NEUTRAL, CEIL = "#0ca30c", "#d03b3b", "#c3c2b7", "#7a7974"
METRICS = {
    "balanced_acc": "Balanced sign accuracy",
    "sign_acc": "Sign accuracy",
    "sign_acc_top20": "Top-20 sign accuracy",
    "auroc": "AUROC (up vs down)",
    "spearman_deg": "Spearman on DEGs",
}
VIEWS = ["Perturbation", "Overview", "Strength & distributions", "Head-to-head"]
REPORT_LABELS = {"full": "Full effect", "specific": "Perturbation-specific (shared response removed)"}


def default_root() -> pathlib.Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    root = args.root or os.environ.get("DASHBOARD_ROOT")
    return (
        pathlib.Path(root)
        if root
        else pathlib.Path(__file__).resolve().parents[2] / "outputs" / "perturbation_direction"
    )


# ---------------------------------------------------------------- cached loaders
@st.cache_data
def runs_under(root: str) -> list[str]:
    return [str(p) for p in dd.find_runs(root)]


@st.cache_data
def load_run(path: str) -> dd.Run:
    return dd.load_run(path)


@st.cache_data
def load_truth(path: str) -> dd.Truth:
    run = load_run(path)
    return dd.load_truth(dd.resolve(run.manifest["truth_dir"]))


@st.cache_data
def load_effects(path: str, name: str, report: str) -> pd.DataFrame:
    run = load_run(path)
    eff = dd.load_effects(run.manifest, name, load_truth(path))
    if report == "specific":
        eff = remove_shared(eff, shared_ref(path))[0]
    return eff


@st.cache_data
def shared_ref(path: str) -> pd.Series:
    return dd.shared_reference(load_run(path).manifest)


@st.cache_data
def truth_for(path: str, report: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = load_truth(path)
    return dd.specific_truth(truth, shared_ref(path)) if report == "specific" else (truth.delta, truth.deg)


# ---------------------------------------------------------------- helpers
def entity(name: str) -> str:
    return name.removesuffix(" (shuffled)")


def color_map(run: dd.Run) -> dict[str, str]:
    """Fixed color per method entity for the whole run (models first), so selections never repaint a method."""
    kinds = {m["name"]: m["kind"] for m in (run.manifest or {}).get("methods", [])}
    ents = list(dict.fromkeys(entity(m) for m in run.methods if m != dd.CEILING))
    ents.sort(key=lambda m: 0 if kinds.get(m) == "pred" else 1)
    return {m: SERIES[i] for i, m in enumerate(ents[: len(SERIES)])}


def method_kind(run: dd.Run, name: str) -> str:
    if name == dd.CEILING:
        return "ceiling"
    if name.endswith("(shuffled)"):
        return "control"
    kinds = {m["name"]: m["kind"] for m in (run.manifest or {}).get("methods", [])}
    return "model" if kinds.get(name) == "pred" else "baseline"


def goto(pert: str) -> None:
    st.session_state["_goto"] = pert
    st.rerun()


def handle_selection(event) -> None:
    pts = getattr(getattr(event, "selection", None), "points", None) or []
    for p in pts:
        cd = p.get("customdata")
        if cd:
            goto(cd[0] if isinstance(cd, list | tuple) else cd)


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10}, legend_title_text="", hoverlabel={"font_size": 12}
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def reference_lines(fig: go.Figure, ceiling: float | None, vertical: bool = False) -> None:
    add = fig.add_vline if vertical else fig.add_hline
    add(
        0.5,
        line={"color": CEIL, "width": 1, "dash": "dot"},
        annotation_text="chance",
        annotation_font_color=CEIL,
        annotation_position="bottom right",
    )
    if ceiling is not None and np.isfinite(ceiling):
        add(
            ceiling,
            line={"color": CEIL, "width": 1.5, "dash": "dash"},
            annotation_text="noise ceiling",
            annotation_font_color=CEIL,
            annotation_position="bottom right",
        )


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Perturbation direction explorer", layout="wide")
if "_goto" in st.session_state:  # a chart click asked to open a perturbation
    st.session_state["view"] = "Perturbation"
    st.session_state["pert"] = st.session_state.pop("_goto")

root = default_root()
runs = runs_under(str(root))
if not runs:
    st.error(f"No evaluation runs (directories with summary.csv) under `{root}`.")
    st.stop()

with st.sidebar:
    st.title("Direction explorer")
    run_path = st.selectbox(
        "Evaluation run",
        runs,
        format_func=lambda p: str(pathlib.Path(p).relative_to(root)) if pathlib.Path(p).is_relative_to(root) else p,
    )
    run = load_run(run_path)
    view = st.radio("View", VIEWS, key="view")
    reports = [r for r in ("full", "specific") if r in run.summary]
    report = st.radio("Report", reports, format_func=REPORT_LABELS.get, key="report")
    candidates = [m for m in run.per_pert[report] if m != dd.CEILING]
    default = [m for m in candidates if not m.endswith("(shuffled)") and not m.startswith("transfer_")][:8]
    selected = st.multiselect("Methods", candidates, default=default, key=f"methods_{report}")
    show_ceiling = st.checkbox("Show noise ceiling", value=True)
    metric = st.selectbox(
        "Metric", [m for m in METRICS if m in next(iter(run.per_pert[report].values()))], format_func=METRICS.get
    )
    min_degs = st.number_input(
        "Min. true DEGs per perturbation", 0, 1000, int((run.manifest or {}).get("min_degs", 5)), step=1
    )
    st.caption("Click a perturbation in any chart to open it in the Perturbation view.")

colors = color_map(run)
has_manifest = run.manifest is not None
exclude_target = bool((run.manifest or {}).get("exclude_target", True))
per_pert = run.per_pert[report]
if has_manifest:
    truth = load_truth(run_path)
    t_delta, t_deg = truth_for(run_path, report)
    n_true = dd.deg_counts(t_deg, exclude_target)
else:
    truth = None
    n_true = next(iter(per_pert.values()))["n_deg"]
keep = n_true.index[n_true >= min_degs]
shown = selected + ([dd.CEILING] if show_ceiling and dd.CEILING in per_pert else [])


def scores(name: str) -> pd.Series:
    df = per_pert[name]
    return df.loc[df.index.intersection(keep), metric].dropna()


ceiling_median = scores(dd.CEILING).median() if show_ceiling and dd.CEILING in per_pert else None

st.header(view)
st.caption(
    f"{REPORT_LABELS[report]} · {METRICS[metric]} · perturbations with ≥ {min_degs} true DEGs: "
    f"{len(keep)} of {len(n_true)}"
)

# ================================================================ Perturbation
if view == "Perturbation":
    if not has_manifest:
        st.info("This run has no methods.json (made by an older evaluate.py). Re-run evaluate.py to enable this view.")
        st.stop()
    order = n_true.sort_values(ascending=False).index.tolist()
    if st.session_state.get("pert") not in order:
        st.session_state["pert"] = order[0]
    pert = st.selectbox(
        "Perturbation (sorted by number of true DEGs)",
        order,
        key="pert",
        format_func=lambda p: f"{p}  ·  {int(n_true[p])} DEGs",
    )
    pick = [m for m in selected if m in run.methods]
    show = st.multiselect("Methods to plot (max 4)", pick, default=pick[:3], max_selections=4, key="pert_methods")
    effects = {m: load_effects(run_path, m, report) for m in show}
    genes, mets = dd.perturbation_detail(
        pert, t_delta, t_deg, truth.padj, effects, exclude_target, float(run.manifest.get("eps", 0.0))
    )

    deg_row = genes[genes["deg"]]
    c = st.columns(4)
    c[0].metric("Cells", f"{truth.info.get('n_cells', {}).get(pert, '–')}")
    c[1].metric("True DEGs", f"{len(deg_row)}", help="Significant genes (target gene excluded) in this report")
    c[2].metric("Up / down", f"{int((deg_row['true_delta'] > 0).sum())} / {int((deg_row['true_delta'] < 0).sum())}")
    if pert in t_delta.columns:
        c[3].metric("Target gene Δ (true)", f"{truth.delta.loc[pert, pert]:+.2f}")
    else:
        c[3].metric("Target gene", "not in gene set", help="The knocked-down gene is not among the scored genes")

    if not show:
        st.info("Select at least one method.")
    else:
        st.markdown(
            f'<span style="color:{NEUTRAL}">●</span> not a DEG &nbsp;&nbsp; '
            f'<span style="color:{GOOD}">●</span> ✓ DEG, right sign &nbsp;&nbsp; '
            f'<span style="color:{BAD}">✖</span> ✗ DEG, wrong sign',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(show))
        lim = float(np.nanmax(np.abs(genes[["true_delta", *show]].to_numpy()))) * 1.05 or 1.0
        for col, m in zip(cols, show, strict=True):
            g = genes[["true_delta", m, "deg"] + (["padj"] if "padj" in genes else [])].dropna(subset=[m])
            right = np.sign(g[m]) == np.sign(g["true_delta"])
            fig = go.Figure()
            for label, mask, color, symbol, size, opacity in (
                ("not a DEG", ~g["deg"], NEUTRAL, "circle", 5, 0.5),
                ("✓ DEG, right sign", g["deg"] & right, GOOD, "circle", 8, 0.9),
                ("✗ DEG, wrong sign", g["deg"] & ~right, BAD, "x", 9, 0.95),
            ):
                s = g[mask]
                padj_txt = s["padj"].map("{:.1e}".format) if "padj" in s else pd.Series("–", index=s.index)
                fig.add_scatter(
                    x=s["true_delta"],
                    y=s[m],
                    mode="markers",
                    name=label,
                    marker={"color": color, "symbol": symbol, "size": size, "opacity": opacity, "line": {"width": 0}},
                    customdata=np.stack([s.index, padj_txt], axis=1) if len(s) else None,
                    hovertemplate="<b>%{customdata[0]}</b><br>true Δ %{x:.3f} (padj %{customdata[1]})"
                    "<br>predicted %{y:.3f}<extra></extra>",
                )
            fig.add_hline(0, line={"color": CEIL, "width": 1})
            fig.add_vline(0, line={"color": CEIL, "width": 1})
            fig.update_xaxes(title="true Δ", range=[-lim, lim])
            fig.update_yaxes(title="predicted", zeroline=False)
            style(fig, 380).update_layout(title={"text": m, "x": 0}, showlegend=False)
            col.plotly_chart(fig, width="stretch", key=f"pert_scatter_{m}")
            if m in mets.index:
                r = mets.loc[m]
                col.caption(
                    f"balanced acc **{r['balanced_acc']:.3f}** · sign acc {r['sign_acc']:.3f} · "
                    f"AUROC {r['auroc']:.3f} · Spearman (DEGs) {r['spearman_deg']:.3f}"
                )
        st.caption("Each point is a gene. Correct direction = same sign as the true effect; only DEGs are scored.")

    st.subheader("Genes")
    f1, f2 = st.columns([1, 3])
    only_deg = f1.checkbox("Only DEGs", value=True)
    query = f2.text_input("Search gene", "")
    tbl = genes[genes["deg"]] if only_deg else genes
    if query:
        tbl = tbl[tbl.index.str.contains(query, case=False, regex=False)]
    view_tbl = tbl.copy()
    for m in show:
        view_tbl[f"{m} ✓"] = view_tbl[f"{m} ✓"].map({1.0: "✓", 0.0: "✗"}).fillna("")
    st.dataframe(
        view_tbl,
        width="stretch",
        height=380,
        column_config={
            "true_delta": st.column_config.NumberColumn("true Δ", format="%.3f"),
            "padj": st.column_config.NumberColumn("padj", format="%.1e"),
            **{m: st.column_config.NumberColumn(m, format="%.3f") for m in show},
        },
    )
    st.download_button("Download table (CSV)", tbl.to_csv().encode(), file_name=f"{pert}_{report}_genes.csv")

# ================================================================ Overview
elif view == "Overview":
    rows = []
    for m in shown:
        if m in per_pert:
            s = scores(m)
            rows.append(
                {
                    "method": m,
                    "kind": method_kind(run, m),
                    "median": s.median(),
                    "q1": s.quantile(0.25),
                    "q3": s.quantile(0.75),
                    "n": len(s),
                }
            )
    stats = pd.DataFrame(rows).sort_values("median")
    kind_colors = {"model": SERIES[0], "baseline": SERIES[1], "control": NEUTRAL, "ceiling": CEIL}
    fig = go.Figure()
    for kind, color in kind_colors.items():
        s = stats[stats["kind"] == kind]
        if s.empty:
            continue
        fig.add_scatter(
            x=s["median"],
            y=s["method"],
            mode="markers",
            name=kind,
            marker={"color": color, "size": 12, "line": {"color": "white", "width": 2}},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": s["q3"] - s["median"],
                "arrayminus": s["median"] - s["q1"],
                "color": color,
                "thickness": 2,
                "width": 0,
            },
            customdata=s[["q1", "q3", "n"]].to_numpy(),
            hovertemplate="<b>%{y}</b><br>median %{x:.3f}<br>IQR %{customdata[0]:.3f}–%{customdata[1]:.3f}"
            "<br>%{customdata[2]} perturbations<extra></extra>",
        )
    reference_lines(fig, None, vertical=True)
    fig.update_xaxes(title=f"{METRICS[metric]} (median, bar = IQR across perturbations)")
    fig.update_yaxes(categoryorder="array", categoryarray=stats["method"].tolist())
    style(fig, 80 + 34 * len(stats)).update_layout(legend={"orientation": "h", "y": 1.08})
    st.plotly_chart(fig, width="stretch", key="overview_dots")

    st.subheader("Full effect vs perturbation-specific part")
    if "specific" in run.per_pert:
        both = []
        for m in shown:
            r = {"method": m}
            for rep in ("full", "specific"):
                df = run.per_pert[rep].get(m)
                ks = (
                    dd.deg_counts(truth_for(run_path, rep)[1], exclude_target)
                    if has_manifest
                    else df["n_deg"]
                    if df is not None
                    else None
                )
                if df is not None and metric in df:
                    k = ks.index[ks >= min_degs]
                    r[rep] = df.loc[df.index.intersection(k), metric].median()
            both.append(r)
        both = pd.DataFrame(both).set_index("method").reindex(stats["method"])
        fig2 = go.Figure()
        for rep, color, symbol in (("full", SERIES[0], "circle"), ("specific", SERIES[1], "diamond")):
            if rep in both:
                fig2.add_scatter(
                    x=both[rep],
                    y=both.index,
                    mode="markers",
                    name=REPORT_LABELS[rep],
                    marker={"color": color, "size": 11, "symbol": symbol, "line": {"color": "white", "width": 2}},
                    hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra>" + rep + "</extra>",
                )
        reference_lines(fig2, None, vertical=True)
        fig2.update_xaxes(title=f"{METRICS[metric]} (median)")
        fig2.update_yaxes(categoryorder="array", categoryarray=both.index.tolist())
        style(fig2, 80 + 34 * len(both)).update_layout(legend={"orientation": "h", "y": 1.08})
        st.plotly_chart(fig2, width="stretch", key="overview_reports")

    st.subheader("Summary table")
    summ = run.summary[report]
    cols = [
        c
        for c in summ.columns
        if c.endswith("_median")
        or c in ("n_perturbations", "shared_strength_corr", "sign_acc_pooled", "target_gene_down_frac")
    ]
    st.dataframe(
        summ.loc[summ.index.intersection(shown), cols].sort_values(f"{metric}_median", ascending=False)
        if f"{metric}_median" in summ
        else summ.loc[summ.index.intersection(shown), cols],
        width="stretch",
    )
    st.caption("The table is evaluate.py's summary (its own min-DEG filter); the charts above use the sidebar filter.")

# ================================================================ Strength & distributions
elif view == "Strength & distributions":
    edges_txt = st.text_input("Strength bin edges (true DEGs per perturbation)", "5, 20, 100, 500")
    try:
        edges = sorted({float(x) for x in edges_txt.split(",") if x.strip()}) + [np.inf]
    except ValueError:
        st.error("Bin edges must be comma-separated numbers.")
        st.stop()
    table = dd.strength_table({m: per_pert[m] for m in shown if m in per_pert}, n_true, edges, metric)
    sizes = dd.bin_sizes(n_true, edges)
    xlabels = [f"{b}<br>(n={int(sizes[b])})" for b in table.columns]
    fig = go.Figure()
    for m in table.index:
        is_ceiling, is_ctrl = m == dd.CEILING, m.endswith("(shuffled)")
        color = CEIL if is_ceiling else colors.get(entity(m), NEUTRAL)
        fig.add_scatter(
            x=xlabels,
            y=table.loc[m],
            mode="lines+markers",
            name=m,
            line={"color": color, "width": 2, "dash": "dash" if (is_ceiling or is_ctrl) else "solid"},
            marker={"size": 9, "color": color, "line": {"color": "white", "width": 2}},
            hovertemplate="<b>" + m + "</b><br>%{x}<br>median %{y:.3f}<extra></extra>",
        )
    reference_lines(fig, None)
    fig.update_yaxes(title=f"median {METRICS[metric].lower()}")
    fig.update_xaxes(title="true DEGs per perturbation")
    style(fig, 460).update_layout(legend={"x": 1.01, "y": 1})
    st.plotly_chart(fig, width="stretch", key="strength_lines")
    if any(entity(m) not in colors for m in table.index if m != dd.CEILING):
        st.caption("Methods beyond the 8 palette colors are drawn in gray; narrow the selection to tell them apart.")

    st.subheader("Per-perturbation scores")
    order = sorted([m for m in shown if m in per_pert], key=lambda m: scores(m).median(), reverse=True)
    rng = np.random.default_rng(0)
    fig2 = go.Figure()
    for i, m in enumerate(order):
        s = scores(m)
        fig2.add_box(
            y=s,
            x=[i] * len(s),
            name=m,
            boxpoints=False,
            fillcolor="rgba(0,0,0,0)",
            width=0.6,
            line={"color": CEIL, "width": 1.5},
            showlegend=False,
            hoverinfo="skip",
        )
        fig2.add_scatter(
            x=i + rng.uniform(-0.25, 0.25, len(s)),
            y=s,
            mode="markers",
            showlegend=False,
            marker={"color": SERIES[0], "size": 6, "opacity": 0.55},
            customdata=np.stack([s.index, n_true.reindex(s.index).to_numpy()], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + m + ": %{y:.3f}<br>%{customdata[1]} DEGs<extra></extra>",
        )
    fig2.update_xaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=order)
    fig2.update_yaxes(title=METRICS[metric])
    reference_lines(fig2, None)
    style(fig2, 480).update_layout(dragmode="select")
    handle_selection(
        st.plotly_chart(fig2, width="stretch", key="dist_points", on_select="rerun", selection_mode="points")
    )
    st.caption("Box = median and quartiles; each dot is one perturbation (click one to open it).")

# ================================================================ Head-to-head
elif view == "Head-to-head":
    opts = [m for m in shown if m in per_pert]
    if len(opts) < 2:
        st.info("Select at least two methods in the sidebar.")
        st.stop()
    c1, c2 = st.columns(2)
    a = c1.selectbox("Method A (y-axis)", opts, index=0)
    b = c2.selectbox("Method B (x-axis)", opts, index=1 if opts[0] == a else 0)
    df = pd.concat([scores(a).rename("a"), scores(b).rename("b")], axis=1).dropna()
    df["n_deg"] = n_true.reindex(df.index)
    wins, losses = int((df["a"] > df["b"]).sum()), int((df["a"] < df["b"]).sum())
    st.markdown(
        f"**{a}** is better on **{wins}** perturbations, **{b}** on **{losses}** (ties {len(df) - wins - losses})."
    )
    fig = go.Figure()
    fig.add_scatter(
        x=df["b"],
        y=df["a"],
        mode="markers",
        marker={
            "size": 9,
            "color": np.log10(df["n_deg"].clip(lower=1)),
            "opacity": 0.8,
            "colorscale": [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
            "colorbar": {"title": "DEGs", "tickvals": [1, 2, 3], "ticktext": ["10", "100", "1000"]},
            "line": {"color": "white", "width": 1},
        },
        customdata=np.stack([df.index, df["n_deg"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>" + a + ": %{y:.3f}<br>" + b + ": %{x:.3f}"
        "<br>%{customdata[1]} DEGs<extra></extra>",
    )
    lo = float(min(df["a"].min(), df["b"].min(), 0.4))
    fig.add_scatter(
        x=[lo, 1], y=[lo, 1], mode="lines", line={"color": CEIL, "dash": "dot"}, hoverinfo="skip", showlegend=False
    )
    fig.update_xaxes(title=b, range=[lo, 1.01])
    fig.update_yaxes(title=a, range=[lo, 1.01])
    style(fig, 560).update_layout(showlegend=False)
    handle_selection(st.plotly_chart(fig, width="stretch", key="h2h", on_select="rerun", selection_mode="points"))
    st.caption("Points above the diagonal: A is better. Color = number of true DEGs. Click a point to open it.")
