"""Data layer for dashboard.py: discovers evaluation runs and rebuilds everything the views need.

Pure functions (no Streamlit), so they can be tested against evaluate.py's own outputs.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.evaluation.metrics import direction_metrics, remove_shared, shuffled_prediction

REPORTS = {"full": "", "specific": "_specific"}
CEILING = "noise ceiling (split-half)"


def method_filename(name: str) -> str:
    """File name evaluate.py uses for a method's per-perturbation table."""
    return name.replace(" ", "_").replace("/", "_") + ".csv"


@dataclass
class Run:
    path: pathlib.Path
    summary: dict[str, pd.DataFrame]  # report -> method x metric
    per_pert: dict[str, dict[str, pd.DataFrame]]  # report -> method -> perturbation x metric
    manifest: dict | None = None
    methods: list[str] = field(default_factory=list)  # all methods in the full report, in summary order


def find_runs(root: str | pathlib.Path) -> list[pathlib.Path]:
    """Evaluation directories (containing summary.csv) under root, newest first."""
    # A run is an evaluate.py output: summary.csv AND at least one per-knockdown table. A folder
    # that merely contains a summary.csv (vcc_profiles.py output, a hand-copied summary) has no
    # per_perturbation tables, so every view would fail to find a method to plot.
    runs = [
        p.parent for p in pathlib.Path(root).rglob("summary.csv") if any((p.parent / "per_perturbation").glob("*.csv"))
    ]
    return sorted(runs, key=lambda p: (p / "summary.csv").stat().st_mtime, reverse=True)


def load_run(eval_dir: str | pathlib.Path) -> Run:
    eval_dir = pathlib.Path(eval_dir)
    summary, per_pert = {}, {}
    for report, tag in REPORTS.items():
        f = eval_dir / f"summary{tag}.csv"
        if not f.exists():
            continue
        summary[report] = pd.read_csv(f, index_col=0)
        tables = {}
        for name in summary[report].index:
            t = eval_dir / f"per_perturbation{tag}" / method_filename(name)
            if t.exists():
                tables[name] = pd.read_csv(t, index_col=0)
        per_pert[report] = tables
    m = eval_dir / "methods.json"
    manifest = json.loads(m.read_text()) if m.exists() else None
    return Run(eval_dir, summary, per_pert, manifest, list(summary.get("full", pd.DataFrame()).index))


def resolve(entry: dict | None, base: pathlib.Path | None = None) -> pathlib.Path | None:
    """Path from a manifest entry: absolute path if it exists, else the stored path relative to base / cwd."""
    if not entry:
        return None
    for cand in (
        pathlib.Path(entry["abs_path"]),
        (base or pathlib.Path.cwd()) / entry["path"],
        pathlib.Path(entry["path"]),
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot find {entry['path']} (also tried {entry['abs_path']})")


@dataclass
class Truth:
    delta: pd.DataFrame
    deg: pd.DataFrame
    padj: pd.DataFrame | None
    info: dict

    @property
    def perturbations(self) -> list[str]:
        return list(self.delta.index)


def load_truth(truth_dir: str | pathlib.Path) -> Truth:
    d = pathlib.Path(truth_dir)
    padj = pd.read_parquet(d / "padj.parquet") if (d / "padj.parquet").exists() else None
    return Truth(
        pd.read_parquet(d / "delta_true.parquet"),
        pd.read_parquet(d / "deg.parquet"),
        padj,
        json.loads((d / "info.json").read_text()),
    )


def load_effects(manifest: dict, name: str, truth: Truth, base: pathlib.Path | None = None) -> pd.DataFrame:
    """Perturbation x gene effect matrix of a method, exactly as evaluate.py scored it."""
    entry = next(m for m in manifest["methods"] if m["name"] == name)
    if entry["kind"] == "shuffled":
        return shuffled_prediction(load_effects(manifest, entry["source"], truth, base), seed=entry.get("seed", 0))
    df = pd.read_parquet(resolve(entry, base))
    if entry["kind"] == "pred":
        df = df.loc[df.index.intersection(truth.perturbations)] - df.loc[manifest["control"]]
    return df


def shared_reference(manifest: dict, base: pathlib.Path | None = None) -> pd.Series | None:
    path = resolve(manifest.get("specific_ref"), base)
    return None if path is None else pd.read_parquet(path).mean()


def specific_truth(truth: Truth, ref: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truth effect and DEG mask with the shared response projected out (as in evaluate.py's specific report)."""
    resid, _ = remove_shared(truth.delta, ref)
    return resid, truth.deg & (resid.abs() > truth.info.get("min_lfc", 0.1))


def deg_counts(deg: pd.DataFrame, exclude_target: bool) -> pd.Series:
    """Number of DEGs per perturbation (target gene excluded), the strength measure used by evaluate.py."""
    n = deg.sum(1)
    if exclude_target:
        n -= pd.Series({p: bool(deg.loc[p, p]) if p in deg.columns else False for p in deg.index})
    return n


def strength_table(
    per_pert: dict[str, pd.DataFrame], n_true: pd.Series, edges: list[float], metric: str
) -> pd.DataFrame:
    """Median of ``metric`` per method in bins of true DEG counts (bins left-closed, like evaluate.py)."""
    labels = [
        f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f"{int(lo)}+"
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    bins = pd.cut(n_true, edges, right=False, labels=labels)
    rows = {}
    for name, df in per_pert.items():
        if metric in df:
            rows[name] = df[metric].groupby(bins.reindex(df.index), observed=False).median()
    out = pd.DataFrame(rows).T
    out.columns = labels
    return out


def bin_sizes(n_true: pd.Series, edges: list[float]) -> pd.Series:
    labels = [
        f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f"{int(lo)}+"
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return pd.cut(n_true, edges, right=False, labels=labels).value_counts().reindex(labels)


def perturbation_detail(
    pert: str,
    true_delta: pd.DataFrame,
    deg: pd.DataFrame,
    padj: pd.DataFrame | None,
    effects: dict[str, pd.DataFrame],
    exclude_target: bool = True,
    eps: float = 0.0,
):
    """Per-gene table and per-method metrics for one perturbation.

    Returns (genes, metrics): genes has true_delta, padj, deg and, per method, its predicted delta and whether the
    sign agrees; metrics are direction_metrics per method on the same DEG mask evaluate.py uses.
    """
    genes = true_delta.columns
    t = true_delta.loc[pert, genes].to_numpy()
    mask = deg.loc[pert, genes].to_numpy(bool).copy()
    if exclude_target:
        mask &= genes != pert
    table = pd.DataFrame({"true_delta": t, "deg": mask}, index=pd.Index(genes, name="gene"))
    if padj is not None and pert in padj.index:
        table.insert(1, "padj", padj.loc[pert].reindex(genes).to_numpy())
    metrics = {}
    for name, eff in effects.items():
        if pert not in eff.index:
            continue
        p = eff.loc[pert].reindex(genes).to_numpy()
        table[name] = p
        table[f"{name} ✓"] = np.where(t == 0, np.nan, (np.sign(p) == np.sign(t)).astype(float))
        metrics[name] = direction_metrics(p, t, mask, eps=eps)
    table = table.sort_values(
        ["deg", "true_delta"], key=lambda s: s.abs() if s.name == "true_delta" else s, ascending=[False, False]
    )
    return table, pd.DataFrame(metrics).T
