"""Per-knockdown gap between two methods, plotted against knockdown strength, one panel per report.

For every knockdown on a report's committed shared panel, the gap is ``score(method_a) - score(method_b)``
(balanced sign accuracy for the full-effect and perturbation-specific reports, detect AUROC for the
changed-vs-unchanged report). The x-axis is the knockdown's number of DEGs in the full-effect report of the
selected version (target excluded), on a log scale, so a knockdown sits at the same x in every panel.
Points are individual knockdowns; the line joins the median gap in fixed log-spaced bins, and each bin is
annotated with its number of knockdowns. Weak knockdowns have few DEGs and therefore noisy scores, so the
points fan out to the left; read the binned medians, not single points.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.evaluation.freeze_results import sha256

# report -> (per-knockdown table directory in the eval dir, metric column, panel title)
REPORTS = {
    "full": ("per_perturbation", "balanced_acc", "Full effect\n(balanced sign accuracy)"),
    "specific": ("per_perturbation_specific", "balanced_acc", "Knockdown-specific\n(balanced sign accuracy)"),
    "unchanged": ("per_perturbation_unchanged", "detect_auroc", "Changed vs. unchanged\n(AUROC)"),
}

# Reference palette of the dataviz guidance: series-1 blue for the one data series, neutral inks otherwise.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
POINT = "#9a9993"


def method_file(eval_dir: pathlib.Path, table_dir: str, method: str) -> pathlib.Path:
    """Per-knockdown CSV of a method, as evaluate.py names it (spaces become underscores)."""
    return eval_dir / table_dir / f"{method.replace(' ', '_')}.csv"


def paired_gaps(
    eval_dir: pathlib.Path, panel: pd.DataFrame, version: str, method_a: str, method_b: str
) -> pd.DataFrame:
    """Long table: report, perturbation, n_deg (full-effect, shared panel), score_a, score_b, gap."""
    strength = panel[panel["report"] == "full"].set_index("perturbation")[f"n_deg_{version}"]
    rows = []
    for report, (table_dir, metric, _) in REPORTS.items():
        shared = panel[(panel["report"] == report) & panel["shared"].astype(bool)]["perturbation"]
        a = pd.read_csv(method_file(eval_dir, table_dir, method_a), index_col="perturbation")[metric]
        b = pd.read_csv(method_file(eval_dir, table_dir, method_b), index_col="perturbation")[metric]
        df = pd.DataFrame({"score_a": a.reindex(shared), "score_b": b.reindex(shared)})
        df = df[np.isfinite(df["score_a"]) & np.isfinite(df["score_b"])]
        df["gap"] = df["score_a"] - df["score_b"]
        df["n_deg"] = strength.reindex(df.index).to_numpy()
        df["report"] = report
        rows.append(df.reset_index())
    out = pd.concat(rows, ignore_index=True)
    if out["n_deg"].isna().any() or (out["n_deg"] <= 0).any():
        raise ValueError("every plotted knockdown needs a positive full-effect DEG count")
    return out


def binned_medians(df: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    """Median gap per bin [lo, hi); the last bin includes its upper edge."""
    edges = np.asarray(edges, dtype=float)
    if df["n_deg"].min() < edges[0] or df["n_deg"].max() > edges[-1]:
        raise ValueError(f"bin edges {edges.tolist()} do not cover DEG counts {df['n_deg'].min()}-{df['n_deg'].max()}")
    idx = np.clip(np.searchsorted(edges, df["n_deg"].to_numpy(), side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        g = df["gap"].to_numpy()[idx == i]
        rows.append(
            {
                "lo": edges[i],
                "hi": edges[i + 1],
                "x": float(np.sqrt(edges[i] * edges[i + 1])),
                "n": int(len(g)),
                "median_gap": float(np.median(g)) if len(g) else np.nan,
                "share_a_wins": float(np.mean(g > 0)) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot(gaps: pd.DataFrame, bins: dict[str, pd.DataFrame], label_a: str, label_b: str, edges, out: pathlib.Path):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, len(REPORTS), figsize=(7.2, 2.9), sharex=True, sharey=True)
    lim = float(np.nanmax(np.abs(gaps["gap"]))) * 1.08
    for ax, (report, (_, _, title)) in zip(axes, REPORTS.items(), strict=True):
        df, b = gaps[gaps["report"] == report], bins[report]
        ax.axhline(0, color=INK_SECONDARY, lw=0.8, zorder=1)
        ax.scatter(df["n_deg"], df["gap"], s=9, color=POINT, alpha=0.55, lw=0, zorder=2)
        ax.plot(b["x"], b["median_gap"], color=SERIES, lw=2, marker="o", ms=4, zorder=3)
        for i, (x, n) in enumerate(zip(b["x"], b["n"], strict=True)):  # knockdowns per bin; "n =" only on the first
            label = f"n = {int(n)}" if i == 0 else str(int(n))
            ax.text(x, -lim * 0.97, label, ha="center", va="bottom", fontsize=6, color=INK_SECONDARY)
        ax.set_xscale("log")
        ax.set_xlim(edges[0] * 0.9, edges[-1] * 1.1)
        ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("DEGs of the knockdown (log scale)")
        ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"{label_a} − {label_b}")
    axes[0].text(edges[0], lim * 0.95, f"↑ {label_a} better", fontsize=6.5, va="top", color=INK_SECONDARY)
    axes[0].text(edges[0], -lim * 0.80, f"↓ {label_b} better", fontsize=6.5, va="bottom", color=INK_SECONDARY)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), dpi=220)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. version=all")
    args = parser.parse_args(argv)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))

    eval_dir, panel_csv, out_dir = (pathlib.Path(cfg[k]) for k in ("eval_dir", "panel_csv", "out_dir"))
    panel = pd.read_csv(panel_csv)
    gaps = paired_gaps(eval_dir, panel, cfg.version, cfg.method_a, cfg.method_b)
    bins = {r: binned_medians(gaps[gaps["report"] == r], list(cfg.bin_edges)) for r in REPORTS}
    plot(gaps, bins, cfg.label_a, cfg.label_b, list(cfg.bin_edges), out_dir / "strength_gap")

    gaps.to_csv(out_dir / "strength_gap_points.csv", index=False)
    pd.concat(bins, names=["report"]).reset_index(level=0).to_csv(out_dir / "strength_gap_bins.csv", index=False)
    inputs = [panel_csv] + [
        method_file(eval_dir, d, m) for d, _, _ in REPORTS.values() for m in (cfg.method_a, cfg.method_b)
    ]
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "n_per_report": gaps.groupby("report").size().to_dict(),
                "inputs_sha256": {str(p): sha256(p) for p in inputs},
            },
            indent=2,
        )
    )
    print(pd.concat(bins, names=["report"]).to_string())


if __name__ == "__main__":
    main()
