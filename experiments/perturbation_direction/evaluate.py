"""Score predicted perturbation effect directions against observed ones.

Inputs:
  --truth-dir   output of ground_truth.py (delta_true, deg, ceiling, info)
  --pred        NAME=PATH, generate.py parquet of per-condition profiles (control row included);
                effects are row - control row. Repeatable.
  --baseline    NAME=PATH, perturbation x gene effect parquet (e.g. baselines.py output). Repeatable.

For every prediction a "shuffled" specificity control is added: each perturbation is scored with the
prediction made for a different perturbation. If it scores like the matched prediction, the model's
direction calls are not perturbation-specific.

Writes summary.csv (one row per method), by_strength.csv, per_perturbation/<method>.csv and plots.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from metrics import evaluate, shuffled_prediction, summarize

STRENGTH_BINS = [5, 20, 100, 500, np.inf]
# reference categorical palette (fixed order, light mode) + text/grid tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def parse_named(items):
    return dict(item.split("=", 1) for item in (items or []))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_by_strength(by_strength: pd.DataFrame, path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    bins = list(by_strength.columns)
    x = np.arange(len(bins))
    # ceiling drawn first (underneath); model colors follow the method's position, not draw order
    methods = [m for m in by_strength.index if not m.startswith("noise ceiling")]
    order = [m for m in by_strength.index if m.startswith("noise ceiling")] + methods
    for method in order:
        row = by_strength.loc[method]
        is_ceiling = method.startswith("noise ceiling")
        color = TEXT_2 if is_ceiling else SERIES[methods.index(method) % len(SERIES)]
        ls = "--" if "shuffled" in method or is_ceiling else "-"
        ax.plot(
            x,
            row.to_numpy(),
            ls,
            color=color,
            lw=2,
            marker="o",
            ms=11 if is_ceiling else 8,
            mec=SURFACE,
            mew=2,
            label=method,
            zorder=2 if is_ceiling else 3,
        )
    ax.axhline(0.5, color=TEXT_2, lw=1, ls=":")
    ax.text(len(bins) - 1, 0.505, "chance", color=TEXT_2, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x, bins)
    ax.set_xlabel("true DEGs per perturbation", color=TEXT_2, fontsize=9)
    ax.set_ylabel("median balanced sign accuracy", color=TEXT_2, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Direction accuracy by perturbation strength", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_head_to_head(a: pd.Series, b: pd.Series, name_a: str, name_b: str, path: pathlib.Path):
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls=":")
    ax.scatter(df["b"], df["a"], s=22, color=SERIES[0], alpha=0.7, edgecolor=SURFACE, linewidth=0.8)
    wins = int((df["a"] > df["b"]).sum())
    ax.text(
        0.02, 0.98, f"{name_a} better on {wins}/{len(df)}", transform=ax.transAxes, va="top", color=TEXT_2, fontsize=8
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{name_b}: balanced sign accuracy", color=TEXT_2, fontsize=9)
    ax.set_ylabel(f"{name_a}: balanced sign accuracy", color=TEXT_2, fontsize=9)
    ax.set_title("Per-perturbation comparison", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--pred", nargs="*", default=[], help="NAME=PATH generate.py output")
    ap.add_argument("--baseline", nargs="*", default=[], help="NAME=PATH effect matrix")
    ap.add_argument("--exclude-target", action="store_true", help="drop gene == perturbation label (CRISPRi)")
    ap.add_argument("--eps", type=float, default=0.0, help="|delta_pred| <= eps counts as no call")
    ap.add_argument("--min-degs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    truth_dir, out = pathlib.Path(args.truth_dir), pathlib.Path(args.out_dir)
    (out / "per_perturbation").mkdir(parents=True, exist_ok=True)
    info = json.loads((truth_dir / "info.json").read_text())
    delta_true = pd.read_parquet(truth_dir / "delta_true.parquet")
    deg = pd.read_parquet(truth_dir / "deg.parquet")
    perts = list(delta_true.index)
    exclude = {p: [p] for p in perts} if args.exclude_target else None

    methods: dict[str, pd.DataFrame] = {}
    target_down = {}
    for name, path in parse_named(args.pred).items():
        prof = pd.read_parquet(path)
        missing = sorted(set(perts) - set(prof.index))
        if missing:
            print(f"[{name}] no prediction for {len(missing)} perturbations, e.g. {missing[:5]}")
        delta = prof.loc[prof.index.intersection(perts)] - prof.loc[info["control"]]
        methods[name] = delta
        methods[f"{name} (shuffled)"] = shuffled_prediction(delta, seed=args.seed)
        tg = [p for p in delta.index if p in delta.columns]
        if tg:
            target_down[name] = float(np.mean([delta.loc[p, p] < 0 for p in tg]))
    for name, path in parse_named(args.baseline).items():
        methods[name] = pd.read_parquet(path)

    per_pert = {name: evaluate(d, delta_true, deg, exclude_genes=exclude, eps=args.eps) for name, d in methods.items()}
    per_pert["noise ceiling (split-half)"] = pd.read_csv(truth_dir / "ceiling.csv", index_col=0)

    # perturbation strength = number of full-data DEGs (target excluded); shared by all methods so that
    # every method is filtered and stratified on the same perturbations (the ceiling's own n_deg comes
    # from half the cells and would put perturbations in different bins)
    n_true = deg.sum(1)
    if exclude:
        n_true -= pd.Series({p: bool(deg.loc[p, p]) if p in deg.columns else False for p in deg.index})
    keep = n_true.index[n_true >= args.min_degs]

    rows, strength = {}, {}
    labels = [
        f"{lo}-{hi - 1}" if np.isfinite(hi) else f"{lo}+"
        for lo, hi in zip(STRENGTH_BINS[:-1], STRENGTH_BINS[1:], strict=True)
    ]
    bins = pd.cut(n_true, STRENGTH_BINS, right=False, labels=labels)
    for name, df in per_pert.items():
        df.to_csv(out / "per_perturbation" / f"{name.replace(' ', '_').replace('/', '_')}.csv")
        rows[name] = summarize(df.loc[df.index.intersection(keep)], min_degs=0)
        rows[name]["n_perturbations_total"] = len(df)
        strength[name] = df["balanced_acc"].groupby(bins.reindex(df.index), observed=False).median()
    summary = pd.DataFrame(rows).T
    if target_down:
        summary["target_gene_down_frac"] = pd.Series(target_down)
    cols = [
        "n_perturbations",
        "sign_acc_median",
        "balanced_acc_median",
        "sign_acc_top20_median",
        "auroc_median",
        "spearman_deg_median",
        "pearson_all_median",
        "sign_acc_pooled",
        "no_call_rate_median",
    ]
    cols += [c for c in summary.columns if c not in cols]
    summary = summary[cols]
    summary.to_csv(out / "summary.csv")
    by_strength = pd.DataFrame(strength).T
    by_strength.to_csv(out / "by_strength.csv")

    n_per_bin = bins.value_counts().reindex(labels)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(summary[cols[:9]])
        print(
            "\nmedian balanced accuracy by #true DEGs (perturbations per bin: "
            + ", ".join(f"{k}: {v}" for k, v in n_per_bin.items())
            + ")"
        )
        print(by_strength)

    plot_by_strength(by_strength, out / "accuracy_by_strength.png")
    preds = list(parse_named(args.pred))
    bases = list(parse_named(args.baseline))
    if preds and bases:
        plot_head_to_head(
            per_pert[preds[0]]["balanced_acc"],
            per_pert[bases[0]]["balanced_acc"],
            preds[0],
            bases[0],
            out / f"head_to_head_{preds[0]}_vs_{bases[0]}.png",
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
