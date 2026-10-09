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


def report(methods, delta_true, deg, ceiling, exclude, args, out: pathlib.Path, tag: str, target_down: dict):
    """Score every method, write summary/by_strength/per-perturbation tables and plots (suffixed by tag)."""
    (out / f"per_perturbation{tag}").mkdir(parents=True, exist_ok=True)
    per_pert = {name: evaluate(d, delta_true, deg, exclude_genes=exclude, eps=args.eps) for name, d in methods.items()}
    if ceiling is not None:
        per_pert["noise ceiling (split-half)"] = ceiling

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
        df.to_csv(out / f"per_perturbation{tag}" / f"{name.replace(' ', '_').replace('/', '_')}.csv")
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
    summary.to_csv(out / f"summary{tag}.csv")
    by_strength = pd.DataFrame(strength).T
    by_strength.to_csv(out / f"by_strength{tag}.csv")

    n_per_bin = bins.value_counts().reindex(labels)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(summary[cols[:9]])
        print(
            "\nmedian balanced accuracy by #true DEGs (perturbations per bin: "
            + ", ".join(f"{k}: {v}" for k, v in n_per_bin.items())
            + ")"
        )
        print(by_strength)

    plot_by_strength(by_strength, out / f"accuracy_by_strength{tag}.png")
    # first model vs the strongest baseline (by median balanced accuracy) in this report
    preds = [n for n in parse_named(args.pred) if n in per_pert]
    bases = [n for n in parse_named(args.baseline) if n in per_pert]
    if preds and bases:
        best = max(bases, key=lambda b: np.nan_to_num(summary.loc[b, "balanced_acc_median"], nan=-1))
        plot_head_to_head(
            per_pert[preds[0]]["balanced_acc"],
            per_pert[best]["balanced_acc"],
            preds[0],
            best,
            out / f"head_to_head_{preds[0]}_vs_{best}{tag}.png",
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--pred", nargs="*", default=[], help="NAME=PATH generate.py output")
    ap.add_argument("--baseline", nargs="*", default=[], help="NAME=PATH effect matrix")
    ap.add_argument("--exclude-target", action="store_true", help="drop gene == perturbation label (CRISPRi)")
    ap.add_argument("--eps", type=float, default=0.0, help="|delta_pred| <= eps counts as no call")
    ap.add_argument("--min-degs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--specific-ref",
        help="effect matrix of the shared response (e.g. baselines/delta_generic.parquet); adds a second report "
        "scoring the direction of (effect - shared response), i.e. the perturbation-specific part",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    truth_dir, out = pathlib.Path(args.truth_dir), pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
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

    ceiling = pd.read_csv(truth_dir / "ceiling.csv", index_col=0)
    report(methods, delta_true, deg, ceiling, exclude, args, out, "", target_down)

    if args.specific_ref:
        # perturbation-specific component: subtract the shared response from truth and every prediction
        ref = pd.read_parquet(args.specific_ref)
        ref_vec = ref.mean() if len(ref) else None
        min_lfc = info.get("min_lfc", 0.1)
        true_s = delta_true - ref_vec
        deg_s = deg & (true_s.abs() > min_lfc)
        methods_s = {n: d - ref_vec for n, d in methods.items() if not np.allclose(d.to_numpy(), ref_vec.to_numpy())}
        ceiling_s = None
        if (truth_dir / "delta_half_a.parquet").exists():
            d_a = pd.read_parquet(truth_dir / "delta_half_a.parquet") - ref_vec
            d_b = pd.read_parquet(truth_dir / "delta_half_b.parquet") - ref_vec
            deg_a = pd.read_parquet(truth_dir / "deg_half_a.parquet") & (d_a.abs() > min_lfc)
            ceiling_s = evaluate(d_b, d_a, deg_a, exclude_genes=exclude, eps=args.eps)
        print("\n=== perturbation-specific component (effect minus shared response) ===")
        report(methods_s, true_s, deg_s, ceiling_s, exclude, args, out, "_specific", {})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
