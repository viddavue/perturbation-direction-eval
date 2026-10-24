"""Run the pinned cell-eval2 vcc2026 preset on real cell-level count matrices.

No challenge submission is made. Outputs are RAW Replogle metrics, not leaderboard scores.
Requires the optional vcc_requirements.txt environment. Counts and cell numbers are validated;
mean profiles replicated into artificial cells are not a valid input to this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from vcc_profiles import PIN


def check_counts(adata):
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("cell-level raw nonnegative integer counts required")
    if adata.var_names.has_duplicates:
        raise ValueError("duplicate gene names")


def sample_panel(real, pred, key, control, n_cells, seed):
    """Match labels/genes strictly, then sample without replacement, including independent controls.

    Predicted controls are optional for the VCC real-control protocol. Real groups may include
    extra perturbations: refusing them avoids silently changing the retrieval candidate panel.
    """
    check_counts(real)
    check_counts(pred)
    if n_cells < 2:
        raise ValueError("at least two cells per group required")
    if set(real.var_names) != set(pred.var_names):
        raise ValueError("gene sets differ; explicitly prepare a fixed shared gene universe first")
    real_groups = set(real.obs[key].astype(str))
    pred_groups = set(pred.obs[key].astype(str))
    if control not in real_groups:
        raise ValueError("reference control missing")
    if real_groups - {control} != pred_groups - {control}:
        raise ValueError("perturbation panels differ")
    # Equalise per perturbation, not across the panel: real perturbations have 30-433 cells, so a
    # fixed count would either drop the small ones or force a low count everywhere. Both arms get
    # the SAME n within a perturbation, which is what the DE comparison requires; n varies between
    # perturbations, which the DE test tolerates. Realised counts are recorded in the manifest.
    real_labels = real.obs[key].astype(str).to_numpy()
    pred_labels = pred.obs[key].astype(str).to_numpy()
    shared = {}
    for label in sorted(set(real_labels)):
        n = min(n_cells, int((real_labels == label).sum()), int((pred_labels == label).sum()))
        if n < 2:
            raise ValueError(f"{label}: too few cells in one arm for a DE test")
        shared[label] = n

    rng = np.random.default_rng(seed)
    def draw(adata, labels):
        keep = []
        for label, n in shared.items():
            rows = np.flatnonzero(labels == label)
            keep.extend(rng.choice(rows, n, replace=False))
        return adata[keep, real.var_names].copy()

    return draw(real, real_labels), draw(pred, pred_labels)


def metric_coverage(records, n_panel):
    """Count both missing and explicit NaN outputs; DE metrics may emit only eligible rows."""
    frame = pd.DataFrame(records)
    rows = []
    for metric, group in frame.groupby("metric"):
        finite = np.isfinite(group.value.to_numpy(dtype=float))
        rows.append({"metric": metric, "n_panel": n_panel, "n_rows_emitted": len(group),
                     "n_finite": int(finite.sum()), "n_unscored": n_panel - int(finite.sum())})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pert-key", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--n-cells", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1,
                        help="independent cell subsamples; the panel scores vary with the draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
    real, pred = sample_panel(ad.read_h5ad(args.real), ad.read_h5ad(args.pred), args.pert_key,
                              args.control, args.n_cells, args.seed)
    cfg = replace(EvalConfig.from_preset("vcc2026"), pert_col=args.pert_key,
                  control=args.control, device="cpu", num_threads=1)
    panel = sorted(set(real.obs[args.pert_key].astype(str)) - {args.control})
    target_map = {p: p if p in real.var_names else None for p in panel}
    # This experiment uses gene-symbol labels on a restricted HVG universe. Record targets
    # outside it explicitly; never pretend their knockdown coordinate was excluded.
    cfg = replace(cfg, target_gene_map=target_map)
    cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Repeated draws: a single subsample can make or break a DE call, so report spread as well
    # as the point value. Each repeat resamples cells independently from both arms.
    summaries, coverages = [], []
    first_summary = None
    for rep in range(args.repeats):
        real_rep, pred_rep = (
            (real, pred) if args.repeats == 1 else
            sample_panel(ad.read_h5ad(args.real), ad.read_h5ad(args.pred), args.pert_key,
                         args.control, args.n_cells, args.seed + rep)
        )
        result = compute_metrics(pred_rep, real_rep, config=cfg)
        result.write_csv(args.out_dir / f"metrics_raw_rep{rep}.csv")
        summary = aggregate_metrics(result)
        summary.write_csv(args.out_dir / f"summary_raw_rep{rep}.csv")
        if rep == 0:
            first_summary = summary
        summaries.append(summary.to_pandas().set_index("metric")["mean"].rename(rep))
        coverages.append(metric_coverage(result.to_dicts(), len(panel)).set_index("metric")["n_finite"].rename(rep))
    if args.repeats > 1:
        combined = pd.concat(summaries, axis=1)
        agg = pd.DataFrame({"mean": combined.mean(axis=1), "min": combined.min(axis=1),
                            "max": combined.max(axis=1), "n_repeats": args.repeats})
        agg.to_csv(args.out_dir / "summary_across_repeats.csv")
        pd.concat(coverages, axis=1).assign(mean=lambda d: d.mean(axis=1)).to_csv(
            args.out_dir / "coverage_across_repeats.csv")
        print(agg.round(4).to_string())
    else:
        # keep the evaluator's own table (metric, mean, agg) for the single-draw case
        first_summary.write_csv(args.out_dir / "summary_raw.csv")
        coverages[0].to_frame("n_finite").to_csv(args.out_dir / "coverage.csv")
        print(first_summary)
    manifest = {"expected_source_commit": PIN, "version": version("cell-eval2"),
                "config": asdict(cfg), "n_cells": args.n_cells, "seed": args.seed, "repeats": args.repeats,
                "panel": sorted(set(real.obs[args.pert_key].astype(str)) - {args.control}),
                "genes": list(real.var_names),
                "note": "RAW Replogle metrics with CPU scanpy DE backend, not official anchored VCC scores",
                "inputs": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for k, p in (("real", args.real), ("pred", args.pred))}}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(aggregate_metrics(result))


if __name__ == "__main__":
    main()
