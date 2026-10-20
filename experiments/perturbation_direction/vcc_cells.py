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
    rng = np.random.default_rng(seed)
    def sample(adata):
        labels = adata.obs[key].astype(str).to_numpy()
        indices = []
        for label in sorted(set(labels)):
            rows = np.flatnonzero(labels == label)
            if len(rows) < n_cells:
                raise ValueError(f"{label} has {len(rows)} cells, fewer than requested {n_cells}")
            indices.extend(rng.choice(rows, n_cells, replace=False))
        return adata[indices, real.var_names].copy()
    return sample(real), sample(pred)


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
    result = compute_metrics(pred, real, config=cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.out_dir / "metrics_raw.csv")
    aggregate_metrics(result).write_csv(args.out_dir / "summary_raw.csv")
    metric_coverage(result.to_dicts(), len(panel)).to_csv(args.out_dir / "coverage.csv", index=False)
    manifest = {"expected_source_commit": PIN, "version": version("cell-eval2"),
                "config": asdict(cfg), "n_cells": args.n_cells, "seed": args.seed,
                "panel": sorted(set(real.obs[args.pert_key].astype(str)) - {args.control}),
                "genes": list(real.var_names),
                "note": "RAW Replogle metrics with CPU scanpy DE backend, not official anchored VCC scores",
                "inputs": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for k, p in (("real", args.real), ("pred", args.pred))}}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(aggregate_metrics(result))


if __name__ == "__main__":
    main()
