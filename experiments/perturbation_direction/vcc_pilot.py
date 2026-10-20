"""Prepare a small fixed Replogle panel and INDEPENDENT real-cell halves for metric validation.

replicate.h5ad is an empirical replicate comparator, not a model prediction. Panel selection
uses observed DEG bins for descriptive diagnostics, not hyperparameter selection. All outputs
contain genuine cells sampled without replacement; neither side shares control cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from ground_truth import load_cells, parse_context


def split_groups(adata, key, n_cells, seed):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for label, idx in adata.obs.groupby(key, observed=True, sort=True).indices.items():
        if len(idx) < 2 * n_cells:
            raise ValueError(f"{label}: need {2 * n_cells} independent cells, got {len(idx)}")
        chosen = rng.choice(idx, 2 * n_cells, replace=False)
        a.extend(chosen[:n_cells])
        b.extend(chosen[n_cells:])
    return adata[a].copy(), adata[b].copy()


def select_panel(deg, n_cells_by_pert, n_cells, per_bin, seed):
    rng = np.random.default_rng(seed)
    counts = deg.sum(1)
    bins = {"weak_or_null": (0, 5), "moderate": (5, 100), "strong": (100, np.inf)}
    rows = []
    for name, (lo, hi) in bins.items():
        candidates = sorted(p for p in counts.index
                            if lo <= counts[p] < hi and n_cells_by_pert.get(p, 0) >= 2 * n_cells)
        chosen = rng.choice(candidates, min(per_bin, len(candidates)), replace=False)
        rows.extend({"perturbation": str(p), "stratum": name, "n_deg": int(counts[p]),
                     "n_real_cells": int(n_cells_by_pert[p])} for p in sorted(chosen))
    if len(rows) < 2:
        raise ValueError("not enough eligible perturbations")
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5ad", required=True)
    parser.add_argument("--test-h5ad", required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--context", nargs="*", default=["cell_line=hepg2"])
    parser.add_argument("--pert-key", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--n-cells", type=int, default=32)
    parser.add_argument("--per-bin", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_cells < 2 or args.per_bin < 1:
        raise ValueError("n_cells >=2 and per_bin >=1 required")
    deg = pd.read_parquet(args.truth_dir / "deg.parquet")
    info = json.loads((args.truth_dir / "info.json").read_text())
    panel = select_panel(deg, info["n_cells"], args.n_cells, args.per_bin, args.seed)
    context = parse_context(args.context)
    perts = panel.perturbation.tolist()
    pert = load_cells(args.test_h5ad, "X", context, {args.pert_key: perts})
    ctrl = load_cells(args.train_h5ad, "X", context, {args.pert_key: [args.control]})
    cells = ad.concat([pert[:, deg.columns], ctrl[:, deg.columns]], index_unique="-")
    cells.obs[args.pert_key] = cells.obs[args.pert_key].astype(str)
    a, b = split_groups(cells, args.pert_key, args.n_cells, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(args.out_dir / "real.h5ad")
    b.write_h5ad(args.out_dir / "replicate.h5ad")
    panel.to_csv(args.out_dir / "panel.csv", index=False)
    conditions = pd.DataFrame({"id": [args.control, *perts], args.pert_key: [args.control, *perts]})
    for key, value in context.items():
        conditions[key] = value
    conditions.to_csv(args.out_dir / "conditions.csv", index=False)
    manifest = {"seed": args.seed, "n_cells_per_group_per_half": args.n_cells, "context": context,
                "genes": list(deg.columns), "purpose": "independent real-cell replicate, NOT model performance",
                "source": {"train": args.train_h5ad, "test": args.test_h5ad, "truth": str(args.truth_dir)},
                "panel_hash": hashlib.sha256(panel.to_csv(index=False).encode()).hexdigest(),
                "groups": {name: list(data.obs_names) for name, data in (("a", a), ("b", b))}}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(panel.to_string(index=False))


if __name__ == "__main__":
    main()
