"""Shared-response reference for a panel, so `evaluate.py --specific-ref` can be run.

In Replogle this vector is the mean effect of the target context's *training* perturbations: the
stress/proliferation signature most knockdowns share. `--specific-ref` then projects it out of the
truth and of every prediction, per perturbation, and scores what is left.

For Parse1M the equivalent is the mean effect of the training cytokines in the same cell type, so
this computes effects for a set of conditions and writes their mean as a one-row parquet.

Train/test separation matters: pass only training conditions. The cytokines that make up the test
panel must not contribute to the reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.evaluation.ground_truth import group_means, load_cells, lognorm, parse_context


def mean_effect(adata: ad.AnnData, key: str, conditions: list[str], control: str) -> pd.DataFrame:
    """Log1p(CP10k) difference of each condition against the control, same estimator as ground_truth.py."""
    means = group_means(adata, key)
    if control not in means.index:
        raise KeyError(f"control {control!r} not among the groups")
    missing = [c for c in conditions if c not in means.index]
    if missing:
        raise KeyError(f"{len(missing)} conditions absent from the data, e.g. {missing[:5]}")
    return means.loc[conditions] - means.loc[control]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", default="X")
    ap.add_argument("--pert-key", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--context", nargs="*", help="key=value filters, e.g. cell_type='CD4 Naive'")
    ap.add_argument("--conditions", nargs="+", required=True, help="training conditions only")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--out", required=True, help="one-row parquet holding the mean effect")
    args = ap.parse_args(argv)

    rows = load_cells(args.h5ad, args.layer, parse_context(args.context), {})
    keep = {args.pert_key: [args.control, *args.conditions]}
    rows = rows[rows.obs[args.pert_key].astype(str).isin(keep[args.pert_key])].copy()
    rows.obs[args.pert_key] = rows.obs[args.pert_key].astype(str)
    sizes = rows.obs[args.pert_key].value_counts()
    small = sorted(c for c in args.conditions if sizes.get(c, 0) < args.min_cells)
    if small:
        raise ValueError(f"conditions with fewer than {args.min_cells} cells: {small[:5]}")
    adata = lognorm(rows)

    effects = mean_effect(adata, args.pert_key, args.conditions, args.control)
    reference = effects.mean(axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([reference.to_numpy()], index=pd.Index(["shared"]), columns=effects.columns).to_parquet(out)

    info = {
        "purpose": "shared-response reference for --specific-ref (training conditions only)",
        "n_conditions": len(args.conditions),
        "control": args.control,
        "pert_key": args.pert_key,
        "context": parse_context(args.context),
        "min_cells": args.min_cells,
        "conditions": sorted(args.conditions),
        "n_genes": int(effects.shape[1]),
        "excluded_from_reference": "the test conditions are not part of this mean by construction",
        "cells": int(adata.n_obs),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"{len(args.conditions)} conditions -> shared vector over {effects.shape[1]} genes: {out}")
    print(f"largest |effect| genes: {', '.join(reference.abs().nlargest(8).index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
