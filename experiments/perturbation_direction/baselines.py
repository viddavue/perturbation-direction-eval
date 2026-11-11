"""Non-generative baselines for perturbation effect direction, computed from training cells.

  transfer  For each perturbation p: the observed effect of p in every OTHER context (e.g. cell line)
            of the training data, averaged over contexts. Answers "is scLDM better than looking up the
            known effect of the same perturbation elsewhere?"
  generic   One effect vector shared by all perturbations: the mean effect over all training
            perturbations (in the target context if it has perturbed training cells, else in the
            other contexts). Catches predictions that only encode a generic "perturbed vs control" shift.

  knn       Correlation-weighted mean target-context effect of the k target-context training
            perturbations whose other-context effects are most similar; fitted on the target context's
            TRAINING perturbations only (never the test ones), k by K-fold CV.

All are written as perturbation x gene effect matrices (log1p(CP10k) differences), directly
comparable to ``delta_true.parquet`` from ground_truth.py. Per-context effects of all training
perturbations are cached as effects_<ctx>.parquet (reused unless --recompute).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ground_truth import group_means, load_cells, lognorm
from transfer_models import knn_transfer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5ad", required=True)
    ap.add_argument("--layer", default="X")
    ap.add_argument("--pert-key", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--context-key", required=True, help="e.g. cell_line")
    ap.add_argument("--target-context", required=True, help="held-out context, e.g. hepg2")
    ap.add_argument("--truth-dir", required=True, help="ground_truth.py output (perturbations + genes)")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--recompute", action="store_true", help="ignore cached effects_<ctx>.parquet")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    truth = pd.read_parquet(pathlib.Path(args.truth_dir) / "delta_true.parquet")
    perts, genes = list(truth.index), list(truth.columns)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.train_h5ad, "r") as f:
        contexts = sorted(set(read_elem(f["obs"])[args.context_key].astype(str)))

    # one context at a time keeps peak memory at the size of a single cell line
    effects: dict[str, pd.DataFrame] = {}
    for c in contexts:
        cache = out / f"effects_{c}.parquet"
        if cache.exists() and not args.recompute:
            effects[c] = pd.read_parquet(cache)
            print(f"{c}: {len(effects[c])} perturbations (cached)")
            continue
        adata = load_cells(args.train_h5ad, args.layer, {args.context_key: c}, {})[:, genes].copy()
        adata.obs[args.pert_key] = adata.obs[args.pert_key].astype(str)
        sizes = adata.obs[args.pert_key].value_counts()
        if sizes.get(args.control, 0) < args.min_cells:
            print(f"{c}: only {sizes.get(args.control, 0)} control cells, skipped")
            continue
        means = group_means(lognorm(adata), args.pert_key)
        rows = [p for p in means.index if p != args.control and sizes[p] >= args.min_cells]
        effects[c] = means.loc[rows] - means.loc[args.control]
        effects[c].to_parquet(cache)
        print(f"{c}: {adata.n_obs} cells, {len(rows)} perturbations with >= {args.min_cells} cells")
        del adata, means

    others = [c for c in effects if c != args.target_context]
    stack = pd.concat({c: effects[c].reindex(perts) for c in others})
    transfer = stack.groupby(level=1).mean().reindex(perts)  # NaN rows where p unseen in other contexts
    n_ctx = stack.notna().all(axis=1).groupby(level=1).sum().reindex(perts)
    transfer.dropna(how="all").to_parquet(out / "delta_transfer.parquet")

    in_target = effects.get(args.target_context)
    if in_target is not None and len(in_target.index.difference(perts)) > 0:
        generic_src, generic_vec = "target_context_train", in_target.loc[in_target.index.difference(perts)].mean()
    else:
        generic_src, generic_vec = "other_contexts", pd.concat([effects[c] for c in others]).mean()
    generic = pd.DataFrame(np.tile(generic_vec.to_numpy(), (len(perts), 1)), index=perts, columns=genes)
    generic.to_parquet(out / "delta_generic.parquet")

    learned = {}
    n_target_train = 0 if in_target is None else len(set(in_target.index) - set(perts))
    if n_target_train >= 10:
        leaked = sorted(set(in_target.index) & set(perts))
        if leaked:
            print(
                f"WARNING: {len(leaked)} test perturbations have target-context TRAINING cells; excluded from fitting"
            )
        train_perts = sorted(set(in_target.index) - set(perts))
        knn, learned["knn"] = knn_transfer(effects, args.target_context, others, train_perts, perts, genes)
        knn.to_parquet(out / "delta_knn.parquet")
        learned["test_perts_with_target_train_cells"] = leaked
    else:
        print(f"only {n_target_train} target-context training perturbations: knn baseline skipped")

    info = {
        "contexts_with_controls": list(effects),
        "learned_baselines": learned,
        "transfer_contexts_per_perturbation": n_ctx.fillna(0).astype(int).describe().to_dict(),
        "perturbations_without_transfer": [p for p in perts if transfer.loc[p].isna().all()],
        "generic_source": generic_src,
        "target_context_train_perturbations": 0 if in_target is None else len(in_target),
    }
    (out / "baselines_info.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float))


if __name__ == "__main__":
    main()
