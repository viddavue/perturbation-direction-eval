"""Alternative *target geometry* for the observed effect, for the scDEBART comparison.

Why
---
scDEBART predicts a log2 fold change of normalised expression. Our default target is a
difference of mean log1p(CP10k) values. Those are different quantities, and on sparse genes the
difference matters. This script rebuilds the observed effect under the log2-ratio geometry so the
same predictions can be scored both ways, without touching the existing truth directory.

Definition used here
--------------------
Pseudobulk CP10k with a Dirichlet-style pseudocount, so that genes absent from both arms give a
ratio of 1 (and therefore a log2 effect of 0) instead of an exploding value:

    freq_p(g) = (sum_p(g) + prior) / (lib_p + prior * n_genes)
    cp10k_p(g) = freq_p(g) * 1e4
    delta(g)  = log2(cp10k_p(g) / cp10k_c(g))

Cells are halved with the same RNG sequence as ``ground_truth.py``, so the split-half ceiling
here is directly comparable to the one in the default truth directory. As a guard against silent
drift, the script recomputes the log1p half-deltas and checks them against the ones already on
disk.

The DEG mask is *not* recomputed: this is a geometry ablation, and changing the mask at the same
time would confound the two.

Example:
    python experiments/perturbation_direction/target_variants.py \
        --pert-h5ad _artifacts/datasets/replogle_state/replogle_test.h5ad \
        --ctrl-h5ad _artifacts/datasets/replogle_state/replogle_train.h5ad \
        --layer X --pert-key gene --control non-targeting --context cell_line=hepg2 \
        --metadata-json metadata/replogle_train.json --min-cells 15 \
        --truth-dir outputs/perturbation_direction/replogle/truth \
        --out-dir outputs/perturbation_direction/replogle/truth_log2ratio
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ground_truth import load_cells, lognorm, parse_context  # noqa: E402
from metrics import direction_metrics  # noqa: E402


def pseudobulk_cp10k(adata: ad.AnnData, key: str, prior: float = 0.5) -> pd.DataFrame:
    """Group x gene normalised frequency, times 1e4, with a ``prior`` pseudocount per gene."""
    labels = adata.obs[key].astype(str)
    cats = pd.Index(sorted(labels.unique()))
    ind = sparse.csr_matrix(
        (np.ones(adata.n_obs), (cats.get_indexer(labels), np.arange(adata.n_obs))), shape=(len(cats), adata.n_obs)
    )
    counts = adata.layers["counts"]
    sums = np.asarray((ind @ counts).todense()) if sparse.issparse(counts) else ind @ counts
    libs = sums.sum(axis=1)
    n_genes = sums.shape[1]
    freq = (sums + prior) / (libs[:, None] + prior * n_genes)
    return pd.DataFrame(freq * 1e4, index=cats, columns=adata.var_names)


def log2_effect(means: pd.DataFrame, perts: list[str], control: str) -> pd.DataFrame:
    """log2(perturbed / control) on pseudobulk CP10k; 0 where both arms are absent."""
    p = means.loc[perts].to_numpy()
    c = means.loc[control].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.log2(p / c)
    delta[~np.isfinite(delta)] = 0.0
    return pd.DataFrame(delta, index=perts, columns=means.columns)


def ceiling_rows(delta_a: pd.DataFrame, delta_b: pd.DataFrame, deg_a: pd.DataFrame, perts: list[str]) -> pd.DataFrame:
    """Split-half noise ceiling under whatever geometry ``delta_a``/``delta_b`` encode."""
    rows = []
    for p in perts:
        rows.append(
            {
                "perturbation": p,
                **direction_metrics(delta_b.loc[p].to_numpy(), delta_a.loc[p].to_numpy(), deg_a.loc[p].to_numpy()),
            }
        )
    return pd.DataFrame(rows).set_index("perturbation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pert-h5ad", required=True)
    ap.add_argument("--ctrl-h5ad", required=True)
    ap.add_argument("--layer", default="X")
    ap.add_argument("--pert-key", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--context", nargs="*")
    ap.add_argument("--metadata-json", required=True)
    ap.add_argument("--min-cells", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", type=float, default=0.5, help="pseudocount per gene in the CP10k estimate")
    ap.add_argument("--truth-dir", required=True, help="existing truth dir; its DEG mask and halves are reused")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    truth_dir, out = pathlib.Path(args.truth_dir), pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = parse_context(args.context)
    genes = json.loads(pathlib.Path(args.metadata_json).read_text())["genes"]

    delta_ref = pd.read_parquet(truth_dir / "delta_true.parquet")
    deg = pd.read_parquet(truth_dir / "deg.parquet")
    perts = [p for p in delta_ref.index if p in set(deg.index)]

    # Rebuild the cell table in the same order ground_truth.py used, so the half split matches.
    pert = load_cells(args.pert_h5ad, args.layer, context, {})
    pert = pert[pert.obs[args.pert_key].astype(str) != args.control].copy()
    ctrl = load_cells(args.ctrl_h5ad, args.layer, context, {args.pert_key: [args.control]})
    missing = sorted(set(genes) - set(pert.var_names))
    if missing:
        raise ValueError(f"{len(missing)} metadata genes missing from data, e.g. {missing[:5]}")

    adata = ad.concat([pert[:, genes], ctrl[:, genes]], join="inner", index_unique="-")
    adata.obs[args.pert_key] = adata.obs[args.pert_key].astype(str)
    adata = lognorm(adata)

    rng = np.random.default_rng(args.seed)
    half = np.empty(adata.n_obs, dtype="<U1")
    for _, idx in adata.obs.groupby(args.pert_key, observed=True).indices.items():
        idx = rng.permutation(idx)
        half[idx[: len(idx) // 2]], half[idx[len(idx) // 2 :]] = "A", "B"
    adata.obs["grp"] = adata.obs[args.pert_key].astype(str) + "|" + half

    # --- validation: the full-data log1p effect must reproduce delta_true exactly ------------
    log_full = _mean_log1p(adata, args.pert_key)
    check_full = log_full.loc[perts] - log_full.loc[args.control]
    drift = float(
        np.nanmax(np.abs(check_full[genes].to_numpy() - delta_ref.reindex(index=perts, columns=genes).to_numpy()))
    )
    if drift > 1e-6:
        raise RuntimeError(
            f"reconstructed log1p effects differ from {truth_dir}/delta_true.parquet (max |diff| {drift:.2e}); "
            "the cells, genes or gene order changed, so this variant would not be comparable"
        )

    # --- both geometries on ONE half split, so the geometry comparison is not confounded ------
    log_means_a = _mean_log1p(adata[half == "A"], "grp")
    log_means_b = _mean_log1p(adata[half == "B"], "grp")
    log_a = log_means_a.loc[[f"{p}|A" for p in perts]] - log_means_a.loc[f"{args.control}|A"]
    log_b = log_means_b.loc[[f"{p}|B" for p in perts]] - log_means_b.loc[f"{args.control}|B"]
    log_a.index = log_b.index = perts

    means_a = pseudobulk_cp10k(adata[half == "A"], "grp", prior=args.prior)
    means_b = pseudobulk_cp10k(adata[half == "B"], "grp", prior=args.prior)
    delta_a = log2_effect(means_a, [f"{p}|A" for p in perts], f"{args.control}|A")
    delta_b = log2_effect(means_b, [f"{p}|B" for p in perts], f"{args.control}|B")
    delta_a.index = delta_b.index = perts
    delta = log2_effect(pseudobulk_cp10k(adata, args.pert_key, prior=args.prior), perts, args.control)

    delta.to_parquet(out / "delta_true.parquet")
    deg.to_parquet(out / "deg.parquet")
    ceiling_rows(delta_a, delta_b, deg, perts).to_csv(out / "ceiling.csv")
    # the same split under the default geometry: separates "geometry" from "split" in the ceiling
    ceiling_rows(log_a, log_b, deg, perts).to_csv(out / "ceiling_log1p_samesplit.csv")
    shutil.copyfile(truth_dir / "conditions.csv", out / "conditions.csv")

    info = json.loads((truth_dir / "info.json").read_text())
    info["target"] = "log2_ratio_pseudobulk_cp10k"
    info["prior"] = args.prior
    info["derived_from"] = str(truth_dir)
    info["half_split_seed"] = args.seed
    info["full_data_log1p_check_max_abs_diff"] = drift
    info["ceiling_note"] = (
        "ceiling.csv is this target's ceiling on the split with half_split_seed; "
        "ceiling_log1p_samesplit.csv is the default geometry on the same split, so the two differ only "
        "by target geometry. The ceiling in the parent truth dir used a different split."
    )
    (out / "info.json").write_text(json.dumps(info, indent=1))

    print(f"target log2-ratio; {len(perts)} perturbations, {len(genes)} genes, prior={args.prior}")
    print(f"full-data log1p check passed (max |diff| {drift:.2e})")
    print(f"wrote {out}")
    return 0


def _mean_log1p(adata: ad.AnnData, key: str) -> pd.DataFrame:
    from ground_truth import group_means

    return group_means(adata, key)


if __name__ == "__main__":
    raise SystemExit(main())
