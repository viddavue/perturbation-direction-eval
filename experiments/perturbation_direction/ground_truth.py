"""Observed perturbation effects, DEGs and a split-half noise ceiling from real cells.

For each perturbation p (cells with ``obs[pert_key] == p``) the reference is the control cells
(``obs[pert_key] == control``) from the same context (e.g. ``cell_line=hepg2``). Perturbed and
control cells may come from different files (Replogle: perturbed in test, controls in train).

Outputs in ``--out-dir``:
  delta_true.parquet   perturbation x gene, mean log1p(CP10k) perturbed - control
  deg.parquet          perturbation x gene, bool (BH padj < alpha and |delta| > min_lfc)
  ceiling.csv          per-perturbation direction metrics of half B vs DEGs selected in half A
  conditions.csv       conditions to generate (control first), input to generate.py
  info.json            cell counts, control library size, settings
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.io import read_elem
from scipy import sparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from metrics import direction_metrics


def parse_context(items: list[str] | None) -> dict[str, str]:
    return dict(item.split("=", 1) for item in (items or []))


def read_rows(node, rows: np.ndarray, block: int = 50_000) -> sparse.csr_matrix:
    """Read the given (sorted) rows of an on-disk CSR group or dense dataset.

    Streams in row blocks so that only the selected rows plus one block are held in memory.
    """
    n = node.attrs["shape"][0] if isinstance(node, h5py.Group) else node.shape[0]
    out = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        sel = rows[(rows >= start) & (rows < stop)] - start
        if not len(sel):
            continue
        if isinstance(node, h5py.Group):
            ip = node["indptr"][start : stop + 1]
            chunk = sparse.csr_matrix(
                (node["data"][ip[0] : ip[-1]], node["indices"][ip[0] : ip[-1]], ip - ip[0]),
                shape=(stop - start, node.attrs["shape"][1]),
            )
            out.append(chunk[sel].astype(np.float32))
        else:
            out.append(sparse.csr_matrix(node[start:stop][sel], dtype=np.float32))
    return sparse.vstack(out, format="csr") if out else sparse.csr_matrix((0, 0), dtype=np.float32)


def load_cells(path: str, layer: str, context: dict[str, str], keep: dict[str, list[str]]) -> ad.AnnData:
    """Load counts for cells matching ``context`` and whose ``obs[k]`` is in ``keep[k]``.

    Only obs/var are read eagerly; the expression matrix is streamed and subset on the fly.
    """
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        mask = np.ones(len(obs), dtype=bool)
        for k, v in context.items():
            mask &= obs[k].astype(str).to_numpy() == v
        for k, vals in keep.items():
            mask &= obs[k].astype(str).isin(vals).to_numpy()
        rows = np.flatnonzero(mask)
        X = read_rows(f["X"] if layer == "X" else f["layers"][layer], rows)
    sample = X[: min(500, X.shape[0])].data
    if sample.size and not np.allclose(sample, np.round(sample)):
        raise ValueError(f"{path}:{layer} does not look like raw counts (non-integer values)")
    return ad.AnnData(X=X, obs=obs.iloc[rows].copy(), var=pd.DataFrame(index=var.index.astype(str)))


def lognorm(adata: ad.AnnData) -> ad.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def group_means(adata: ad.AnnData, key: str) -> pd.DataFrame:
    """Per-group mean of adata.X (log-normalised), via a sparse indicator matmul."""
    labels = adata.obs[key].astype(str)
    cats = pd.Index(sorted(labels.unique()))
    ind = sparse.csr_matrix(
        (np.ones(adata.n_obs), (cats.get_indexer(labels), np.arange(adata.n_obs))), shape=(len(cats), adata.n_obs)
    )
    sums = np.asarray((ind @ adata.X).todense()) if sparse.issparse(adata.X) else ind @ adata.X
    counts = np.asarray(ind.sum(1)).ravel()
    return pd.DataFrame(sums / counts[:, None], index=cats, columns=adata.var_names)


def group_stats(adata: ad.AnnData, key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per-group mean, unbiased variance and size of adata.X, via sparse indicator matmuls."""
    labels = adata.obs[key].astype(str)
    cats = pd.Index(sorted(labels.unique()))
    ind = sparse.csr_matrix(
        (np.ones(adata.n_obs), (cats.get_indexer(labels), np.arange(adata.n_obs))), shape=(len(cats), adata.n_obs)
    )
    X = adata.X
    dense = lambda m: np.asarray(m.todense()) if sparse.issparse(m) else np.asarray(m)
    sums = dense(ind @ X)
    sq = dense(ind @ (X.multiply(X) if sparse.issparse(X) else X * X))
    n = np.asarray(ind.sum(1)).ravel()
    mean = sums / n[:, None]
    var = np.clip(sq - n[:, None] * mean**2, 0, None) / np.maximum(n - 1, 1)[:, None]
    return (
        pd.DataFrame(mean, index=cats, columns=adata.var_names),
        pd.DataFrame(var, index=cats, columns=adata.var_names),
        pd.Series(n, index=cats),
    )


def welch_degs(mean, var, n, control: str, perts: list[str], alpha: float, min_lfc: float):
    """Effects and DEG masks for each perturbation vs control: Welch t-test, BH within perturbation, |delta| > min_lfc.

    Returns (delta, padj) as perturbation x gene frames.
    """
    from metrics import benjamini_hochberg
    from scipy.stats import t as t_dist

    delta = mean.loc[perts] - mean.loc[control]
    se2_p = var.loc[perts].to_numpy() / n.loc[perts].to_numpy()[:, None]
    se2_c = var.loc[control].to_numpy() / n.loc[control]
    se = np.sqrt(se2_p + se2_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = delta.to_numpy() / se
        df = (se2_p + se2_c) ** 2 / (
            se2_p**2 / np.maximum(n.loc[perts].to_numpy()[:, None] - 1, 1) + se2_c**2 / max(n.loc[control] - 1, 1)
        )
    p = 2 * t_dist.sf(np.abs(np.nan_to_num(tstat)), np.nan_to_num(df, nan=1.0))
    p[se == 0] = 1.0  # genes without variance in both groups carry no evidence
    padj = np.vstack([benjamini_hochberg(row) for row in p])
    return delta, pd.DataFrame(padj, index=delta.index, columns=delta.columns)


def wilcoxon_padj(adata: ad.AnnData, key: str, groups: list[str], reference: str) -> pd.DataFrame:
    """BH-adjusted Wilcoxon p-values, group x gene."""
    sc.tl.rank_genes_groups(
        adata,
        key,
        groups=groups,
        reference=reference,
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        tie_correct=True,
        n_genes=adata.n_vars,
        use_raw=False,
    )
    res = adata.uns["rank_genes_groups"]
    out = {}
    for g in groups:
        out[g] = pd.Series(res["pvals_adj"][g], index=res["names"][g]).reindex(adata.var_names).to_numpy()
    del adata.uns["rank_genes_groups"]
    return pd.DataFrame(out, index=adata.var_names).T


def effects_and_degs(adata, key, perts, control, alpha, min_lfc):
    means = group_means(adata, key)
    delta = means.loc[perts] - means.loc[control]
    padj = wilcoxon_padj(adata, key, perts, control)
    deg = (padj < alpha) & (delta.abs() > min_lfc)
    return delta, padj, deg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pert-h5ad", required=True)
    ap.add_argument("--ctrl-h5ad", required=True)
    ap.add_argument("--layer", default="X", help="'X' or a layer holding raw counts")
    ap.add_argument("--pert-key", required=True, help="obs column with perturbation labels (e.g. gene)")
    ap.add_argument("--control", required=True, help="control label (e.g. non-targeting)")
    ap.add_argument("--context", nargs="*", help="key=value filters applied to all cells, e.g. cell_line=hepg2")
    ap.add_argument("--perturbations", nargs="*", help="subset of perturbations (default: all in --pert-h5ad)")
    ap.add_argument("--metadata-json", required=True, help="TRAIN metadata json; genes are restricted/ordered to it")
    ap.add_argument("--min-cells", type=int, default=20, help="min cells per half for a perturbation to be kept")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-lfc", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = parse_context(args.context)
    genes = json.loads(pathlib.Path(args.metadata_json).read_text())["genes"]

    pert_filter = {args.pert_key: args.perturbations} if args.perturbations else {}
    pert = load_cells(args.pert_h5ad, args.layer, context, pert_filter)
    pert = pert[pert.obs[args.pert_key].astype(str) != args.control].copy()
    ctrl = load_cells(args.ctrl_h5ad, args.layer, context, {args.pert_key: [args.control]})
    if ctrl.n_obs < 2 * args.min_cells:
        raise ValueError(f"only {ctrl.n_obs} control cells for context {context}")
    missing = sorted(set(genes) - set(pert.var_names))
    if missing:
        raise ValueError(f"{len(missing)} metadata genes missing from data, e.g. {missing[:5]}")

    adata = ad.concat([pert[:, genes], ctrl[:, genes]], join="inner", index_unique="-")
    adata.obs[args.pert_key] = adata.obs[args.pert_key].astype(str)
    ctrl_lib = float(np.median(np.asarray(adata[adata.obs[args.pert_key] == args.control].X.sum(1)).ravel()))
    adata = lognorm(adata)

    n_cells = adata.obs[args.pert_key].value_counts()
    perts = sorted(p for p in n_cells.index if p != args.control and n_cells[p] >= 2 * args.min_cells)
    dropped = sorted(p for p in n_cells.index if p != args.control and p not in perts)
    print(
        f"{len(perts)} perturbations kept, {len(dropped)} dropped (<{2 * args.min_cells} cells), "
        f"{n_cells[args.control]} control cells"
    )

    # full-data effects and DEGs
    delta, padj, deg = effects_and_degs(adata, args.pert_key, perts, args.control, args.alpha, args.min_lfc)
    delta.to_parquet(out / "delta_true.parquet")
    padj.to_parquet(out / "padj.parquet")
    deg.to_parquet(out / "deg.parquet")

    # split-half ceiling: DEGs + direction from half A, scored on half B (independent cells)
    rng = np.random.default_rng(args.seed)
    half = np.empty(adata.n_obs, dtype="<U1")
    for _, idx in adata.obs.groupby(args.pert_key, observed=True).indices.items():
        idx = rng.permutation(idx)
        half[idx[: len(idx) // 2]], half[idx[len(idx) // 2 :]] = "A", "B"
    adata.obs["grp"] = adata.obs[args.pert_key].astype(str) + "|" + half
    d_a, _, deg_a = effects_and_degs(
        adata, "grp", [f"{p}|A" for p in perts], f"{args.control}|A", args.alpha, args.min_lfc
    )
    means_b = group_means(adata[half == "B"], "grp")
    d_b = means_b.loc[[f"{p}|B" for p in perts]] - means_b.loc[f"{args.control}|B"]
    rows = []
    for p in perts:
        m = direction_metrics(
            d_b.loc[f"{p}|B"].to_numpy(), d_a.loc[f"{p}|A"].to_numpy(), deg_a.loc[f"{p}|A"].to_numpy()
        )
        rows.append({"perturbation": p, **m})
    pd.DataFrame(rows).set_index("perturbation").to_csv(out / "ceiling.csv")
    # halves saved so evaluate.py can build ceilings for derived targets (e.g. effect minus generic shift)
    for name, df in (("delta_half_a", d_a), ("delta_half_b", d_b), ("deg_half_a", deg_a)):
        df = df.copy()
        df.index = [i.rsplit("|", 1)[0] for i in df.index]
        df.to_parquet(out / f"{name}.parquet")

    # conditions for generation: control first, then perturbations, all within the context
    cond = pd.DataFrame({"id": [args.control, *perts], args.pert_key: [args.control, *perts]})
    for k, v in context.items():
        cond[k] = v
    cond.to_csv(out / "conditions.csv", index=False)

    info = {
        "context": context,
        "pert_key": args.pert_key,
        "control": args.control,
        "n_control_cells": int(n_cells[args.control]),
        "control_median_library_size": ctrl_lib,
        "n_cells": {p: int(n_cells[p]) for p in perts},
        "dropped": dropped,
        "alpha": args.alpha,
        "min_lfc": args.min_lfc,
        "n_degs": deg.sum(1).astype(int).to_dict(),
    }
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(f"median DEGs per perturbation: {int(deg.sum(1).median())}; wrote {out}")


if __name__ == "__main__":
    main()
