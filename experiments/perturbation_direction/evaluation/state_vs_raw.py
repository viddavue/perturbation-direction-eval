"""Do the STATE file's values reproduce from the raw GEO counts, cell by cell?

The STATE file stores log1p-normalised expression, but its values cannot be turned back into counts by
any single per-cell factor (expm1 is not on an integer grid and the implied library is unrelated to
obs.UMI_count). So recovery -- which our ground truth relies on -- has to be validated directly:
take cells present in both files and ask which transformation of the raw counts reproduces X.

Candidates, each with a library size estimated from the same cells:
  log1p(c / lib * 1e4)      paper: CP10k over the normalised matrix, then log1p
  log1p(c / lib * 1e6)      CPM
  log1p(c)                  log1p of raw counts
  c / lib * 1e4             linear CP10k in log space

Pass criteria: max |difference| over nonzero entries and the Pearson correlation on the same cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem


def block(ds, rows: np.ndarray) -> np.ndarray:
    """Dense read of the given rows; rows must be sorted."""
    return np.asarray(ds[rows.min() : rows.max() + 1])[rows - rows.min()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cell-line", default="hepg2")
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--gene-col", default="gene_name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.state, "r") as fs, h5py.File(args.raw, "r") as fr:
        sobs, svar = read_elem(fs["obs"]), read_elem(fs["var"])
        robs, rvar = read_elem(fr["obs"]), read_elem(fr["var"])
        sobs_idx = sobs.index.astype(str).to_numpy()
        robs_idx = robs.index.astype(str).to_numpy()
        line = sobs["cell_line"].astype(str).to_numpy()

        # genes: state index first, then any name duplicated in the raw file (keep first occurrence)
        rnames = rvar[args.gene_col].astype(str).to_numpy()
        first = {}
        for i, n in enumerate(rnames):
            first.setdefault(n, i)
        sgenes = svar.index.astype(str).to_numpy()
        shared = [g for g in sgenes if g in first]
        s_cols = np.array([int(np.flatnonzero(sgenes == g)[0]) for g in shared])
        r_cols = np.array([first[g] for g in shared])

        # cells: hepg2 cells of the state file that also exist in the raw file
        hep = np.flatnonzero(line == args.cell_line)
        pos = {c: i for i, c in enumerate(robs_idx)}
        hep = np.array([i for i in hep if sobs_idx[i] in pos])
        pick = np.sort(np.random.default_rng(0).choice(hep, min(args.cells, hep.size), replace=False))

        S = np.asarray(fs["X"][pick])[:, s_cols].astype(np.float64)
        rows = np.array([pos[sobs_idx[i]] for i in pick])
        C = block(fr["X"], rows)[:, r_cols].astype(np.float64)

    if not np.allclose(C, np.round(C)):
        raise SystemExit("raw matrix is not integer counts; wrong file")
    lib = C.sum(1)
    report = {"n_cells": int(len(pick)), "n_genes": len(shared), "candidates": {}}
    cands = [
        ("log1p_cp10k", np.log1p(C / lib[:, None] * 1e4)),
        ("log1p_cpm", np.log1p(C / lib[:, None] * 1e6)),
        ("log1p_counts", np.log1p(C)),
        ("linear_cp10k", C / lib[:, None] * 1e4),
    ]
    # STATE normalised after selecting genes, so the library may be the sum over only the genes in the file
    cands.append(("log1p_cp10k_in_file", np.log1p(C / np.maximum(C.sum(1, keepdims=True), 1) * 1e4)))
    for name, pred in cands:
        diff = np.abs(pred - S)
        nz = (C > 0) | (S > 0)
        report["candidates"][name] = {
            "max_abs_diff": float(diff[nz].max()),
            "median_abs_diff": float(np.median(diff[nz])),
            "pearson": float(np.corrcoef(pred[nz], S[nz])[0, 1]),
            "frac_within_1e-3": float(np.mean(diff[nz] < 1e-3)),
        }

    # how state and raw differ in support, and size factors per cell
    both = (C > 0) & (S > 0)
    report["support"] = {
        "nonzero_both": int(both.sum()),
        "only_raw": int(((C > 0) & (S == 0)).sum()),
        "only_state": int(((C == 0) & (S > 0)).sum()),
        "total_nonzero": int(((C > 0) | (S > 0)).sum()),
    }
    report["per_cell"] = {
        "median_lib": float(np.median(lib)),
        "median_state_expm1_mass": float(np.median(np.expm1(S).sum(1))),
        "ratio_quantiles": np.quantile(np.expm1(S).sum(1) / lib, [0.05, 0.5, 0.95]).round(3).tolist(),
    }
    out.write_text(json.dumps(report, indent=2))
    for name, r in report["candidates"].items():
        print(
            f"{name:15s} max|diff|={r['max_abs_diff']:.4g}  median={r['median_abs_diff']:.4g}  "
            f"pearson={r['pearson']:.4f}  within1e-3={r['frac_within_1e-3']:.3f}"
        )
    print("support:", report["support"])
    print("per cell:", report["per_cell"])
    pd.Series({k: v["max_abs_diff"] for k, v in report["candidates"].items()}, name="max_abs_diff").to_csv(
        out.with_suffix(".csv")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
