"""Split a cell-level dataset into two disjoint halves for a real-vs-real reference.

DE metrics need an anchor: a number like "fidelity 0.69" only means something next to what a
replicate of the same experiment scores. This writes two files whose groups — including the
control — are disjoint, so the DE test in each half uses its own control cells and no cell is
used on both sides.

Halves are capped per group so nothing is inflated by an unequal draw; groups too small to split
are dropped, and the counts are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def split_halves(adata: ad.AnnData, key: str, seed: int, min_cells: int = 4):
    """Return (half_a, half_b, info) with disjoint cells per group."""
    rng = np.random.default_rng(seed)
    labels = adata.obs[key].astype(str).to_numpy()
    keep_a, keep_b, info = [], [], {}
    for label in sorted(set(labels)):
        rows = np.flatnonzero(labels == label)
        if len(rows) < 2 * min_cells:
            info[label] = {"available": int(len(rows)), "kept": 0, "dropped": True}
            continue
        perm = rng.permutation(rows)
        n = len(rows) // 2
        keep_a.extend(perm[:n])
        keep_b.extend(perm[n:])
        info[label] = {"available": int(len(rows)), "kept": int(n), "dropped": False}
    if not keep_a or not keep_b:
        raise ValueError("no group was large enough to split")
    return adata[keep_a].copy(), adata[keep_b].copy(), info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--pert-key", default="gene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-cells", type=int, default=4, help="minimum cells per half")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    adata = ad.read_h5ad(args.real)
    a, b, info = split_halves(adata, args.pert_key, args.seed, args.min_cells)
    if set(a.obs_names) & set(b.obs_names):
        raise AssertionError("halves must be disjoint")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(out / "real_half_a.h5ad", compression="gzip")
    b.write_h5ad(out / "real_half_b.h5ad", compression="gzip")
    dropped = sorted(k for k, v in info.items() if v["dropped"])
    manifest = {"source": args.real, "seed": args.seed, "min_cells_per_half": args.min_cells,
                "n_cells_a": int(a.n_obs), "n_cells_b": int(b.n_obs),
                "n_groups": len(info), "dropped_groups": dropped,
                "note": "disjoint halves including controls; a real-vs-real anchor, not model performance",
                "groups": info}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"half A {a.n_obs} cells, half B {b.n_obs} cells, {len(dropped)} groups dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
