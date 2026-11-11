"""Sensitivity check: does the STATE cell filter change the ground truth?

STATE keeps 96,616 of the 145,473 HepG2 cells in the raw GEO file (GSE264667); the filter looks like
a per-cell knockdown filter. Our counts equal the raw counts for the kept cells, so the only open
question is what the dropped cells would have changed.

This compares, on the same perturbations:

* the reference we use today (STATE-kept cells, the ``truth``/``truth_test312`` dirs), and
* the reference rebuilt from *all* raw HepG2 cells with the unchanged pipeline.

Then it rescores existing predictions (scLDM, kNN, generic) against both references, using the same
`evaluate.py`. Nothing is regenerated: the predictions are model output and do not depend on the
reference.

The raw file is in GEO orientation (cells x genes, ``var.gene_name`` names, `gene` in obs holds the
CRISPR target). ``--materialise`` rewrites it once into the layout `ground_truth.py` expects (sparse
integer counts, ``obs.cell_line``, metadata gene order) so the reference is built by the unchanged
script rather than a reimplementation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse


def materialise(raw: Path, genes: list[str], out: Path, cell_line: str, block: int = 20_000) -> dict:
    """Write the raw GEO file as an AnnData with integer counts in the metadata gene order."""
    with h5py.File(raw, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        matrix = f["X"]
        shape = matrix.attrs["shape"] if isinstance(matrix, h5py.Group) else matrix.shape
        names = var["gene_name"].astype(str).to_numpy()
        if set(genes) - set(names):
            raise ValueError(f"{len(set(genes) - set(names))} metadata genes absent from the raw file")
        col = {g: i for i, g in enumerate(names)}
        cols = np.array([col[g] for g in genes])
        # this file stores X dense float32; anndata's python backend would turn it sparse and the
        # read below would become O(cells) per row, so read blocks straight from the dataset
        if isinstance(matrix, h5py.Group):
            raise ValueError("sparse X group layout not handled here; add a CSR block reader")
        out_obs = pd.DataFrame(
            {"gene": obs["gene"].astype(str).to_numpy(), "cell_line": cell_line}, index=obs.index.astype(str)
        )
        chunks = []
        for start in range(0, shape[0], block):
            stop = min(start + block, shape[0])
            chunk = sparse.csr_matrix(np.asarray(matrix[start:stop])[:, cols], dtype=np.float32)
            chunks.append(chunk)
        X = sparse.vstack(chunks, format="csr")
        X.sort_indices()
    if not np.allclose(X.data, np.round(X.data)):
        raise ValueError("raw matrix holds non-integer values; not counts")
    adata = ad.AnnData(X=X, obs=out_obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    counts = out_obs["gene"].value_counts()
    control = "non-targeting"
    return {
        "raw_cells": int(shape[0]),
        "kept_genes": len(genes),
        "targets": int((counts.index != control).sum()),
        "control_cells": int(counts.get(control, 0)),
        "genes_sorted": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw GEO HepG2 h5ad")
    ap.add_argument("--metadata-json", required=True, help="replogle_train.json (gene order)")
    ap.add_argument("--cell-line", default="hepg2")
    ap.add_argument("--perturbations", required=True, help="conditions.csv used for the reported panel")
    ap.add_argument("--min-cells", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "hepg2_raw_all.h5ad"
    if not raw.exists():
        genes = json.loads(Path(args.metadata_json).read_text())["genes"]
        print(f"materialising {args.raw} -> {raw}")
        print(json.dumps(materialise(Path(args.raw), genes, raw, args.cell_line), indent=1))

    here = Path(__file__).resolve().parent
    truth = out / "truth_raw_all"
    if not (truth / "delta_true.parquet").exists():
        cmd = [
            sys.executable, str(here / "ground_truth.py"),
            "--pert-h5ad", str(raw), "--ctrl-h5ad", str(raw),
            "--pert-key", "gene", "--control", "non-targeting",
            "--context", f"cell_line={args.cell_line}",
            "--perturbations", *pd.read_csv(args.perturbations)["gene"].astype(str).tolist(),
            "--metadata-json", args.metadata_json,
            "--min-cells", str(args.min_cells), "--out-dir", str(truth),
        ]
        print(" ".join(cmd[:6]) + " ...")
        subprocess.run(cmd, check=True)
    print(f"ground truth on all raw cells: {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
