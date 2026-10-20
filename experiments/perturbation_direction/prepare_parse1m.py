"""Build a CD4-Naive Parse1M counts file from the public figshare release.

The release (Klein 2025, figshare 28589774) is one 13 GB h5ad of 9.7 M cells x 2,000 genes.
It is NOT the `CD4_Naive_holdout` train/test pair the configs reference, and its 2,000 genes
are its own HVGs: only 1,313 of the model's 2,000-gene vocabulary are present. Ground truth can
therefore only be built on those 1,313, while generation still uses the model's full vocabulary.

Outputs in --out-dir:
  parse1m_cd4naive.h5ad        raw counts, cells of one cell type, selected conditions
  parse1m_genes_available.json derived metadata json with only the genes present in the data
                               (for ground_truth.py; NEVER pass this to generate.py, which
                               needs the model's full vocabulary ordering)
  prepare_info.json            source identity, counts per condition, missing genes

Example:
    python experiments/perturbation_direction/prepare_parse1m.py \
        --src /path/adata_for_cellflow_datasets_with_embeddings.h5ad \
        --metadata-json metadata/parse1m_train.json --test-json metadata/parse1m_test.json \
        --cell-type "CD4 Naive" --control PBS --max-cells 2000 --out-dir outputs/.../parse1m
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def md5_stream(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_categories(node) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in node["categories"][:]]


def select_rows(cell_type: str, cytokines: set[str], max_cells: int, seed: int, src: Path) -> tuple[np.ndarray, dict]:
    """Pick up to ``max_cells`` rows per cytokine, deterministically, from one cell type."""
    with h5py.File(src, "r") as f:
        ct_cats = read_categories(f["obs/cell_type"])
        cy_cats = read_categories(f["obs/cytokine"])
        ct = f["obs/cell_type/codes"][:]
        cy = f["obs/cytokine/codes"][:]
    if cell_type not in ct_cats:
        raise ValueError(f"cell type {cell_type!r} not in source; have {ct_cats[:5]}...")
    unknown = sorted(cytokines - set(cy_cats))
    if unknown:
        raise ValueError(f"cytokines absent from source: {unknown}")
    in_type = ct == ct_cats.index(cell_type)
    rng = np.random.default_rng(seed)
    chosen, per_condition = [], {}
    for cyto in sorted(cytokines):
        rows = np.flatnonzero(in_type & (cy == cy_cats.index(cyto)))
        if len(rows) == 0:
            raise ValueError(f"no cells for {cyto!r} in {cell_type!r}")
        take = rows if len(rows) <= max_cells else np.sort(rng.choice(rows, max_cells, replace=False))
        per_condition[cyto] = {"available": int(len(rows)), "selected": int(len(take))}
        chosen.append(take)
    return np.sort(np.concatenate(chosen)), per_condition


def read_selected_rows(f: h5py.File, mask: np.ndarray, n_cols: int, chunk: int = 100_000) -> sparse.csr_matrix:
    """Stream the source once, keeping the selected rows.

    Iterating over FILE ranges (not over selected rows) bounds memory by ``chunk`` and never
    materialises a block spanning the whole file, which is what happens when the selected cells
    are scattered through 9.7M rows.
    """
    indptr = f["X/indptr"][:]
    data, indices = f["X/data"], f["X/indices"]
    if len(mask) != len(indptr) - 1:
        raise ValueError("mask length does not match the source row count")
    parts = []
    for lo in range(0, len(mask), chunk):
        hi = min(lo + chunk, len(mask))
        local = np.flatnonzero(mask[lo:hi])
        if not len(local):
            continue
        block = sparse.csr_matrix(
            (data[indptr[lo] : indptr[hi]], indices[indptr[lo] : indptr[hi]], indptr[lo : hi + 1] - indptr[lo]),
            shape=(hi - lo, n_cols),
        )
        parts.append(block[local])
    if not parts:
        raise ValueError("no rows selected")
    out = sparse.vstack(parts, format="csr")
    if out.shape[0] != int(mask.sum()):
        raise ValueError("row selection lost or duplicated observations")
    return out


def check_integer_counts(matrix: sparse.csr_matrix, sample: int = 1_000_000) -> None:
    values = matrix.data[:sample]
    if values.size and (not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all()):
        raise ValueError("source X is not raw nonnegative integer counts")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--metadata-json", required=True, help="model vocabulary; supplies the gene order")
    ap.add_argument("--test-json", required=True, help="supplies the held-out cytokine labels")
    ap.add_argument("--cell-type", default="CD4 Naive")
    ap.add_argument("--control", default="PBS")
    ap.add_argument("--max-cells", type=int, default=2000, help="cells kept per condition (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-md5", action="store_true", help="stream the whole source to record its md5")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_genes = json.loads(Path(args.metadata_json).read_text())["genes"]
    test_labels = json.loads(Path(args.test_json).read_text())["labels"]
    cytokines = {str(c) for c in test_labels["cytokine"]} | {args.control}

    with h5py.File(src, "r") as f:
        n_cols = int(f["X"].attrs["shape"][1])
        var = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["var/_index"][:]])
        if n_cols != len(var):
            raise ValueError("X width and var length disagree")
        rows, per_condition = select_rows(args.cell_type, cytokines, args.max_cells or 10**9, args.seed, src)
        mask = np.zeros(int(f["X"].attrs["shape"][0]), dtype=bool)
        mask[rows] = True
        matrix = read_selected_rows(f, mask, n_cols)
        cyto_cats = read_categories(f["obs/cytokine"])
        labels = [cyto_cats[int(code)] for code in f["obs/cytokine/codes"][:][rows]]
        obs = pd.DataFrame(
            {"cell_type": pd.Categorical([args.cell_type] * len(rows)), "cytokine": pd.Categorical(labels)},
            index=[f"cell{i}" for i in range(len(rows))],
        )
    check_integer_counts(matrix)

    available = [g for g in model_genes if g in set(var)]
    missing = [g for g in model_genes if g not in set(var)]
    if not available:
        raise ValueError("none of the model genes are present in the source")
    col = [int(np.flatnonzero(var == g)[0]) for g in available]
    matrix = matrix[:, col]

    adata = ad.AnnData(sparse.csr_matrix(matrix), obs=obs, var=pd.DataFrame(index=pd.Index(available, name=None)))
    adata.write_h5ad(out / "parse1m_cd4naive.h5ad", compression="gzip")

    derived = {"genes": available, "labels": {}, "dataset": "parse1m",
               "source_h5ad": str(out / "parse1m_cd4naive.h5ad"),
               "derived_from": args.metadata_json,
               "note": "only genes present in the public source; for ground_truth.py, not generation"}
    for key, values in json.loads(Path(args.metadata_json).read_text())["labels"].items():
        derived["labels"][key] = [str(v) for v in values]
    derived["labels"]["cytokine"] = sorted(cytokines)
    (out / "parse1m_genes_available.json").write_text(json.dumps(derived, indent=1))

    info = {
        "source": str(src),
        "source_md5": md5_stream(src) if args.verify_md5 else None,
        "source_size_bytes": src.stat().st_size,
        "cell_type": args.cell_type, "control": args.control,
        "max_cells_per_condition": args.max_cells or None, "seed": args.seed,
        "n_cells": int(adata.n_obs), "n_genes_written": len(available),
        "n_model_genes": len(model_genes), "n_model_genes_missing": len(missing),
        "missing_model_genes": missing,
        "per_condition": per_condition,
        "genes_order": "model vocabulary order, restricted to genes present in the source",
        "warning": "ground truth is limited to the available genes; generation still uses the full vocabulary",
        "expected_source_md5": "d89559875a6c75ab37e0adf84c6e177d",
    }
    (out / "prepare_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("n_cells", "n_genes_written", "n_model_genes_missing")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
