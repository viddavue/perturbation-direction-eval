"""Per-screen audit of a raw Replogle-Nadig h5ad, run before anything is merged.

One blocked pass over ``X`` (dense dataset or CSR group) reports, for one screen:

- cells, genes, ``X`` layout and dtype; whether every value is a finite, non-negative integer
- the target label column, its label categories (controls, single targets found in ``var``, single
  targets not in ``var``, anything else) and labels outside the model vocabulary
- control cells, cells per target (min / median / max)
- library size over ALL genes of the file, genes detected per cell, cells with a zero library
- mitochondrial share (genes named ``MT-*``) and the share of the library carried by the model genes
- duplicated gene names, and whether any model gene is among them; every model gene must be present
  exactly once
- whether both guides of each cell's guide pair (``sgID_AB``) name the cell's target (report only)

Gates (exit code 1 when any fails): non-integer, negative or non-finite counts; fewer than
``--min-controls`` control cells; any cell with a zero library; a model gene missing or duplicated;
a duplicated cell id (the obs index is the cell key downstream).

The module also holds the small h5 readers the other raw-pipeline scripts share. They read the
legacy 0.1.0 dataframe layout of the raw GEO/figshare files (categorical codes with a ``categories``
object reference) as well as the current anndata layout.

Usage::

    python audit_raw_screen.py --h5ad GSE264667_hepg2_raw_singlecell_01.h5ad --cell-line hepg2 \
        --metadata-json metadata/replogle_train.json --out audit_hepg2.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from collections.abc import Iterator

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
_SEPARATORS = re.compile(r"[,;+|&\s]")


# ---------------------------------------------------------------------------
# shared h5 readers
# ---------------------------------------------------------------------------
def decode(values) -> np.ndarray:
    """Bytes -> str, as an object array (h5py returns variable-length strings as bytes)."""
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def read_column(group: h5py.Group, name: str) -> np.ndarray:
    """One obs/var column; categoricals are decoded to their labels (missing codes become None).

    Supports the current anndata layout (a group with ``codes``/``categories``) and the legacy
    0.1.0 layout of the raw Replogle files (integer codes whose ``categories`` attribute is an object
    reference to ``__categories/<name>``).
    """
    node = group[name]
    if isinstance(node, h5py.Group):
        if "codes" in node and "categories" in node:
            cats, codes = decode(node["categories"][:]), node["codes"][:]
        elif "values" in node and "mask" in node:  # nullable integer / boolean
            vals = node["values"][:].astype(object)
            vals[node["mask"][:]] = None
            return vals
        else:
            raise ValueError(f"unsupported column encoding for {group.name}/{name}: {dict(node.attrs)}")
    elif "categories" in node.attrs:
        cats, codes = decode(group.file[node.attrs["categories"]][:]), node[:]
    else:
        vals = node[:]
        return decode(vals) if vals.dtype.kind in "SO" else vals
    out = np.full(len(codes), None, dtype=object)
    ok = codes >= 0
    out[ok] = cats[codes[ok]]
    return out


def read_index(group: h5py.Group) -> np.ndarray:
    """The dataframe index (e.g. cell barcodes) as a str array."""
    return read_column(group, group.attrs["_index"]).astype(str)


def x_layout(node) -> dict:
    """Layout, shape and dtype of an X node without reading it."""
    if isinstance(node, h5py.Dataset):
        return {"layout": "dense", "shape": [int(s) for s in node.shape], "dtype": str(node.dtype)}
    enc = node.attrs.get("encoding-type", "")
    enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    if enc != "csr_matrix":
        raise ValueError(f"unsupported X encoding {enc!r} (dense dataset or csr_matrix group expected)")
    return {
        "layout": "csr",
        "shape": [int(s) for s in node.attrs["shape"]],
        "dtype": str(node["data"].dtype),
        "nnz": int(node["data"].shape[0]),
    }


def iter_row_blocks(node, block: int) -> Iterator[tuple[int, int, np.ndarray | sparse.csr_matrix]]:
    """Yield ``(start, stop, rows)`` over X in contiguous row blocks; dense stays dense, CSR stays CSR."""
    info = x_layout(node)
    n, m = info["shape"]
    for start in range(0, n, block):
        stop = min(start + block, n)
        if info["layout"] == "dense":
            yield start, stop, node[start:stop]
        else:
            ip = node["indptr"][start : stop + 1].astype(np.int64)
            yield (
                start,
                stop,
                sparse.csr_matrix(
                    (node["data"][ip[0] : ip[-1]], node["indices"][ip[0] : ip[-1]], ip - ip[0]), shape=(stop - start, m)
                ),
            )


def first_positions(names: np.ndarray) -> dict[str, int]:
    """Name -> position of its first occurrence (what ``pd.Index.get_indexer`` resolves to)."""
    first: dict[str, int] = {}
    for i, g in enumerate(names):
        first.setdefault(str(g), i)
    return first


def model_gene_positions(var_names: np.ndarray, genes: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Column of each model gene in the file, plus the model genes that are missing / duplicated there."""
    counts = pd.Series(var_names).value_counts()
    first = first_positions(var_names)
    missing = [g for g in genes if g not in first]
    duplicated = [g for g in genes if counts.get(g, 0) > 1]
    pos = np.array([first.get(g, -1) for g in genes], dtype=np.int64)
    return pos, missing, duplicated


def take_columns(rows, cols: np.ndarray) -> np.ndarray:
    """Dense float64 copy of the given columns of a dense or CSR block."""
    if sparse.issparse(rows):
        return np.asarray(rows[:, cols].todense(), dtype=np.float64)
    return np.asarray(rows[:, cols], dtype=np.float64)


def row_sums(rows) -> np.ndarray:
    """Exact float64 row sums of a dense or CSR block."""
    if sparse.issparse(rows):
        return np.asarray(rows.sum(1, dtype=np.float64)).ravel()
    return rows.sum(1, dtype=np.float64)


def quantiles(x: np.ndarray) -> dict[str, float] | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {str(q): float(np.quantile(x, q)) for q in QUANTILES} if x.size else None


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays and sets for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj) if isinstance(obj, (set, frozenset)) else obj
        return [to_jsonable(v) for v in items]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def classify_labels(labels: pd.Series, control: str, var_names: set[str]) -> dict[str, list[str]]:
    """Sort label categories into control / single target in var / single target not in var / other."""
    out: dict[str, list[str]] = {"control": [], "single_in_var": [], "single_not_in_var": [], "other": []}
    for lab in labels.dropna().unique():
        lab = str(lab)
        if lab == control:
            out["control"].append(lab)
        elif not lab or _SEPARATORS.search(lab):
            out["other"].append(lab)
        elif lab in var_names:
            out["single_in_var"].append(lab)
        else:
            out["single_not_in_var"].append(lab)
    return {k: sorted(v) for k, v in out.items()}


def guide_agreement(obs: pd.DataFrame, pert_key: str, guide_key: str, control: str) -> dict | None:
    """How many cells have a guide pair whose guides both name the cell's target (report only)."""
    if guide_key not in obs:
        return None
    guides = obs[guide_key].astype(str)
    target = obs[pert_key].astype(str)
    pairs = pd.DataFrame({"g": guides, "t": target}).drop_duplicates()

    def agrees(g: str, t: str) -> bool:
        names = {p.split("_")[0] for p in g.split("|")}
        return names == {t if t != control else "non-targeting"}

    ok = {(g, t) for g, t in zip(pairs.g, pairs.t, strict=True) if agrees(g, t)}
    cell_ok = np.array([(g, t) in ok for g, t in zip(guides, target, strict=True)])
    bad = pairs[[(g, t) not in ok for g, t in zip(pairs.g, pairs.t, strict=True)]]
    return {
        "guide_key": guide_key,
        "cells_checked": int(len(obs)),
        "cells_guides_name_target": int(cell_ok.sum()),
        "cells_disagreeing": int((~cell_ok).sum()),
        "examples_disagreeing": [f"{g} -> {t}" for g, t in zip(bad.g[:10], bad.t[:10], strict=True)],
    }


def audit(
    h5ad: str | pathlib.Path,
    *,
    cell_line: str,
    genes: list[str],
    vocab: list[str],
    pert_key: str = "gene",
    control: str = "non-targeting",
    gene_col: str = "gene_name",
    guide_key: str = "sgID_AB",
    min_controls: int = 300,
    block: int = 10_000,
) -> dict:
    """Audit one raw screen; returns the report dict (``report['pass']`` is the overall verdict)."""
    t0 = time.time()
    h5ad = pathlib.Path(h5ad)
    with h5py.File(h5ad, "r") as f:
        layout = x_layout(f["X"])
        n_cells, n_genes = layout["shape"]
        var_names = (
            read_column(f["var"], gene_col).astype(str) if gene_col in f["var"] else read_index(f["var"])
        ).astype(str)
        cell_ids = read_index(f["obs"])
        obs_cols = [c for c in (pert_key, guide_key, "gem_group", "UMI_count") if c in f["obs"]]
        obs = pd.DataFrame({c: read_column(f["obs"], c) for c in obs_cols}, index=cell_ids)
        if pert_key not in obs:
            raise KeyError(f"obs has no target column {pert_key!r}")

        pos, missing, dup_model = model_gene_positions(var_names, genes)
        mt_cols = np.flatnonzero(np.char.startswith(var_names.astype(str), "MT-"))

        library = np.zeros(n_cells)
        model_lib = np.zeros(n_cells)
        detected = np.zeros(n_cells, dtype=np.int64)
        mt = np.zeros(n_cells)
        n_nonint = n_neg = n_nonfinite = 0
        vmax = 0.0
        for start, stop, rows in iter_row_blocks(f["X"], block):
            v = rows.data if sparse.issparse(rows) else rows
            bad = int(v.size - np.isfinite(v).sum())
            if bad:
                n_nonfinite += bad
                v = np.where(np.isfinite(v), v, 0)
            n_neg += int((v < 0).sum())
            n_nonint += int((v != np.rint(v)).sum())
            vmax = max(vmax, float(v.max()) if v.size else 0.0)
            library[start:stop] = row_sums(rows)
            model_lib[start:stop] = take_columns(rows, pos[pos >= 0]).sum(1)
            if sparse.issparse(rows):
                nz = rows.copy()
                nz.eliminate_zeros()
                detected[start:stop] = np.diff(nz.indptr)
            else:
                detected[start:stop] = (rows > 0).sum(1)
            if len(mt_cols):
                mt[start:stop] = take_columns(rows, mt_cols).sum(1)

    labels = obs[pert_key]
    is_ctrl = (labels.astype(str) == control).to_numpy()
    cats = classify_labels(labels, control, set(var_names))
    per_target = labels[~is_ctrl].astype(str).value_counts()
    vocab_set = set(vocab)
    outside = per_target[~per_target.index.isin(vocab_set)]
    dup_names = pd.Series(var_names).value_counts()
    dup_names = dup_names[dup_names > 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mt_share = np.where(library > 0, mt / library, np.nan)
        model_share = np.where(library > 0, model_lib / library, np.nan)
        umi_ratio = (
            np.where(
                obs["UMI_count"].to_numpy(dtype=np.float64) > 0, library / obs["UMI_count"].to_numpy(np.float64), np.nan
            )
            if "UMI_count" in obs
            else None
        )

    checks = {
        "integer_counts": {
            "pass": n_nonint == 0 and n_neg == 0 and n_nonfinite == 0,
            "non_integer_values": n_nonint,
            "negative_values": n_neg,
            "non_finite_values": n_nonfinite,
            "max_value": vmax,
        },
        "min_controls": {
            "pass": int(is_ctrl.sum()) >= min_controls,
            "controls": int(is_ctrl.sum()),
            "min": min_controls,
        },
        "no_zero_library": {"pass": int((library <= 0).sum()) == 0, "zero_library_cells": int((library <= 0).sum())},
        "model_genes_once": {
            "pass": not missing and not dup_model,
            "missing": missing,
            "duplicated": dup_model,
            "n_model_genes": len(genes),
        },
        "unique_cell_ids": {
            "pass": bool(pd.Index(cell_ids).is_unique),
            "duplicated_ids": int(pd.Index(cell_ids).duplicated().sum()),
        },
    }
    report = {
        "file": str(h5ad),
        "size_bytes": h5ad.stat().st_size,
        "cell_line": cell_line,
        "cells": n_cells,
        "genes": n_genes,
        "X": layout,
        "target_column": pert_key,
        "target_extraction": f"obs[{pert_key!r}] used as-is (one target symbol per cell, {control!r} = control)",
        "obs_columns": sorted(obs_cols),
        "controls": int(is_ctrl.sum()),
        "label_categories": {k: len(v) for k, v in cats.items()},
        "labels_not_single_target": cats["other"],
        "labels_single_not_in_var": cats["single_not_in_var"],
        "cells_with_missing_label": int(labels.isna().sum()),
        "n_targets": int(len(per_target)),
        "cells_per_target": {
            "min": int(per_target.min()) if len(per_target) else None,
            "median": float(per_target.median()) if len(per_target) else None,
            "max": int(per_target.max()) if len(per_target) else None,
            "targets_lt_30": int((per_target < 30).sum()),
        },
        "outside_vocab": {
            "labels": int(len(outside)),
            "cells": int(outside.sum()),
            "examples": sorted(outside.index)[:20],
            "control_in_vocab": control in vocab_set,
        },
        "library_full": quantiles(library),
        "library_full_over_UMI_count": quantiles(umi_ratio) if umi_ratio is not None else None,
        "genes_detected": quantiles(detected),
        "zero_umi_cells": int((library <= 0).sum()),
        "mito": {
            "n_mt_genes": int(len(mt_cols)),
            "mt_genes": var_names[mt_cols].tolist(),
            "share": quantiles(mt_share),
        },
        "duplicate_gene_names": {str(k): int(v) for k, v in dup_names.items()},
        "model_genes_affected_by_duplicates": sorted(set(dup_names.index) & set(genes)),
        "model_gene_library_share": quantiles(model_share),
        "guides": guide_agreement(obs, pert_key, guide_key, control),
        "gem_groups": int(obs["gem_group"].nunique()) if "gem_group" in obs else None,
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
        "seconds": round(time.time() - t0, 1),
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--cell-line", required=True)
    ap.add_argument("--metadata-json", required=True, help="TRAIN metadata json (2,000 genes, label vocabulary)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pert-key", default="gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--gene-col", default="gene_name")
    ap.add_argument("--min-controls", type=int, default=300)
    ap.add_argument("--block", type=int, default=10_000)
    args = ap.parse_args(argv)

    meta = json.loads(pathlib.Path(args.metadata_json).read_text())
    report = audit(
        args.h5ad,
        cell_line=args.cell_line,
        genes=list(meta["genes"]),
        vocab=list(meta["labels"][args.pert_key]),
        pert_key=args.pert_key,
        control=args.control,
        gene_col=args.gene_col,
        min_controls=args.min_controls,
        block=args.block,
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(report), indent=1))
    failed = [k for k, c in report["checks"].items() if not c["pass"]]
    print(
        f"{args.cell_line}: {report['cells']} cells x {report['genes']} genes ({report['X']['layout']} "
        f"{report['X']['dtype']}), {report['controls']} controls, {report['n_targets']} targets, "
        f"{report['outside_vocab']['labels']} labels outside vocab; "
        + ("PASS" if report["pass"] else f"FAIL: {', '.join(failed)}")
        + f" -> {out}"
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
