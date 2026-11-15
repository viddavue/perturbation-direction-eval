"""STATE's on-target knockdown filter, reproduced from the raw Replogle screens.

STATE's ``replogle.h5ad`` keeps every control cell of each line and a knockdown-selected subset of
perturbed cells. The selection is cell-load's ``filter_on_target_knockdown`` (residual expression
0.30 per perturbation, 0.50 per cell, at least 30 cells), and it reproduces exactly from raw data
when it is evaluated

* on the raw UMI counts (``TRANSFORM = "counts"``, not CP10k or log1p),
* separately for each raw screen file (``SCOPE = "per_screen"``), with each file's own
  non-targeting cells as the control mean,
* with targets looked up by ``var.gene_name`` over all genes of the raw file (first occurrence
  for duplicated names), and a strict ``<`` at both thresholds.

That variant gives the identical kept cell set (obs index, after undoing STATE's ``-1`` suffix on
barcodes that collide with an earlier line) on all four lines, 0 cells missing and 0 extra:
HepG2 96,616 cells / 1,340 targets, Jurkat 184,470 / 1,537, K562 essential 188,590 / 1,383,
RPE1 173,737 / 1,499. On HepG2, CP10k or log1p values, ``<=`` (6 cells sit exactly on a 0.5
ratio), a target lookup restricted to STATE's 6,642 genes and the perturbation-level-only version
all miss, by 6 to ~32,000 cells; one filter over the four lines merged misses by 8,687-16,193
cells per line.

The filter is split in two so the 5-10 GB raw files never have to be loaded whole:
:func:`read_target_expression` streams the matrix in row blocks and keeps, per cell, only the value
of its own target gene and its library size, plus the control mean of every gene;
:func:`filter_on_target_knockdown` is a port of cell-load's filter as array code on that summary.

Ported from cell-load (https://github.com/ArcInstitute/cell-load), ``src/cell_load/utils/data_utils.py``,
function ``filter_on_target_knockdown``: the semantics of the loop implementation at
``CELL_LOAD_COMMIT`` (d84018d, 2025-09-17), written in the array form of the vectorised HEAD
(5079861, 2026-06-29; repository HEAD 9ba45e59). cell-load is MIT licensed:

    MIT License

    Copyright (c) 2025 Arc Research Institute

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Version differences (all three-stage versions: unmatched targets dropped, controls kept, strict
``<`` at both thresholds, ``min_cells`` counted after the cell-level stage):

* 892692a/9b2e005 (2025-06-04): perturbation level only, unmatched targets kept. Not STATE.
* 4f370b0 (2025-06-04) to e02a702: three stages; a zero control mean raises ``ValueError``.
* d84018d (2025-09-17) to 67baf42 (2026-02-21): a zero control mean drops the target (``np.isclose``).
  This port. Duplicated gene names resolve to the first occurrence (``var_names_make_unique``
  then ``get_loc``).
* 5079861 (2026-06-29, HEAD): vectorised, same selection, but ``pd.Index.get_indexer`` raises
  ``InvalidIndexError`` on duplicated gene names. Every raw screen duplicates HSPA14 (K562 also
  TBCE), so HEAD cannot run on them and did not build STATE.

No target gene of the four raw screens has a zero control mean (smallest 0.078, RPE1), so every
three-stage 2025 version selects the same cells there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

CELL_LOAD_COMMIT = "d84018d113daa6fa64fc0b3b78fb29b53b235954"
TRANSFORM = "counts"
SCOPE = "per_screen"
TRANSFORMS = ("counts", "cp10k", "log1p_cp10k")


@dataclass
class TargetExpression:
    """Per-cell own-target expression of one screen (or several, see :func:`concat_target_expression`).

    Attributes
    ----------
    perts
        ``(n_cells,)`` str, ``obs[pert_key]``.
    own
        ``(n_cells,)`` float64, each cell's value of its own target gene under the transform; NaN for
        control cells and for cells whose target is not matched.
    ctrl_means
        Matched target -> mean value of that gene over the control cells.
    library
        ``(n_cells,)`` float64, raw count sum over all genes of the file.
    matched
        Targets found in ``var[gene_col]``.
    n_ctrl
        Number of control cells behind ``ctrl_means`` (0 when unknown).
    ctrl_sums
        Gene name -> sum over the control cells, for every gene of the file (first occurrence of a
        duplicated name). Only :func:`concat_target_expression` needs it; empty when unknown.
    """

    perts: np.ndarray
    own: np.ndarray
    ctrl_means: dict[str, float]
    library: np.ndarray
    matched: frozenset[str]
    n_ctrl: int = 0
    ctrl_sums: dict[str, float] = field(default_factory=dict)


def _column(frame: pd.DataFrame, key: str, where: str) -> np.ndarray:
    if key not in frame.columns:
        raise KeyError(f"Column {key!r} not found in {where}.")
    return frame[key].astype(str).to_numpy()


def _transform(values: np.ndarray | sparse.csr_matrix, library: np.ndarray, transform: str):
    """Apply ``transform`` to raw values whose rows (cells) have raw libraries ``library``.

    ``values`` is ``(k,)`` (one value per cell) or ``(k, genes)``, dense float64 or CSR.
    """
    if transform == "counts":
        return values
    scale = np.divide(1e4, library, out=np.zeros(len(library)), where=library > 0)
    if sparse.issparse(values):
        out = sparse.csr_matrix(sparse.diags(scale) @ values)
        return out.log1p() if transform == "log1p_cp10k" else out
    out = values * (scale if values.ndim == 1 else scale[:, None])
    return np.log1p(out) if transform == "log1p_cp10k" else out


def _read_block(node, start: int, stop: int, n_genes: int, indptr: np.ndarray | None):
    """Rows ``[start, stop)`` of a dense dataset (as stored) or a CSR group (as ``csr_matrix``)."""
    if indptr is None:
        return node[start:stop]
    a, b = indptr[start], indptr[stop]
    return sparse.csr_matrix(
        (node["data"][a:b], node["indices"][a:b], indptr[start : stop + 1] - a), shape=(stop - start, n_genes)
    )


def read_target_expression(
    h5ad_path,
    *,
    pert_key: str = "gene",
    control: str = "non-targeting",
    gene_col: str = "gene_name",
    transform: str = TRANSFORM,
    block: int = 20000,
) -> TargetExpression:
    """Stream an h5ad's X in row blocks and summarise what the knockdown filter needs.

    Parameters
    ----------
    h5ad_path
        Raw screen file; X is a dense dataset or a CSR group of raw counts over all genes.
    pert_key
        obs column holding the CRISPR target (control cells carry ``control``).
    control
        Label of control cells.
    gene_col
        var column with gene symbols; targets are looked up by this name, first occurrence for
        duplicates (what cell-load d84018d does via ``var_names_make_unique``; HEAD's ``get_indexer``
        raises on duplicated names instead).
    transform
        ``counts`` (raw values), ``cp10k`` (1e4 * c / library) or ``log1p_cp10k``; library is the sum
        over all genes of the file.
    block
        Rows per read.

    Returns
    -------
    TargetExpression
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    with h5py.File(h5ad_path, "r") as f:
        perts = _column(read_elem(f["obs"]), pert_key, "obs")
        names = _column(read_elem(f["var"]), gene_col, "var")
        first: dict[str, int] = {}
        for i, name in enumerate(names):
            first.setdefault(name, i)
        targets = pd.unique(perts[perts != control])
        matched = [t for t in targets if t in first]
        target_col = {t: first[t] for t in matched}
        own_col = np.array([target_col.get(p, -1) for p in perts], dtype=np.int64)
        is_ctrl = perts == control
        gene_cols = np.array(list(first.values()), dtype=np.int64)

        node = f["X"]
        n_cells, n_genes = len(perts), len(names)
        indptr = None
        if isinstance(node, h5py.Group):
            if node.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError(f"X must be dense or csr_matrix, got {node.attrs.get('encoding-type')!r}")
            indptr = node["indptr"][:]
        shape = tuple(node.attrs["shape"]) if indptr is not None else node.shape
        if shape != (n_cells, n_genes):
            raise ValueError(f"X shape {shape} does not match obs x var ({n_cells}, {n_genes})")

        own = np.full(n_cells, np.nan)
        library = np.zeros(n_cells)
        ctrl_sum = np.zeros(n_genes)
        for start in range(0, n_cells, block):
            stop = min(start + block, n_cells)
            blk = _read_block(node, start, stop, n_genes, indptr)
            lib = np.asarray(blk.sum(axis=1, dtype=np.float64)).ravel()
            library[start:stop] = lib
            # only each cell's own-target value and the control rows are transformed (in float64)
            rows = np.flatnonzero(own_col[start:stop] >= 0)
            if len(rows):
                raw = np.asarray(blk[rows, own_col[start:stop][rows]], dtype=np.float64).ravel()
                own[start + rows] = _transform(raw, lib[rows], transform)
            ctrl_rows = np.flatnonzero(is_ctrl[start:stop])
            if len(ctrl_rows):
                sub = blk[ctrl_rows]
                sub = sub.astype(np.float64) if indptr is not None else np.asarray(sub, dtype=np.float64)
                vals = _transform(sub, lib[ctrl_rows], transform)
                ctrl_sum += np.asarray(vals.sum(axis=0), dtype=np.float64).ravel()
            del blk  # free this block before the next read allocates its own

    n_ctrl = int(is_ctrl.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        ctrl_mean = ctrl_sum / n_ctrl if n_ctrl else np.full(n_genes, np.nan)
    return TargetExpression(
        perts=perts,
        own=own,
        ctrl_means={t: float(ctrl_mean[target_col[t]]) for t in matched},
        library=library,
        matched=frozenset(matched),
        n_ctrl=n_ctrl,
        ctrl_sums={name: float(ctrl_sum[col]) for name, col in zip(first, gene_cols, strict=True)},
    )


def concat_target_expression(parts: list[TargetExpression]) -> TargetExpression:
    """Concatenate screens as if the filter ran once on their inner-joined (merged) AnnData.

    A target is matched when it is matched in at least one part and its gene is present in every
    part (``ctrl_sums``, or ``ctrl_means`` for a part built without sums). Control means are
    recomputed over all parts' control cells. Cells whose target is not matched after the merge get
    ``own = NaN``.

    Parameters
    ----------
    parts
        One :class:`TargetExpression` per screen, each with ``n_ctrl > 0``.

    Returns
    -------
    TargetExpression
    """
    if not parts:
        raise ValueError("need at least one part")
    if any(p.n_ctrl <= 0 for p in parts):
        raise ValueError("every part needs n_ctrl > 0 to pool its control mean")

    def part_sum(p: TargetExpression, gene: str) -> float | None:
        if gene in p.ctrl_sums:
            return p.ctrl_sums[gene]
        if gene in p.ctrl_means:
            return p.ctrl_means[gene] * p.n_ctrl
        return None

    n_ctrl = sum(p.n_ctrl for p in parts)
    ctrl_means: dict[str, float] = {}
    for t in sorted(set().union(*(p.matched for p in parts))):
        sums = [part_sum(p, t) for p in parts]
        if all(s is not None for s in sums):
            ctrl_means[t] = float(sum(sums)) / n_ctrl
    matched = frozenset(ctrl_means)
    perts = np.concatenate([p.perts for p in parts])
    own = np.concatenate([p.own for p in parts]).astype(np.float64)
    own[~pd.Index(perts).isin(list(matched))] = np.nan
    genes = set.intersection(*(set(p.ctrl_sums) for p in parts)) if all(p.ctrl_sums for p in parts) else set()
    return TargetExpression(
        perts=perts,
        own=own,
        ctrl_means=ctrl_means,
        library=np.concatenate([p.library for p in parts]).astype(np.float64),
        matched=matched,
        n_ctrl=n_ctrl,
        ctrl_sums={g: float(sum(p.ctrl_sums[g] for p in parts)) for g in sorted(genes)},
    )


def filter_on_target_knockdown(
    te: TargetExpression,
    *,
    control: str = "non-targeting",
    residual_expression: float = 0.30,
    cell_residual_expression: float = 0.50,
    min_cells: int = 30,
) -> tuple[np.ndarray, dict]:
    """cell-load's three-stage knockdown filter on a :class:`TargetExpression`.

    1. Keep perturbations whose mean own-target value / control mean < ``residual_expression``.
    2. Within those, keep cells whose own-target value / control mean < ``cell_residual_expression``.
    3. Drop perturbations with fewer than ``min_cells`` cells left. Control cells are always kept.

    Targets not in ``te.matched`` are dropped, and so are targets whose control mean is (close to) zero.

    Parameters
    ----------
    te
        Output of :func:`read_target_expression` (or :func:`concat_target_expression`).
    control
        Label of control cells in ``te.perts``.
    residual_expression, cell_residual_expression, min_cells
        cell-load's thresholds; STATE used the defaults.

    Returns
    -------
    keep
        ``(n_cells,)`` bool mask over ``te``'s cells.
    report
        Per-stage cell and perturbation counts (``stages``), the unmatched and kept targets, and per
        target ``n_cells``, ``ctrl_mean``, ``pert_mean``, ``kd_ratio`` (inf for a zero control mean),
        ``stage1``, ``n_after_stage2`` and ``n_kept``.
    """
    perts = np.asarray(te.perts).astype(str)
    own = np.asarray(te.own, dtype=np.float64)
    n_cells = len(perts)
    if own.shape != (n_cells,):
        raise ValueError(f"own has shape {own.shape}, expected ({n_cells},)")

    control_mask = perts == control
    unique_perts = [p for p in pd.unique(perts) if p != control]
    matched_perts = [p for p in unique_perts if p in te.matched]
    unmatched = sorted(set(unique_perts) - set(matched_perts))
    n_matched = len(matched_perts)
    params = {
        "residual_expression": residual_expression,
        "cell_residual_expression": cell_residual_expression,
        "min_cells": min_cells,
        "control": control,
        "cell_load_commit": CELL_LOAD_COMMIT,
    }

    ctrl_means = np.array([te.ctrl_means[p] for p in matched_perts], dtype=np.float64)
    pert_to_col = {p: i for i, p in enumerate(matched_perts)}
    cell_col = np.array([pert_to_col.get(p, -1) for p in perts], dtype=np.int64)
    matched_cells = cell_col >= 0
    valid_row = np.flatnonzero(matched_cells)
    valid_col = cell_col[valid_row]

    # stage 1: perturbation-level mean knockdown
    diag_expr = own[valid_row]
    pert_sums = np.bincount(valid_col, weights=diag_expr, minlength=n_matched)
    pert_counts = np.bincount(valid_col, minlength=n_matched).astype(np.float64)
    pert_means = np.where(pert_counts > 0, pert_sums / np.maximum(pert_counts, 1.0), 0.0)
    valid_ctrl = ~np.isclose(ctrl_means, 0.0)
    kd_ratio = np.where(valid_ctrl, pert_means / np.where(valid_ctrl, ctrl_means, 1.0), np.inf)
    stage1_pass = valid_ctrl & (kd_ratio < residual_expression)

    # stage 2: cell-level knockdown within perturbations that passed stage 1
    passed_sel = stage1_pass[valid_col]
    passed_row = valid_row[passed_sel]
    passed_col = valid_col[passed_sel]
    expr_vals = own[passed_row]
    ctrl_per_cell = ctrl_means[passed_col]
    nonzero_ctrl = ~np.isclose(ctrl_per_cell, 0.0)
    cell_keep = np.zeros(len(passed_row), dtype=bool)
    cell_keep[nonzero_ctrl] = expr_vals[nonzero_ctrl] / ctrl_per_cell[nonzero_ctrl] < cell_residual_expression
    keep = control_mask.copy()
    keep[passed_row] = cell_keep
    after_stage2 = np.bincount(cell_col[keep & matched_cells], minlength=n_matched)
    keep_stage2 = keep.copy()

    # stage 3: minimum cells per perturbation
    drop_pert = after_stage2 < min_cells
    kept_row = np.flatnonzero(keep & matched_cells)
    keep[kept_row[drop_pert[cell_col[kept_row]]]] = False
    n_kept = np.bincount(cell_col[keep & matched_cells], minlength=n_matched)

    def stage(mask: np.ndarray, perts_kept: int) -> dict:
        return {
            "cells": int(mask.sum()),
            "perturbed_cells": int((mask & ~control_mask).sum()),
            "perturbations": perts_kept,
        }

    n_ctrl = int(control_mask.sum())
    stage1_mask = control_mask.copy()
    stage1_mask[valid_row] = stage1_pass[valid_col]
    kept_targets = [p for i, p in enumerate(matched_perts) if n_kept[i] > 0]
    report = {
        "params": params,
        "n_cells": n_cells,
        "n_controls": n_ctrl,
        "n_perturbations": len(unique_perts),
        "n_matched": n_matched,
        "unmatched": unmatched,
        "stages": {
            "input": stage(np.ones(n_cells, dtype=bool), len(unique_perts)),
            "matched": stage(control_mask | matched_cells, n_matched),
            "stage1_perturbation": stage(stage1_mask, int(stage1_pass.sum())),
            "stage2_cell": stage(keep_stage2, int((after_stage2 > 0).sum())),
            "stage3_min_cells": stage(keep, len(kept_targets)),
        },
        "kept_targets": kept_targets,
        "per_target": {
            p: {
                "n_cells": int(pert_counts[i]),
                "ctrl_mean": float(ctrl_means[i]),
                "pert_mean": float(pert_means[i]),
                "kd_ratio": float(kd_ratio[i]),
                "stage1": bool(stage1_pass[i]),
                "n_after_stage2": int(after_stage2[i]),
                "n_kept": int(n_kept[i]),
            }
            for i, p in enumerate(matched_perts)
        },
    }
    return keep, report
