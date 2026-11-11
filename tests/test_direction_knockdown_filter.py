"""knockdown_filter.py must select exactly the cells cell-load's filter_on_target_knockdown selects.

STATE's Replogle file is the output of that filter on each raw screen (raw counts, strict ``<``,
0.30 / 0.50 / 30 cells). The hand-built cases pin each stage; the cross-checks run cell-load's own
functions (copied below, MIT licence) on random AnnData and compare the kept cells:

* ``cellload_loop_filter``: the 2025 loop implementation (d84018d; same semantics from 4f370b0 on,
  except that a zero control mean raised before d84018d). It reproduces STATE.
* ``cellload_head_filter``: the vectorised HEAD (5079861 / 9ba45e59). Same selection, but its
  ``pd.Index.get_indexer`` raises on duplicated gene names, which every raw Replogle screen has.
"""

import logging
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

import knockdown_filter as kf
from knockdown_filter import TargetExpression, concat_target_expression, filter_on_target_knockdown

NT = "non-targeting"
log = logging.getLogger(__name__)


# --- references: cell-load filter_on_target_knockdown --------------------------------------------
# From https://github.com/ArcInstitute/cell-load, src/cell_load/utils/data_utils.py.
# MIT License, Copyright (c) 2025 Arc Research Institute. Permission is hereby granted, free of
# charge, to any person obtaining a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions: The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED.
#
# Loop version at d84018d113daa6fa64fc0b3b78fb29b53b235954 (2025-09-17): _mean, is_on_target_knockdown,
# set_var_index_to_col and filter_on_target_knockdown, copied with docstrings shortened and the bare
# ``try/except`` index fallbacks replaced by the ``.values`` form they fall back to.
def _mean(expr) -> float:
    """Return the mean of a dense or sparse 1-D/2-D slice."""
    if sp.issparse(expr):
        return float(expr.mean())
    return float(np.asarray(expr).mean())


def is_on_target_knockdown(
    adata, target_gene, perturbation_column="gene", control_label="non-targeting", residual_expression=0.30, layer=None
):
    if target_gene == control_label:
        return False
    if target_gene not in adata.var_names:
        print(f"Gene {target_gene!r} not found in `adata.var_names`.")
        return 1
    gene_idx = adata.var_names.get_loc(target_gene)
    X = adata.layers[layer] if layer is not None else adata.X
    control_cells = (adata.obs[perturbation_column] == control_label).values
    perturbed_cells = (adata.obs[perturbation_column] == target_gene).values
    if not perturbed_cells.any():
        raise ValueError(f"No cells labelled with perturbation {target_gene!r}.")
    control_mean = _mean(X[control_cells, gene_idx])
    if np.isclose(control_mean, 0.0):
        log.warning("Skipping perturbation %r: control mean expression is zero.", target_gene)
        return False
    perturbed_mean = _mean(X[perturbed_cells, gene_idx])
    knockdown_ratio = perturbed_mean / control_mean
    return knockdown_ratio < residual_expression


def set_var_index_to_col(adata, col="col", copy=True):
    if col not in adata.var.columns:
        raise KeyError(f"Column {col!r} not found in adata.var.")
    adata.var.index = adata.var[col].astype("str")
    adata.var_names_make_unique()
    return adata


def cellload_loop_filter(
    adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,
    cell_residual_expression=0.50,
    min_cells=30,
    layer=None,
    var_gene_name="gene_name",
):
    adata_ = set_var_index_to_col(adata.copy(), col=var_gene_name)
    X = adata_.layers[layer] if layer is not None else adata_.X
    perts = adata_.obs[perturbation_column]
    control_cells = (perts == control_label).values
    perts_to_keep = [control_label]
    for pert in perts.unique():
        if pert == control_label:
            continue
        if is_on_target_knockdown(
            adata_,
            target_gene=pert,
            perturbation_column=perturbation_column,
            control_label=control_label,
            residual_expression=residual_expression,
            layer=layer,
        ):
            perts_to_keep.append(pert)
    keep_mask = np.zeros(adata_.n_obs, dtype=bool)
    keep_mask[control_cells] = True
    control_mean_cache = {}
    for pert in perts_to_keep:
        if pert == control_label:
            continue
        if pert not in adata_.var_names:
            continue
        gene_idx = adata_.var_names.get_loc(pert)
        if pert not in control_mean_cache:
            ctrl_mean = _mean(X[control_cells, gene_idx])
            control_mean_cache[pert] = ctrl_mean
        else:
            ctrl_mean = control_mean_cache[pert]
        pert_cells = (perts == pert).values
        if np.isclose(ctrl_mean, 0.0):
            keep_mask[pert_cells] = False
            continue
        expr_vals = X[pert_cells, gene_idx].toarray().flatten() if sp.issparse(X) else X[pert_cells, gene_idx]
        ratios = expr_vals / ctrl_mean
        keep_mask[pert_cells] = ratios < cell_residual_expression
    for pert in perts.unique():
        if pert == control_label:
            continue
        pert_mask = (perts == pert).values & keep_mask
        if pert_mask.sum() < min_cells:
            keep_mask[pert_mask] = False
    return adata_[keep_mask]


# Vectorised HEAD at 9ba45e59f6f8117bb7a21371ad38d67175586d53 (function last changed in 5079861,
# 2026-06-29), copied verbatim except that the ``verbose`` printing is removed.
def cellload_head_filter(
    adata,
    perturbation_column="gene",
    control_label="non-targeting",
    residual_expression=0.30,
    cell_residual_expression=0.50,
    min_cells=30,
    layer=None,
    var_gene_name="gene_name",
):
    if var_gene_name not in adata.var.columns:
        raise KeyError(f"Column {var_gene_name!r} not found in adata.var.")
    gene_index = pd.Index(adata.var[var_gene_name])
    X = adata.layers[layer] if layer is not None else adata.X
    perts = adata.obs[perturbation_column]
    n_cells = adata.n_obs
    control_mask = (perts == control_label).values
    unique_perts = [p for p in perts.unique() if p != control_label]
    matched_perts = [p for p in unique_perts if p in gene_index]
    n_matched = len(matched_perts)
    if n_matched == 0:
        return adata[control_mask].copy()
    pert_positions = gene_index.get_indexer(matched_perts)
    if sp.issparse(X):
        X_sub = X[:, pert_positions].toarray()
    else:
        X_sub = np.asarray(X)[:, pert_positions]
    ctrl_means = X_sub[control_mask].mean(axis=0)
    pert_to_col = {p: i for i, p in enumerate(matched_perts)}
    cell_col_mapped = perts.map(pert_to_col)
    cell_col = np.full(n_cells, -1, dtype=np.int32)
    valid = ~cell_col_mapped.isna()
    cell_col[valid.values] = cell_col_mapped[valid].values.astype(np.int32)
    matched_cells = cell_col >= 0
    valid_row = np.where(matched_cells)[0]
    valid_col = cell_col[valid_row]
    diag_expr = X_sub[valid_row, valid_col]
    pert_sums = np.bincount(valid_col, weights=diag_expr, minlength=n_matched)
    pert_counts = np.bincount(valid_col, minlength=n_matched).astype(np.float64)
    pert_means = np.where(pert_counts > 0, pert_sums / np.maximum(pert_counts, 1.0), 0.0)
    valid_ctrl = ~np.isclose(ctrl_means, 0.0)
    kd_ratio = np.where(valid_ctrl, pert_means / np.where(valid_ctrl, ctrl_means, 1.0), np.inf)
    stage1_pass = valid_ctrl & (kd_ratio < residual_expression)
    passed_sel = stage1_pass[valid_col]
    passed_row = valid_row[passed_sel]
    passed_col = valid_col[passed_sel]
    expr_vals = X_sub[passed_row, passed_col]
    ctrl_means_per_cell = ctrl_means[passed_col]
    nonzero_ctrl = ~np.isclose(ctrl_means_per_cell, 0.0)
    cell_keep = np.zeros(len(passed_row), dtype=bool)
    cell_keep[nonzero_ctrl] = expr_vals[nonzero_ctrl] / ctrl_means_per_cell[nonzero_ctrl] < cell_residual_expression
    keep_mask = control_mask.copy()
    keep_mask[passed_row] = cell_keep
    kept_mask = keep_mask & matched_cells
    kept_row = np.where(kept_mask)[0]
    kept_col = cell_col[kept_row]
    if len(kept_col) > 0:
        pert_kept_counts = np.bincount(kept_col, minlength=n_matched)
        drop_pert = pert_kept_counts < min_cells
        cell_drop = drop_pert[kept_col]
        keep_mask[kept_row[cell_drop]] = False
    result = adata[keep_mask].copy()
    if var_gene_name in result.var.columns:
        result.var.index = result.var[var_gene_name].astype(str)
        result.var_names_make_unique()
    return result


# --- helpers -------------------------------------------------------------------------------------
def te_from(cells: dict[str, list[float]], ctrl_means: dict[str, float], n_ctrl: int = 5) -> TargetExpression:
    """TargetExpression from {target: own values}; ``n_ctrl`` control cells first."""
    perts = [NT] * n_ctrl + [p for p, vals in cells.items() for _ in vals]
    own = [np.nan] * n_ctrl + [v for vals in cells.values() for v in vals]
    matched = frozenset(ctrl_means)
    own = [np.nan if p not in matched else v for p, v in zip(perts, own, strict=True)]
    return TargetExpression(
        perts=np.array(perts),
        own=np.array(own, dtype=float),
        ctrl_means=dict(ctrl_means),
        library=np.full(len(perts), 1000.0),
        matched=matched,
        n_ctrl=n_ctrl,
    )


def kept_targets(te, keep):
    return set(te.perts[keep]) - {NT}


def random_screen(seed: int, n_genes: int = 40, n_targets: int = 25, dup: bool = True, prefix: str = "c"):
    """Raw-count AnnData shaped like a Replogle screen: targets, controls, mixed knockdown strength."""
    rng = np.random.default_rng(seed)
    names = [f"G{i}" for i in range(n_genes)]
    if dup:
        names[n_genes - 1] = "G3"  # duplicated symbol: cell-load uses the first column
    targets = [f"G{i}" for i in range(n_targets)] + ["NOT_IN_VAR1", "NOT_IN_VAR2"]
    n_per = rng.integers(20, 60, size=len(targets))
    perts = [NT] * 300 + [t for t, n in zip(targets, n_per, strict=True) for _ in range(n)]
    base = rng.gamma(2.0, 1.5, size=n_genes)
    base[5] = 0.0  # a target gene never seen in controls (zero control mean)
    kd = {t: rng.choice([0.05, 0.2, 0.28, 0.35, 0.6, 1.0]) for t in targets}
    lam = np.tile(base, (len(perts), 1))
    for i, p in enumerate(perts):
        if p in names:
            lam[i, names.index(p)] *= kd[p] * rng.uniform(0, 2)
    X = rng.poisson(lam).astype(np.float32)
    obs = pd.DataFrame({"gene": pd.Categorical(perts)}, index=[f"{prefix}{i}" for i in range(len(perts))])
    var = pd.DataFrame({"gene_name": names}, index=[f"ENSG{i:05d}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def as64(adata):
    """Copy with float64 X: cell-load then averages in float64 like the port, so float32 rounding of a
    control mean cannot flip a ratio that sits exactly on a threshold."""
    a = adata.copy()
    a.X = np.asarray(a.X, dtype=np.float64)
    return a


def write(adata, path, csr: bool):
    a = adata.copy()
    if csr:
        a.X = sp.csr_matrix(a.X)
    a.write_h5ad(path)
    return path


# --- stage 1: perturbation-level ratio -----------------------------------------------------------
def test_stage1_keeps_mean_knockdown_below_residual_and_is_strict():
    te = te_from(
        {"A": [0.29] * 30, "B": [0.31] * 30, "C": [0.30] * 30},
        {"A": 1.0, "B": 1.0, "C": 1.0},
    )
    keep, rep = filter_on_target_knockdown(te, cell_residual_expression=0.5)
    assert kept_targets(te, keep) == {"A"}
    assert rep["per_target"]["C"]["kd_ratio"] == pytest.approx(0.30)
    assert not rep["per_target"]["C"]["stage1"]  # strict '<': ratio exactly 0.30 fails
    assert rep["stages"]["stage1_perturbation"]["perturbations"] == 1


def test_stage1_uses_the_mean_not_each_cell():
    # mean 0.25 passes although half the cells sit at 0.45; all of them then pass stage 2 (< 0.5)
    te = te_from({"A": [0.05] * 15 + [0.45] * 15}, {"A": 1.0})
    keep, rep = filter_on_target_knockdown(te)
    assert rep["per_target"]["A"]["kd_ratio"] == pytest.approx(0.25)
    assert keep.sum() == 5 + 30


# --- stage 2: cell-level ratio -------------------------------------------------------------------
def test_stage2_drops_cells_at_or_above_cell_residual():
    vals = [0.0] * 30 + [0.49, 0.5, 0.51, 2.0]  # mean stays < 0.3
    te = te_from({"A": vals}, {"A": 1.0})
    keep, rep = filter_on_target_knockdown(te)
    own_kept = te.own[keep & (te.perts == "A")]
    assert sorted(own_kept.tolist()) == sorted([0.0] * 30 + [0.49])
    assert rep["per_target"]["A"]["n_after_stage2"] == 31


def test_ratios_are_relative_to_the_control_mean():
    te = te_from({"A": [1.0] * 40, "B": [1.0] * 40}, {"A": 4.0, "B": 2.0})
    keep, _ = filter_on_target_knockdown(te)
    assert kept_targets(te, keep) == {"A"}  # 1/4 < 0.3 and < 0.5; 1/2 fails stage 1


# --- stage 3: min_cells --------------------------------------------------------------------------
def test_min_cells_counts_cells_left_after_stage2():
    te = te_from(
        {"A": [0.0] * 30, "B": [0.0] * 29, "C": [0.0] * 29 + [5.0] * 3},  # C: 32 cells, 29 survive stage 2
        {"A": 10.0, "B": 10.0, "C": 10.0},
    )
    keep, rep = filter_on_target_knockdown(te)
    assert kept_targets(te, keep) == {"A"}
    assert rep["per_target"]["C"]["n_cells"] == 32
    assert rep["per_target"]["C"]["n_after_stage2"] == 29
    assert rep["per_target"]["C"]["n_kept"] == 0
    keep, _ = filter_on_target_knockdown(te, min_cells=29)
    assert kept_targets(te, keep) == {"A", "B", "C"}


# --- controls, unmatched, zero control mean ------------------------------------------------------
def test_controls_are_always_kept():
    te = te_from({"A": [9.0] * 40}, {"A": 1.0}, n_ctrl=7)
    keep, rep = filter_on_target_knockdown(te)
    assert keep.sum() == 7 and (te.perts[keep] == NT).all()
    assert rep["stages"]["stage3_min_cells"]["perturbations"] == 0
    # nothing matched at all: still the controls
    te = te_from({"Z": [0.0] * 40}, {}, n_ctrl=7)
    keep, rep = filter_on_target_knockdown(te)
    assert keep.sum() == 7 and rep["unmatched"] == ["Z"]


def test_unmatched_targets_are_dropped_even_with_perfect_knockdown():
    te = te_from({"A": [0.0] * 40, "Z": [0.0] * 40}, {"A": 1.0})
    keep, rep = filter_on_target_knockdown(te)
    assert kept_targets(te, keep) == {"A"}
    assert rep["unmatched"] == ["Z"] and "Z" not in rep["per_target"]
    assert rep["stages"]["matched"]["perturbations"] == 1


@pytest.mark.parametrize("ctrl_mean", [0.0, 1e-12])
def test_zero_control_mean_drops_the_target(ctrl_mean):
    te = te_from({"A": [0.0] * 40, "B": [0.0] * 40}, {"A": ctrl_mean, "B": 1.0})
    keep, rep = filter_on_target_knockdown(te)
    assert kept_targets(te, keep) == {"B"}
    assert rep["per_target"]["A"]["kd_ratio"] == np.inf


def test_custom_control_label():
    te = te_from({"A": [0.0] * 40}, {"A": 1.0})
    te.perts = np.where(te.perts == NT, "ctrl", te.perts)
    keep, rep = filter_on_target_knockdown(te, control="ctrl")
    assert keep.all() and rep["n_controls"] == 5


# --- reader --------------------------------------------------------------------------------------
@pytest.mark.parametrize("csr", [False, True])
def test_reader_uses_first_occurrence_of_duplicated_gene_names(tmp_path, csr):
    X = np.array([[4, 0, 9], [2, 1, 7], [0, 1, 3], [1, 0, 5]], dtype=np.float32)
    obs = pd.DataFrame({"gene": [NT, NT, "D", "E"]}, index=list("abcd"))
    var = pd.DataFrame({"gene_name": ["D", "X", "D"]}, index=["e1", "e2", "e3"])
    path = write(anndata.AnnData(X=X, obs=obs, var=var), tmp_path / "dup.h5ad", csr)
    te = kf.read_target_expression(path, block=3)
    assert te.matched == frozenset({"D"})
    assert te.ctrl_means == {"D": 3.0}  # column e1, not e3 (mean 8)
    np.testing.assert_array_equal(te.own, [np.nan, np.nan, 0.0, np.nan])
    np.testing.assert_array_equal(te.library, X.sum(axis=1))
    assert te.n_ctrl == 2 and te.ctrl_sums["D"] == 6.0


@pytest.mark.parametrize("csr", [False, True])
@pytest.mark.parametrize("transform", ["counts", "cp10k", "log1p_cp10k"])
def test_reader_matches_in_memory_values_for_every_block_size(tmp_path, csr, transform):
    a = random_screen(1)
    path = write(a, tmp_path / "s.h5ad", csr)
    X = a.X.astype(np.float64)
    lib = X.sum(axis=1)
    vals = {"counts": X, "cp10k": 1e4 * X / lib[:, None], "log1p_cp10k": np.log1p(1e4 * X / lib[:, None])}[transform]
    names = list(a.var.gene_name)
    perts = a.obs.gene.astype(str).to_numpy()
    ctrl = perts == NT
    for block in (17, 100000):
        te = kf.read_target_expression(path, transform=transform, block=block)
        np.testing.assert_allclose(te.library, lib)
        for t in te.matched:
            j = names.index(t)
            assert te.ctrl_means[t] == pytest.approx(vals[ctrl, j].mean(), rel=1e-12)
            np.testing.assert_allclose(te.own[perts == t], vals[perts == t, j], rtol=1e-12)
        assert np.isnan(te.own[ctrl | ~np.isin(perts, list(te.matched))]).all()
    assert te.matched == frozenset(set(perts) & set(names))


def test_reader_rejects_unknown_columns_and_transforms(tmp_path):
    path = write(random_screen(2), tmp_path / "s.h5ad", csr=False)
    with pytest.raises(KeyError):
        kf.read_target_expression(path, gene_col="symbol")
    with pytest.raises(KeyError):
        kf.read_target_expression(path, pert_key="target")
    with pytest.raises(ValueError):
        kf.read_target_expression(path, transform="log1p")


# --- cross-checks against cell-load --------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("csr", [False, True])
def test_matches_cellload_on_random_screens(tmp_path, seed, csr):
    a = random_screen(seed)
    path = write(a, tmp_path / "s.h5ad", csr)
    ref = cellload_loop_filter(as64(a), min_cells=25)
    te = kf.read_target_expression(path)
    keep, rep = filter_on_target_knockdown(te, min_cells=25)
    assert list(a.obs_names[keep]) == list(ref.obs_names)
    assert set(rep["kept_targets"]) == set(ref.obs.gene.astype(str)) - {NT}
    assert "NOT_IN_VAR1" in rep["unmatched"]
    assert rep["per_target"]["G5"]["kd_ratio"] == np.inf  # zero control mean


@pytest.mark.parametrize("seed", range(4))
def test_matches_cellload_head_on_screens_without_duplicated_names(tmp_path, seed):
    a = random_screen(seed, dup=False)
    ref_head = cellload_head_filter(as64(a), min_cells=25)
    ref_loop = cellload_loop_filter(as64(a), min_cells=25)
    keep, _ = filter_on_target_knockdown(
        kf.read_target_expression(write(a, tmp_path / "s.h5ad", csr=True)), min_cells=25
    )
    assert list(a.obs_names[keep]) == list(ref_head.obs_names) == list(ref_loop.obs_names)


def test_cellload_head_raises_on_duplicated_gene_names_but_the_port_does_not(tmp_path):
    """HSPA14 is duplicated in all four raw screens (TBCE too in K562): HEAD cannot have built STATE."""
    a = random_screen(0, dup=True)
    with pytest.raises(pd.errors.InvalidIndexError):
        cellload_head_filter(as64(a))
    keep, rep = filter_on_target_knockdown(kf.read_target_expression(write(a, tmp_path / "s.h5ad", csr=False)))
    assert list(a.obs_names[keep]) == list(cellload_loop_filter(as64(a)).obs_names)
    # first occurrence: G3's ratio uses column 3, not the duplicated last column
    ctrl = (a.obs.gene == NT).to_numpy()
    assert rep["per_target"]["G3"]["ctrl_mean"] == pytest.approx(a.X[ctrl, 3].astype(float).mean())


def test_random_screens_exercise_every_stage(tmp_path):
    """Guard for the cross-check: the random screens must hit each drop reason, not just pass."""
    seen = {"stage1": 0, "stage2": 0, "min_cells": 0, "kept": 0}
    for seed in range(6):
        te = kf.read_target_expression(write(random_screen(seed), tmp_path / f"{seed}.h5ad", csr=False))
        _, rep = filter_on_target_knockdown(te, min_cells=25)
        for r in rep["per_target"].values():
            seen["stage1"] += not r["stage1"]
            seen["stage2"] += r["stage1"] and r["n_after_stage2"] < r["n_cells"]
            seen["min_cells"] += r["stage1"] and 0 < r["n_after_stage2"] < 25
            seen["kept"] += r["n_kept"] > 0
    assert all(v > 0 for v in seen.values()), seen


@pytest.mark.parametrize("params", [(0.30, 0.50, 30), (0.5, 0.8, 10), (0.2, 0.3, 5)])
def test_matches_cellload_for_other_thresholds(tmp_path, params):
    res, cell_res, n = params
    a = random_screen(11)
    te = kf.read_target_expression(write(a, tmp_path / "s.h5ad", csr=False))
    ref = cellload_loop_filter(as64(a), residual_expression=res, cell_residual_expression=cell_res, min_cells=n)
    keep, _ = filter_on_target_knockdown(te, residual_expression=res, cell_residual_expression=cell_res, min_cells=n)
    assert list(a.obs_names[keep]) == list(ref.obs_names)


def test_concat_matches_cellload_on_the_inner_joined_screens(tmp_path):
    a = random_screen(3, prefix="a")
    b = random_screen(4, prefix="b")
    # b lacks G7's gene: after an inner join G7 is unmatched in both screens
    b = b[:, b.var.gene_name != "G7"].copy()
    merged = anndata.concat([a, b], join="inner", merge="same")
    assert "G7" not in set(merged.var.gene_name)
    ref = cellload_loop_filter(as64(merged), min_cells=40)
    parts = [kf.read_target_expression(write(x, tmp_path / f"{i}.h5ad", csr=False)) for i, x in enumerate([a, b])]
    te = concat_target_expression(parts)
    assert "G7" not in te.matched and "G7" in parts[0].matched
    assert te.n_ctrl == 600
    keep, _ = filter_on_target_knockdown(te, min_cells=40)
    assert list(merged.obs_names[keep]) == list(ref.obs_names)
    # pooled control mean is the mean over both screens' controls
    ctrl = np.r_[a.X[a.obs.gene == NT][:, 0], b.X[b.obs.gene == NT][:, 0]]
    assert te.ctrl_means["G0"] == pytest.approx(ctrl.mean())


def test_concat_without_sums_falls_back_to_weighted_means():
    p1 = te_from({"A": [0.0] * 20}, {"A": 2.0}, n_ctrl=10)
    p2 = te_from({"A": [0.0] * 20, "B": [0.0] * 40}, {"A": 5.0, "B": 1.0}, n_ctrl=30)
    te = concat_target_expression([p1, p2])
    assert te.ctrl_means == {"A": pytest.approx((2.0 * 10 + 5.0 * 30) / 40)}  # B has no value in p1
    assert te.n_ctrl == 40 and len(te.perts) == 10 + 20 + 30 + 20 + 40
    assert np.isnan(te.own[te.perts == "B"]).all()
    keep, rep = filter_on_target_knockdown(te)
    assert kept_targets(te, keep) == {"A"} and rep["unmatched"] == ["B"]


def test_module_constants_name_the_reproducing_variant():
    assert kf.TRANSFORM == "counts" and kf.SCOPE == "per_screen"
    assert len(kf.CELL_LOAD_COMMIT) == 40
