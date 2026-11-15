"""Fixed-panel retrieval, conditioning coverage, and denoising ablation safeguards."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.generation.denoising_ablation import observed_effects, score_variants
from experiments.perturbation_direction.vcc.vcc_cells import check_counts, metric_coverage, sample_panel
from experiments.perturbation_direction.vcc.vcc_pilot import select_panel, split_groups
from experiments.perturbation_direction.vcc.vcc_profiles import align_panel, project_shared, retrieval
from scipy import sparse


def test_panel_intersection_is_global_not_pairwise():
    real = pd.DataFrame(np.ones((4, 3)), index=list("abcd"), columns=list("xyz"))
    truth, methods = align_panel(real, {"one": real.iloc[:3, :2], "two": real.iloc[1:, 1:]})
    assert list(truth.index) == ["b", "c"]
    assert list(truth.columns) == ["y"]
    assert all(frame.equals(truth) for frame in methods.values())


def test_reject_trivial_panel_and_nonfinite():
    real = pd.DataFrame([[1], [2]], index=["a", "b"], columns=["x"])
    with pytest.raises(ValueError, match=">=2"):
        align_panel(real, {"one": real.iloc[:1]})
    with pytest.raises(ValueError, match="non-finite"):
        align_panel(real, {"one": real * np.nan})


def test_shared_projection_removes_parallel_component():
    frame = pd.DataFrame([[2.0, 3.0], [4.0, -1.0]], columns=["x", "y"])
    result = project_shared(frame, pd.Series({"x": 1.0, "y": 0.0}))
    np.testing.assert_allclose(result.x, 0)
    np.testing.assert_allclose(result.y, frame.y)
    generic = pd.DataFrame([[0.1, 0.3]] * 3, columns=["x", "y"])
    np.testing.assert_array_equal(project_shared(generic, generic.iloc[0]), 0)


def test_upstream_retrieval_perfect_and_zero_ties():
    pytest.importorskip("cell_eval2")
    truth = pd.DataFrame(np.eye(3), index=["p1", "p2", "p3"], columns=["x", "y", "z"])
    np.testing.assert_allclose(retrieval(truth, truth).effect_pds, 1)
    zero = retrieval(truth * 0, truth)
    np.testing.assert_allclose(zero.effect_pds, 0.5)
    np.testing.assert_allclose(zero.top1_tie_adjusted, 1 / 3)


def test_upstream_panel_exclusion_not_own_gene_only():
    pytest.importorskip("cell_eval2")
    truth = pd.DataFrame([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0]], index=["a", "b"], columns=["a", "b", "x"])
    np.testing.assert_allclose(retrieval(truth, truth).effect_pds, 0.5)


def test_denoising_filter_changes_mask_not_effects():
    truth = pd.DataFrame([[1.0, -1.0, 2.0]], index=["p"], columns=["x", "y", "z"])
    detection = pd.DataFrame([[1.0, 1.0, 0.1]], index=truth.index, columns=truth.columns)
    results = score_variants({"perfect": truth}, truth, truth, truth.astype(bool), detection)
    assert results.n_deg.tolist() == [3, 2, 3, 2]
    np.testing.assert_allclose(results.balanced_acc, 1)


def test_observed_means_are_per_cell_not_pooled_counts():
    # Same group: unequal libraries; arithmetic mean frequencies differ from pooled frequencies.
    cells = ad.AnnData(
        sparse.csr_matrix([[1, 9], [90, 10], [5, 5]]),
        obs=pd.DataFrame({"gene": ["p", "p", "c"]}, index=["1", "2", "3"]),
        var=pd.DataFrame(index=["x", "y"]),
    )
    np.testing.assert_allclose(observed_effects(cells, ["p"], "gene", "c"), 0, atol=1e-12)


def test_cell_runner_rejects_fractional_counts_and_mismatched_panels():
    cells = ad.AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame({"gene": ["c", "c", "p", "p"]}, index=["1", "2", "3", "4"]),
        var=pd.DataFrame(index=["x", "y"]),
    )
    with pytest.raises(ValueError, match="integer"):
        check_counts(ad.AnnData(cells.X * 0.5))
    with pytest.raises(ValueError, match="panels differ"):
        sample_panel(cells, cells[:2], "gene", "c", 2, 0)
    real, pred = sample_panel(cells, cells, "gene", "c", 2, 0)
    assert real.shape == pred.shape == (4, 2)


def test_real_halves_share_no_cells_including_controls():
    cells = ad.AnnData(
        np.ones((12, 2)), obs=pd.DataFrame({"gene": ["c"] * 6 + ["p"] * 6}, index=[str(i) for i in range(12)])
    )
    a, b = split_groups(cells, "gene", 3, 42)
    assert not set(a.obs_names) & set(b.obs_names)
    assert a.obs.gene.value_counts().to_dict() == {"c": 3, "p": 3}
    with pytest.raises(ValueError, match="independent cells"):
        split_groups(cells, "gene", 4, 42)


def test_pilot_panel_is_seeded_and_has_enough_cells():
    deg = pd.DataFrame([[False] * 6, [True] * 6, [True] * 6], index=["a", "b", "c"])
    panel = select_panel(deg, {"a": 64, "b": 64, "c": 30}, 32, 3, 42)
    assert panel.perturbation.tolist() == ["a", "b"]


def test_coverage_counts_nan_and_omitted_rows():
    result = metric_coverage([{"metric": "de", "value": 1.0}, {"metric": "de", "value": np.nan}], 3)
    assert result.n_finite.iloc[0] == 1
    assert result.n_unscored.iloc[0] == 2


def test_official_cpu_preset_end_to_end(tmp_path):
    pytest.importorskip("cell_eval2")
    from experiments.perturbation_direction.vcc.vcc_cells import main

    rng = np.random.default_rng(7)
    labels = ["non-targeting"] * 16 + ["p1"] * 16 + ["p2"] * 16
    counts = rng.poisson(10, (48, 24))
    counts[16:32, :8] *= 4
    counts[32:, 8:16] *= 4
    data = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=pd.DataFrame({"gene": labels}, index=[str(i) for i in range(48)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(24)]),
    )
    real, pred = tmp_path / "real.h5ad", tmp_path / "pred.h5ad"
    data.write_h5ad(real)
    data.write_h5ad(pred)
    out = tmp_path / "eval"
    main(["--real", str(real), "--pred", str(pred), "--n-cells", "16", "--out-dir", str(out)])
    summary = pd.read_csv(out / "summary_raw.csv")
    assert "expr_mse_unbiased_capped_norm" in set(summary.metric)
    assert "pds_cosine" in set(summary.metric)
    assert (out / "coverage.csv").exists()


def test_repeated_draws_report_spread_not_just_one_value(tmp_path):
    pytest.importorskip("cell_eval2")
    from experiments.perturbation_direction.vcc.vcc_cells import main

    rng = np.random.default_rng(11)
    labels = ["non-targeting"] * 20 + ["p1"] * 20 + ["p2"] * 20
    counts = rng.poisson(10, (60, 24))
    counts[20:40, :8] *= 4
    counts[40:, 8:16] *= 4
    data = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=pd.DataFrame({"gene": labels}, index=[str(i) for i in range(60)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(24)]),
    )
    real, pred = tmp_path / "real.h5ad", tmp_path / "pred.h5ad"
    data.write_h5ad(real)
    data.write_h5ad(pred)
    out = tmp_path / "eval"
    main(["--real", str(real), "--pred", str(pred), "--n-cells", "10", "--repeats", "2", "--out-dir", str(out)])
    across = pd.read_csv(out / "summary_across_repeats.csv", index_col=0)
    assert set(across.columns) >= {"mean", "min", "max"}
    assert (across["n_repeats"] == 2).all()
    assert (out / "metrics_raw_rep1.csv").exists()
    assert (out / "coverage_across_repeats.csv").exists()
