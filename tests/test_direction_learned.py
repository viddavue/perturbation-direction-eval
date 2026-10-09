"""Tests for the ported trained direction baselines (learned_baselines.py, direction_models.py)."""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

import evaluate
import ground_truth
import learned_baselines
from ground_truth import group_stats, lognorm, welch_degs
from metrics import benjamini_hochberg

CTRL = "non-targeting"


def test_benjamini_hochberg():
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    np.testing.assert_allclose(benjamini_hochberg(p)[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert np.isnan(benjamini_hochberg(p)[4])


def test_welch_degs_finds_shifted_genes():
    rng = np.random.default_rng(0)
    n_genes = 40
    rate = np.full(n_genes, 8.0)
    pert_rate = rate.copy()
    pert_rate[:3] *= 2  # +24 counts ...
    pert_rate[3:6] = 0  # ... -24 counts: library size unchanged, so other genes do not shift compositionally
    X = np.vstack([rng.poisson(rate, (200, n_genes)), rng.poisson(pert_rate, (100, n_genes))]).astype(np.float32)
    obs = pd.DataFrame({"gene": [CTRL] * 200 + ["P"] * 100}, index=[f"c{i}" for i in range(300)])
    a = lognorm(ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])))
    mean, var, n = group_stats(a, "gene")
    delta, padj = welch_degs(mean, var, n, CTRL, ["P"], 0.05, 0.1)
    deg = (padj < 0.05) & (delta.abs() > 0.1)
    assert set(deg.columns[deg.loc["P"]]) == {f"g{i}" for i in range(6)}
    assert (delta.loc["P", ["g0", "g1", "g2"]] > 0).all() and (delta.loc["P", ["g3", "g4", "g5"]] < 0).all()


@pytest.mark.parametrize("models", [["prior", "gbdt", "mlp", "bilinear"]])
def test_learned_baselines_end_to_end(tmp_path, models):
    """Clusters of perturbations share a response in each line; the target line's response differs from the other
    lines' but is shared within a cluster, so it can only be learned from the target line's training perturbations."""
    rng = np.random.default_rng(1)
    n_genes, n_clusters, per = 30, 3, 12
    genes = [f"g{i}" for i in range(n_genes)]
    sig = {line: np.zeros((n_clusters, n_genes)) for line in ("A", "B", "T")}
    for line in sig:
        for c in range(n_clusters):
            sig[line][c, rng.choice(n_genes, 8, replace=False)] = rng.choice([-0.9, 0.9], 8)
    perts = {f"c{c}x{i}": c for c in range(n_clusters) for i in range(per)}
    test_perts = [p for p in perts if p.endswith(("x0", "x1"))]

    def cells(line, pert, n=40):
        eff = np.zeros(n_genes) if pert == CTRL else sig[line][perts[pert]]
        X = rng.poisson(6 * np.exp(eff), size=(n, n_genes)).astype(np.float32)
        obs = pd.DataFrame({"cell_line": line, "gene": pert}, index=[f"{line}_{pert}_{i}" for i in range(n)])
        return ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))

    test = ad.concat([cells("T", p) for p in test_perts])
    train = ad.concat(
        [cells("T", CTRL)]
        + [cells("T", p) for p in perts if p not in test_perts]
        + [cells(line, p) for line in ("A", "B") for p in [CTRL, *perts]]
    )
    test.write_h5ad(tmp_path / "test.h5ad")
    train.write_h5ad(tmp_path / "train.h5ad")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"genes": genes, "labels": {"cell_line": ["A", "B", "T"], "gene": [CTRL, *perts]}}))
    truth, out = tmp_path / "truth", tmp_path / "learned"
    ground_truth.main(
        [
            "--pert-h5ad",
            str(tmp_path / "test.h5ad"),
            "--ctrl-h5ad",
            str(tmp_path / "train.h5ad"),
            "--pert-key",
            "gene",
            "--control",
            CTRL,
            "--context",
            "cell_line=T",
            "--metadata-json",
            str(meta),
            "--out-dir",
            str(truth),
        ]
    )
    learned_baselines.main(
        [
            "--train-h5ad",
            str(tmp_path / "train.h5ad"),
            "--truth-dir",
            str(truth),
            "--pert-key",
            "gene",
            "--control",
            CTRL,
            "--context-key",
            "cell_line",
            "--target-context",
            "T",
            "--models",
            *models,
            "--epochs",
            "30",
            "--batch-size",
            "256",
            "--embedding-dim",
            "8",
            "--device",
            "cpu",
            "--out-dir",
            str(out),
        ]
    )
    info = json.loads((out / "learned_info.json").read_text())
    assert set(info["labels_per_context"]) == {"A", "B", "T"}
    assert info["test_perts_without_embedding"] == []

    evaluate.main(
        [
            "--truth-dir",
            str(truth),
            "--baseline",
            *[f"{m}={out / f'delta_{m}.parquet'}" for m in models],
            "--min-degs",
            "1",
            "--out-dir",
            str(tmp_path / "eval"),
        ]
    )
    s = pd.read_csv(tmp_path / "eval" / "summary.csv", index_col=0)["balanced_acc_median"]
    for m in ("mlp", "gbdt"):
        assert s[m] > 0.85, s  # cluster structure + target-line training perturbations -> right directions
