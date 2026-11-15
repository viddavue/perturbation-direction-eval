"""dashboard_data.py must reproduce exactly what evaluate.py scored, from evaluate.py's own outputs."""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.baselines import baselines
from experiments.perturbation_direction.dashboard import dashboard_data as dd
from experiments.perturbation_direction.evaluation import evaluate, ground_truth
from experiments.perturbation_direction.evaluation.metrics import evaluate as score
from experiments.perturbation_direction.evaluation.metrics import remove_shared
from scipy import sparse

CTRL = "non-targeting"
EDGES = [5, 20, 100, 500, np.inf]


def build_run(root: Path) -> Path:
    """Synthetic HepG2-style evaluation (shared + cluster-specific responses), run through the real scripts."""
    rng = np.random.default_rng(3)
    n_genes, n_clusters = 30, 3
    genes = [f"g{i}" for i in range(n_genes)]
    shared = np.zeros(n_genes)
    shared[:6] = 0.8
    sig = {k: np.zeros((n_clusters, n_genes)) for k in ("other", "target")}
    for k in sig:
        for c in range(n_clusters):
            sig[k][c, rng.choice(np.arange(6, n_genes), 8, replace=False)] = rng.choice([-0.8, 0.8], 8)
    perts = {f"c{c}x{i}": c for c in range(n_clusters) for i in range(12)}
    test_perts = [p for p in perts if p.endswith(("x0", "x1"))]

    def cells(line, pert, n=40):
        eff = np.zeros(n_genes) if pert == CTRL else shared + sig["target" if line == "T" else "other"][perts[pert]]
        X = rng.poisson(6 * np.exp(eff), size=(n, n_genes)).astype(np.float32)
        obs = pd.DataFrame({"cell_line": line, "gene": pert}, index=[f"{line}_{pert}_{i}" for i in range(n)])
        return ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes))

    test = ad.concat([cells("T", p) for p in test_perts])
    train = ad.concat(
        [cells("T", CTRL)]
        + [cells("T", p) for p in perts if p not in test_perts]
        + [cells(line, p) for line in ("A", "B") for p in [CTRL, *perts]]
    )
    test.write_h5ad(root / "test.h5ad")
    train.write_h5ad(root / "train.h5ad")
    meta = root / "meta.json"
    meta.write_text(json.dumps({"genes": genes, "labels": {"cell_line": ["A", "B", "T"], "gene": [CTRL, *perts]}}))
    truth, base, out = root / "truth", root / "baselines", root / "eval"
    ground_truth.main(
        [
            "--pert-h5ad",
            str(root / "test.h5ad"),
            "--ctrl-h5ad",
            str(root / "train.h5ad"),
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
    baselines.main(
        [
            "--train-h5ad",
            str(root / "train.h5ad"),
            "--pert-key",
            "gene",
            "--control",
            CTRL,
            "--context-key",
            "cell_line",
            "--target-context",
            "T",
            "--truth-dir",
            str(truth),
            "--out-dir",
            str(base),
        ]
    )
    # a noisy "model": true pseudobulk profiles plus noise, control row included (generate.py format)
    log = ground_truth.lognorm(ad.concat([test, train[(train.obs.cell_line == "T") & (train.obs.gene == CTRL)]]))
    prof = ground_truth.group_means(log, "gene").loc[[CTRL, *test_perts]]
    (prof + rng.normal(scale=0.15, size=prof.shape)).to_parquet(root / "model.parquet")
    evaluate.main(
        [
            "--truth-dir",
            str(truth),
            "--exclude-target",
            "--pred",
            f"model={root / 'model.parquet'}",
            "--baseline",
            f"knn={base / 'delta_knn.parquet'}",
            f"generic={base / 'delta_generic.parquet'}",
            "--specific-ref",
            str(base / "delta_generic.parquet"),
            "--min-degs",
            "1",
            "--out-dir",
            str(out),
        ]
    )
    return out


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_run(tmp_path_factory.mktemp("dash"))


def test_find_and_load_run(run_dir):
    assert dd.find_runs(run_dir.parent) == [run_dir]
    run = dd.load_run(run_dir)
    assert run.methods == ["model", "model (shuffled)", "knn", "generic", dd.CEILING]
    assert set(run.per_pert) == {"full", "specific"}
    assert "generic" not in run.per_pert["specific"]  # nothing specific left for the shared response
    kinds = {m["name"]: m["kind"] for m in run.manifest["methods"]}
    assert kinds == {"model": "pred", "model (shuffled)": "shuffled", "knn": "baseline", "generic": "baseline"}
    assert run.manifest["control"] == CTRL and run.manifest["exclude_target"] is True


@pytest.mark.parametrize("name", ["model", "model (shuffled)", "knn", "generic"])
def test_effects_reproduce_evaluate_scores(run_dir, name):
    run = dd.load_run(run_dir)
    truth = dd.load_truth(dd.resolve(run.manifest["truth_dir"]))
    eff = dd.load_effects(run.manifest, name, truth)
    excl = {p: [p] for p in truth.perturbations}
    pd.testing.assert_frame_equal(
        score(eff, truth.delta, truth.deg, exclude_genes=excl),
        run.per_pert["full"][name],
        check_dtype=False,
        check_names=False,
    )
    if name in run.per_pert["specific"]:
        ref = dd.shared_reference(run.manifest)
        t_s, deg_s = dd.specific_truth(truth, ref)
        got = score(remove_shared(eff, ref)[0], t_s, deg_s, exclude_genes=excl)
        pd.testing.assert_frame_equal(got, run.per_pert["specific"][name], check_dtype=False, check_names=False)


def test_strength_table_matches_evaluate(run_dir):
    run = dd.load_run(run_dir)
    truth = dd.load_truth(dd.resolve(run.manifest["truth_dir"]))
    n_true = dd.deg_counts(truth.deg, exclude_target=True)
    got = dd.strength_table(run.per_pert["full"], n_true, EDGES, "balanced_acc")
    expected = pd.read_csv(run_dir / "by_strength.csv", index_col=0)
    pd.testing.assert_frame_equal(got.loc[expected.index], expected, check_names=False, check_dtype=False)
    assert dd.bin_sizes(n_true, EDGES).sum() == (n_true >= 5).sum()


def test_perturbation_detail_matches_per_perturbation_rows(run_dir):
    run = dd.load_run(run_dir)
    truth = dd.load_truth(dd.resolve(run.manifest["truth_dir"]))
    effects = {n: dd.load_effects(run.manifest, n, truth) for n in ("model", "knn")}
    pert = truth.perturbations[0]
    genes, metrics = dd.perturbation_detail(pert, truth.delta, truth.deg, truth.padj, effects)
    for n in effects:
        row = run.per_pert["full"][n].loc[pert]
        assert metrics.loc[n, "sign_acc"] == pytest.approx(row["sign_acc"])
        assert metrics.loc[n, "n_deg"] == row["n_deg"]
    assert genes["deg"].iloc[: int(genes["deg"].sum())].all()  # DEGs listed first
    assert {"true_delta", "padj", "model", "model ✓", "knn", "knn ✓"} <= set(genes.columns)


def test_resolve_falls_back_to_relative_path(tmp_path):
    (tmp_path / "x.parquet").write_text("")
    entry = {"path": "x.parquet", "abs_path": "/nonexistent/x.parquet"}
    assert dd.resolve(entry, base=tmp_path) == tmp_path / "x.parquet"
    with pytest.raises(FileNotFoundError):
        dd.resolve({"path": "missing.parquet", "abs_path": "/nonexistent/m.parquet"}, base=tmp_path)
