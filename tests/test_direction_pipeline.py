"""End-to-end check of ground_truth -> baselines -> evaluate on synthetic data with known effects."""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

import baselines
import evaluate
import ground_truth

GENES = [f"g{i}" for i in range(30)]
# perturbation -> (up genes, down genes); identical effect in every cell line
EFFECTS = {
    "p1": (range(0, 5), range(5, 10)),
    "p2": (range(5, 10), range(0, 5)),
    "p3": (range(10, 15), range(15, 20)),
    "p4": (range(20, 25), range(25, 30)),
}
CTRL = "non-targeting"


def make_cells(rng, line, pert, n=80):
    rate = np.full(len(GENES), 6.0)
    if pert != CTRL:
        up, down = EFFECTS[pert]
        rate[list(up)] *= 3
        rate[list(down)] /= 3
    X = rng.poisson(rate, size=(n, len(GENES)))
    obs = pd.DataFrame({"cell_line": line, "gene": pert}, index=[f"{line}_{pert}_{i}" for i in range(n)])
    return ad.AnnData(X=sparse.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=GENES))


def test_pipeline_recovers_known_directions(tmp_path):
    rng = np.random.default_rng(0)
    test_perts = ["p1", "p2", "p3"]
    # held-out context T: test perturbations in test; T controls + one other T perturbation in train
    test = ad.concat([make_cells(rng, "T", p) for p in test_perts])
    train = ad.concat(
        [make_cells(rng, "T", CTRL), make_cells(rng, "T", "p4")]
        + [make_cells(rng, line, p) for line in ("A", "B") for p in [CTRL, *test_perts, "p4"]]
    )
    test.write_h5ad(tmp_path / "test.h5ad")
    train.write_h5ad(tmp_path / "train.h5ad")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"genes": GENES, "labels": {"cell_line": ["A", "B", "T"], "gene": [CTRL, *EFFECTS]}}))

    truth = tmp_path / "truth"
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
    deg = pd.read_parquet(truth / "deg.parquet")
    for p in test_perts:
        up, down = EFFECTS[p]
        assert deg.loc[p, [GENES[i] for i in [*up, *down]]].mean() > 0.9  # true effects detected
    cond = pd.read_csv(truth / "conditions.csv")
    assert cond["id"].tolist() == [CTRL, *test_perts] and (cond["cell_line"] == "T").all()

    base = tmp_path / "baselines"
    baselines.main(
        [
            "--train-h5ad",
            str(tmp_path / "train.h5ad"),
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
    info = json.loads((base / "baselines_info.json").read_text())
    assert info["generic_source"] == "target_context_train" and info["target_context_train_perturbations"] == 1
    assert info["learned_baselines"] == {}  # too few target-context training perturbations to fit
    assert not list(base.glob("delta_transfer_*.parquet"))  # per-line transfer files are no longer written

    # an oracle "model": profiles equal to the true pseudobulk means (control row + perturbation rows)
    log = ground_truth.lognorm(ad.concat([test, train[(train.obs.cell_line == "T") & (train.obs.gene == CTRL)]]))
    prof = ground_truth.group_means(log, "gene").loc[[CTRL, *test_perts]]
    prof.to_parquet(tmp_path / "oracle.parquet")

    out = tmp_path / "eval"
    evaluate.main(
        [
            "--truth-dir",
            str(truth),
            "--pred",
            f"oracle={tmp_path / 'oracle.parquet'}",
            "--baseline",
            f"transfer={base / 'delta_transfer.parquet'}",
            f"generic={base / 'delta_generic.parquet'}",
            "--min-degs",
            "1",
            "--out-dir",
            str(out),
        ]
    )
    s = pd.read_csv(out / "summary.csv", index_col=0)
    assert s.loc["oracle", "balanced_acc_median"] == 1.0
    assert s.loc["transfer", "balanced_acc_median"] > 0.95  # same effect in other lines
    assert s.loc["oracle (shuffled)", "balanced_acc_median"] < 0.75  # effects are perturbation-specific
    assert s.loc["generic", "balanced_acc_median"] < 0.75  # p4's effect does not transfer to p1-p3
    assert (out / "accuracy_by_strength.png").exists()


def test_learned_baselines_and_specific_report(tmp_path):
    """Shared response + cluster-specific responses: knn must recover the specific part; in the
    perturbation-specific report the shared-response (generic) baseline is dropped and the oracle stays perfect."""
    rng = np.random.default_rng(1)
    n_genes, n_clusters = 30, 3
    genes = [f"g{i}" for i in range(n_genes)]
    shared = np.zeros(n_genes)
    shared[:6] = 0.8  # every perturbation in every line moves g0-g5 up
    sig_other, sig_target = np.zeros((n_clusters, n_genes)), np.zeros((n_clusters, n_genes))
    for c in range(n_clusters):
        sig_other[c, rng.choice(np.arange(6, n_genes), 8, replace=False)] = rng.choice([-0.8, 0.8], 8)
        sig_target[c, rng.choice(np.arange(6, n_genes), 8, replace=False)] = rng.choice([-0.8, 0.8], 8)
    perts = {f"c{c}_{i}": c for c in range(n_clusters) for i in range(12)}
    test_perts = [p for p in perts if p.endswith(("_0", "_1"))]

    def cells(line, pert, n=40):
        eff = np.zeros(n_genes)
        if pert != CTRL:
            c = perts[pert]
            eff = shared + (sig_target[c] if line == "T" else sig_other[c])
        X = rng.poisson(6 * np.exp(eff), size=(n, n_genes))
        obs = pd.DataFrame({"cell_line": line, "gene": pert}, index=[f"{line}_{pert}_{i}" for i in range(n)])
        return ad.AnnData(X=sparse.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=genes))

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

    truth, base, out = tmp_path / "truth", tmp_path / "baselines", tmp_path / "eval"
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
    baselines.main(
        [
            "--train-h5ad",
            str(tmp_path / "train.h5ad"),
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
    info = json.loads((base / "baselines_info.json").read_text())
    assert info["learned_baselines"]["knn"]["n_train"] == 30
    assert info["learned_baselines"]["test_perts_with_target_train_cells"] == []

    log = ground_truth.lognorm(ad.concat([test, train[(train.obs.cell_line == "T") & (train.obs.gene == CTRL)]]))
    ground_truth.group_means(log, "gene").loc[[CTRL, *test_perts]].to_parquet(tmp_path / "oracle.parquet")
    baseline_args = [f"{n}={base / f'delta_{n}.parquet'}" for n in ("transfer", "generic", "knn")]
    evaluate.main(
        [
            "--truth-dir",
            str(truth),
            "--pred",
            f"oracle={tmp_path / 'oracle.parquet'}",
            "--baseline",
            *baseline_args,
            "--specific-ref",
            str(base / "delta_generic.parquet"),
            "--min-degs",
            "1",
            "--out-dir",
            str(out),
        ]
    )

    s = pd.read_csv(out / "summary.csv", index_col=0)
    sp = pd.read_csv(out / "summary_specific.csv", index_col=0)
    assert s.loc["knn", "balanced_acc_median"] > 0.9
    assert sp.loc["knn", "balanced_acc_median"] > 0.9  # recovers the cluster-specific target response
    assert sp.loc["oracle", "balanced_acc_median"] == 1.0
    assert "generic" not in sp.index  # identical to the reference -> nothing left to score
    assert sp.loc["transfer", "balanced_acc_median"] < 0.75  # other lines' specific response is unrelated
    assert "noise ceiling (split-half)" in sp.index
    assert "shared_strength_corr" in sp.columns and sp["shared_strength_corr"].notna().any()
    assert (out / "accuracy_by_strength_specific.png").exists()
