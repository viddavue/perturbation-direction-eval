import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from prepare_replogle_state import counts_from_lognorm


def test_counts_recovered_from_log1p_cp10k_of_gene_superset():
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(2, 0.3, size=(50, 300)).astype(np.float64)
    counts[0] = 0
    counts[1] = 2 * rng.integers(1, 5, size=300)  # smallest count is 2, not 1 ...
    counts[1, :3] = [3, 5, 7]  # ... but with odd counts present (all-even rows are not identifiable)
    # normalised over a larger gene set (extra genes dropped afterwards), stored as float32
    extra = rng.negative_binomial(2, 0.3, size=(50, 100))
    total = counts.sum(1, keepdims=True) + extra.sum(1, keepdims=True)
    x = np.log1p(counts / total * 1e4).astype(np.float32)

    rec, ok = counts_from_lognorm(x)
    assert ok.all()
    np.testing.assert_array_equal(rec, counts)


def test_non_count_rows_are_flagged():
    x = np.log1p(np.array([[0.37, 1.91, 2.53, 0.0]], dtype=np.float32))
    _, ok = counts_from_lognorm(x)
    assert not ok[0]


def test_prepare_script_end_to_end(tmp_path):
    import json

    import anndata as ad
    import pandas as pd
    from prepare_replogle_state import main

    rng = np.random.default_rng(1)
    n, all_genes = 120, [f"g{i}" for i in range(40)]
    model_genes = all_genes[30:5:-1]  # subset, different order
    counts = rng.negative_binomial(3, 0.4, size=(n, 40)).astype(np.float64)
    counts[:, 0] = 1  # guarantees a count of 1 per row
    umi = counts.sum(1) + rng.integers(0, 50, size=n)  # UMI_count covers genes dropped before saving
    obs = pd.DataFrame(
        {
            "cell_line": pd.Categorical(rng.choice(["hepg2", "k562"], n), categories=["hepg2", "k562"]),
            "gene": pd.Categorical(rng.choice(["non-targeting", "A", "B"], n), categories=["A", "B", "non-targeting"]),
            "UMI_count": umi,
        },
        index=[f"c{i}" for i in range(n)],
    )
    x = np.log1p(counts / umi[:, None] * 1e4).astype(np.float32)
    ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=all_genes)).write_h5ad(tmp_path / "src.h5ad")
    labels = {"cell_line": ["hepg2", "k562"], "gene": ["A", "B", "non-targeting"]}
    (tmp_path / "train.json").write_text(json.dumps({"genes": model_genes, "labels": labels}))
    (tmp_path / "test.json").write_text(
        json.dumps({"genes": model_genes, "labels": {"cell_line": ["hepg2"], "gene": ["A"]}})
    )

    main(
        [
            "--src",
            str(tmp_path / "src.h5ad"),
            "--train-json",
            str(tmp_path / "train.json"),
            "--test-json",
            str(tmp_path / "test.json"),
            "--chunk",
            "50",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    te = ad.read_h5ad(tmp_path / "out" / "replogle_test.h5ad")
    tr = ad.read_h5ad(tmp_path / "out" / "replogle_train.h5ad")
    is_test = (obs.cell_line == "hepg2") & (obs.gene == "A")
    assert te.n_obs == is_test.sum() and tr.n_obs == (~is_test).sum()
    assert set(te.obs.gene) == {"A"} and set(te.obs.cell_line) == {"hepg2"}
    assert list(te.var_names) == model_genes and list(tr.obs.cell_line.cat.categories) == ["hepg2", "k562"]
    col = [all_genes.index(g) for g in model_genes]
    np.testing.assert_array_equal(te.X.toarray(), counts[is_test.to_numpy()][:, col])
    np.testing.assert_array_equal(tr[obs.index[~is_test]].X.toarray(), counts[~is_test.to_numpy()][:, col])
