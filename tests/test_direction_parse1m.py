"""Parse1M preparation must select cells, keep model gene order, and never claim full coverage."""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from prepare_parse1m import check_integer_counts, main, select_rows


def write_source(tmp_path):
    rng = np.random.default_rng(0)
    genes = ["CFH", "GCLC", "LAP3", "AOC1", "EXTRA1", "EXTRA2"]
    labels = ["CD4 Naive"] * 8 + ["CD8 Naive"] * 4
    cytokines = ["PBS"] * 4 + ["IL-2"] * 4 + ["PBS"] * 2 + ["IL-2"] * 2
    counts = sparse.csr_matrix(rng.poisson(5, (12, len(genes))).astype(np.float32))
    obs = pd.DataFrame({"cell_type": pd.Categorical(labels), "cytokine": pd.Categorical(cytokines)},
                       index=[f"c{i}" for i in range(12)])
    path = tmp_path / "src.h5ad"
    ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"genes": ["CFH", "GCLC", "LAP3", "MISSING1", "MISSING2"],
                                "labels": {"cell_type": ["CD4 Naive", "CD8 Naive"],
                                           "cytokine": ["PBS", "IL-2"]}, "dataset": "parse1m"}))
    test = tmp_path / "test.json"
    test.write_text(json.dumps({"labels": {"cytokine": ["IL-2"]}, "genes": ["CFH"]}))
    return path, meta, test


def test_selection_respects_cell_type_and_cap(tmp_path):
    src, _, _ = write_source(tmp_path)
    rows, per = select_rows("CD4 Naive", {"PBS", "IL-2"}, 2, 0, src)
    assert len(rows) == 4
    assert per["PBS"] == {"available": 4, "selected": 2}
    assert per["IL-2"] == {"available": 4, "selected": 2}


def test_unknown_cytokine_or_cell_type_is_rejected(tmp_path):
    src, _, _ = write_source(tmp_path)
    with pytest.raises(ValueError, match="absent from source"):
        select_rows("CD4 Naive", {"NOT_A_CYTOKINE"}, 2, 0, src)
    with pytest.raises(ValueError, match="not in source"):
        select_rows("NK", {"PBS"}, 2, 0, src)


def test_end_to_end_keeps_model_order_and_reports_missing(tmp_path):
    src, meta, test = write_source(tmp_path)
    out = tmp_path / "out"
    main(["--src", str(src), "--metadata-json", str(meta), "--test-json", str(test),
          "--max-cells", "2", "--out-dir", str(out)])
    data = ad.read_h5ad(out / "parse1m_cd4naive.h5ad")
    # model order preserved, only genes present in the source
    assert list(data.var_names) == ["CFH", "GCLC", "LAP3"]
    assert data.obs.cytokine.value_counts().to_dict() == {"IL-2": 2, "PBS": 2}
    info = json.loads((out / "prepare_info.json").read_text())
    assert info["n_model_genes_missing"] == 2
    assert info["missing_model_genes"] == ["MISSING1", "MISSING2"]
    derived = json.loads((out / "parse1m_genes_available.json").read_text())
    assert derived["genes"] == ["CFH", "GCLC", "LAP3"]
    assert "not generation" in derived["note"]


def test_counts_are_written_unchanged(tmp_path):
    src, meta, test = write_source(tmp_path)
    out = tmp_path / "out"
    main(["--src", str(src), "--metadata-json", str(meta), "--test-json", str(test),
          "--max-cells", "2", "--out-dir", str(out)])
    prepared = ad.read_h5ad(out / "parse1m_cd4naive.h5ad")
    source = ad.read_h5ad(src)
    assert prepared.X.dtype == np.float32
    np.testing.assert_array_equal(prepared.X.data, np.round(prepared.X.data))


def test_fractional_counts_are_rejected():
    with pytest.raises(ValueError, match="integer counts"):
        check_integer_counts(sparse.csr_matrix(np.array([[1.5, 2.0]])))
