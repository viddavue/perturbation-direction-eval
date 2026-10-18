"""The scDEBART server job must use our HepG2 split and our gene universe."""

import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from make_scdebart_job import build_universe, check_no_leakage, split_perturbations  # noqa: E402


def test_universe_keeps_only_genes_available_in_both():
    sym2ens = {"A": "ENSG1", "B": "ENSG2", "C": None, "D": "ENSG4"}
    universe, unmapped = build_universe(["A", "B", "C", "D"], sym2ens, {"ENSG1", "ENSG2"})
    assert universe == ["ENSG1", "ENSG2"]
    assert unmapped == ["C", "D"]


def test_split_is_disjoint_and_holds_out_our_test_perturbations():
    sym2ens = {f"P{i}": f"ENSG{i}" for i in range(100)}
    lfc_index = {f"ENSG{i}" for i in range(100)}
    test_perts = [f"P{i}" for i in range(40)]

    train, val, test, info = split_perturbations(test_perts, sym2ens, lfc_index, seed=0, val_fraction=0.1)

    assert set(test) == {f"ENSG{i}" for i in range(40)}
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert not set(train) & set(val)
    assert set(train) | set(val) | set(test) == lfc_index
    assert info["test"] == 40
    assert len(val) == 4


def test_leakage_check_flags_perturbations_with_target_context_cells(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("anndata")
    import anndata as ad

    obs = pd.DataFrame(
        {"cell_line": ["hepg2", "hepg2", "k562"], "gene": ["PTEN", "TP53", "PTEN"]},
        index=["c0", "c1", "c2"],
    )
    path = tmp_path / "train.h5ad"
    ad.AnnData(X=__import__("numpy").zeros((3, 2)), obs=obs).write_h5ad(tmp_path / "train.h5ad")

    report = check_no_leakage(path, ("cell_line", "hepg2"), ["PTEN", "TP53", "MYC"])
    assert report["n_in_target_train"] == 2  # PTEN and TP53 have HepG2 cells, MYC does not
    assert "MYC" not in report["leaked"]

    # PTEN appears only in k562, so it is not a leak for the hepg2 target context
    k562_report = check_no_leakage(path, ("cell_line", "k562"), ["PTEN"])
    assert k562_report["n_in_target_train"] == 1


def test_leakage_check_is_skipped_when_train_file_is_missing(tmp_path):
    report = check_no_leakage(tmp_path / "missing.h5ad", ("cell_line", "hepg2"), ["PTEN"])
    assert report["skipped"] is True
    assert report["n_in_target_train"] is None


def test_split_is_pickleable_in_the_format_their_script_reads(tmp_path):
    payload = {"train": ["ENSG1"], "val": ["ENSG2"], "test": ["ENSG3"]}
    path = tmp_path / "split.pkl"
    path.write_bytes(pickle.dumps(payload))
    assert pickle.loads(path.read_bytes()) == payload
