"""scDEBART's npz -> parquet conversion must align gene ids, symbols and perturbations."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from scdebart_predictions import to_frame  # noqa: E402

ID2ENS = {0: "ENSG_P0", 1: "ENSG_P1", 10: "ENSG_G1", 11: "ENSG_G2", 12: "ENSG_NOSYMBOL"}
ENS2SYM = {"ENSG_G1": "GENEA", "ENSG_G2": "GENEB"}


def test_maps_ids_to_symbols_and_keys_rows_by_perturbation():
    pred = {0: [np.array([1.5, -2.0], dtype=np.float16)]}
    genes = {0: [np.array([10, 11])]}
    frame = to_frame(pred, genes, ID2ENS, ENS2SYM)
    assert list(frame.index) == ["ENSG_P0"]
    assert sorted(frame.columns) == ["GENEA", "GENEB"]
    assert frame.loc["ENSG_P0", "GENEA"] == 1.5
    assert frame.loc["ENSG_P0", "GENEB"] == -2.0


def test_gene_without_symbol_is_dropped():
    pred = {0: [np.array([1.0, 5.0])]}
    genes = {0: [np.array([10, 12])]}  # 12 has no symbol
    frame = to_frame(pred, genes, ID2ENS, ENS2SYM)
    assert list(frame.columns) == ["GENEA"]


def test_multiple_perturbations_share_the_gene_index():
    pred = {0: [np.array([1.0, 2.0])], 1: [np.array([3.0, 4.0])]}
    genes = {0: [np.array([10, 11])], 1: [np.array([10, 11])]}
    frame = to_frame(pred, genes, ID2ENS, ENS2SYM)
    assert len(frame) == 2
    assert frame.loc["ENSG_P1", "GENEB"] == 4.0


def test_unknown_gene_id_is_skipped_not_crashed():
    pred = {0: [np.array([1.0, 2.0])]}
    genes = {0: [np.array([10, 99999])]}
    frame = to_frame(pred, genes, ID2ENS, ENS2SYM)
    assert list(frame.columns) == ["GENEA"]
