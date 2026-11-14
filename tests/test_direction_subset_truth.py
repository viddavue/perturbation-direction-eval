"""Subsetting a truth directory must keep the split exactly and refuse to guess."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

from subset_truth import main


def make_truth(tmp_path, perts=("p1", "p2", "p3")):
    src = tmp_path / "truth"
    src.mkdir()
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=list(perts), columns=["a", "b"])
    frame.to_parquet(src / "delta_true.parquet")
    (frame.abs() > 2).to_parquet(src / "deg.parquet")
    frame.to_csv(src / "ceiling.csv")
    (src / "info.json").write_text(json.dumps({"n_cells": dict.fromkeys(perts, 10),
                                               "n_degs": dict.fromkeys(perts, 5)}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test": ["p1", "p3"], "val": ["p2"]}))
    return src, split


def test_subset_keeps_exactly_the_requested_rows(tmp_path):
    src, split = make_truth(tmp_path)
    out = tmp_path / "out"
    main(["--truth-dir", str(src), "--split-json", str(split), "--which", "test", "--out-dir", str(out)])
    assert list(pd.read_parquet(out / "delta_true.parquet").index) == ["p1", "p3"]
    assert list(pd.read_csv(out / "ceiling.csv", index_col=0).index) == ["p1", "p3"]
    info = json.loads((out / "info.json").read_text())
    assert info["subset"]["n"] == 2 and set(info["n_cells"]) == {"p1", "p3"}


def test_split_half_files_are_subset_too(tmp_path):
    """Every per-perturbation file of ground_truth.py follows the split, half A's padj included."""
    src, split = make_truth(tmp_path)
    frame = pd.read_parquet(src / "delta_true.parquet")
    names = ("padj", "delta_half_a", "delta_half_b", "deg_half_a", "padj_half_a")
    for i, name in enumerate(names):
        (frame + i).to_parquet(src / f"{name}.parquet")
    out = tmp_path / "out"
    main(["--truth-dir", str(src), "--split-json", str(split), "--which", "test", "--out-dir", str(out)])
    for i, name in enumerate(names):
        pd.testing.assert_frame_equal(pd.read_parquet(out / f"{name}.parquet"), (frame + i).loc[["p1", "p3"]])
    # a truth dir written before padj_half_a existed is still subset (the file is simply not there)
    (src / "padj_half_a.parquet").unlink()
    main(["--truth-dir", str(src), "--split-json", str(split), "--which", "val", "--out-dir", str(tmp_path / "v")])
    assert not (tmp_path / "v" / "padj_half_a.parquet").exists() and (tmp_path / "v" / "deg_half_a.parquet").exists()


def test_val_side_is_selectable(tmp_path):
    src, split = make_truth(tmp_path)
    out = tmp_path / "val"
    main(["--truth-dir", str(src), "--split-json", str(split), "--which", "val", "--out-dir", str(out)])
    assert list(pd.read_parquet(out / "delta_true.parquet").index) == ["p2"]


def test_missing_perturbation_is_an_error_not_a_silent_drop(tmp_path):
    src, split = make_truth(tmp_path)
    split.write_text(json.dumps({"test": ["p1", "p4"], "val": []}))
    with pytest.raises(ValueError, match="absent"):
        main(["--truth-dir", str(src), "--split-json", str(split), "--which", "test",
              "--out-dir", str(tmp_path / "x")])
