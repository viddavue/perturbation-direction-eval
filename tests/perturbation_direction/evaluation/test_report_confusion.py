import json

import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.visualization.confusion import (
    confusion_counts,
    main,
    normalize_rows,
    summarize_counts,
)


def test_classes_boundaries_grey_and_missing():
    counts = confusion_counts(
        [-0.11, -0.1, 0.1, 0.11, np.nan],
        [-1, -1, 0, 1, 0.06],
        [True, True, False, True, False],
        [False, False, True, False, False],
    )
    np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="nonfinite predictions"):
        confusion_counts([np.nan], [1], [True], [False])


def test_macro_not_pooled_and_absent_classes():
    counts = np.array([[[9, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 2, 0], [0, 0, 0]]])
    macro, pooled, total, support = summarize_counts(counts)
    assert macro[0, 0] == pytest.approx(0.45)
    assert pooled[0, 0] == pytest.approx(9 / 11)
    np.testing.assert_array_equal(support, [2, 1, 0])
    assert np.isnan(macro[2]).all()
    np.testing.assert_array_equal(total, counts.sum(axis=0))
    np.testing.assert_allclose(normalize_rows([[0, 2, 0]]), [[0, 1, 0]])


def test_cli_reconstructs_prediction_effect_and_excludes_target(tmp_path):
    truth = tmp_path / "truth"
    truth.mkdir()
    genes = ["P", "down", "unch", "up", "grey"]
    delta = pd.DataFrame([[-1, -1, 0, 1, 0.07]], index=["P"], columns=genes)
    delta.to_parquet(truth / "delta_true.parquet")
    pd.DataFrame([[True, True, False, True, False]], index=["P"], columns=genes).to_parquet(truth / "deg.parquet")
    pd.DataFrame([[0.01, 0.01, 0.9, 0.01, 0.2]], index=["P"], columns=genes).to_parquet(truth / "padj.parquet")
    (truth / "info.json").write_text("{}")
    pred = tmp_path / "prediction.parquet"
    pd.DataFrame([[2, 2, 2, 2, 2], [1, 1, 2, 3, 2]], index=["ctrl", "P"], columns=genes).to_parquet(pred)
    entry = lambda p: {"path": str(p), "abs_path": str(p)}
    manifest = {
        "truth_dir": entry(truth),
        "control": "ctrl",
        "exclude_target": True,
        "min_degs": 2,
        "methods": [{"name": "model", "kind": "pred", **entry(pred)}],
        "unchanged": {"max_effect": 0.05, "min_padj": 0.5, "min_unchanged": 1, "tau": 0.1},
    }
    (tmp_path / "methods.json").write_text(json.dumps(manifest))
    panel = tmp_path / "panel.csv"
    pd.DataFrame({"report": ["unchanged"], "perturbation": ["P"], "shared": [True]}).to_csv(panel, index=False)
    cfg = tmp_path / "config.yaml"
    out = tmp_path / "out"
    cfg.write_text(f"eval_dir: {tmp_path}\npanel_csv: {panel}\nmethods: [model]\nout_dir: {out}\n")
    main(["--config", str(cfg)])
    counts = pd.read_csv(out / "per_perturbation_counts.csv")
    assert counts["count"].sum() == 3
    assert counts.loc[counts.observed == counts.predicted, "count"].sum() == 3
    assert (out / "confusion_macro.pdf").exists()
    assert (out / "confusion_pooled.png").exists()
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["panel"] == ["P"]
    assert len(provenance["inputs_sha256"]) == 7
