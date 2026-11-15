"""A reference on fewer genes than the predictions must not silently zero every residual.

Regression: when the reference covered a restricted gene set, `remove_shared` reindexed the
predictions to NaN columns, its NaN-aware arithmetic returned all-zero residuals, and evaluate.py's
"pure shared response" guard dropped every method from the specific report without a word.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.perturbation_direction.evaluation.metrics import remove_shared


def test_projection_with_a_narrower_reference_is_not_silently_zero():
    genes_small = ["a", "b", "c"]
    genes_full = ["a", "b", "c", "d", "e"]
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, len(genes_small)), index=genes_small)
    delta = pd.DataFrame(rng.normal(0, 1, (2, len(genes_full))), index=["p1", "p2"], columns=genes_full)

    # the naive path: project without restricting, as evaluate.py did before the fix
    naive, _ = remove_shared(delta, ref)
    assert np.allclose(np.nan_to_num(naive.to_numpy()), 0), "this is the bug the fix guards against"

    # the fixed path: restrict to the reference's genes first
    shared = [g for g in ref.index if g in delta.columns]
    resid, beta = remove_shared(delta[shared], ref)
    assert not np.allclose(resid.to_numpy(), 0)
    assert np.isfinite(resid.to_numpy()).all()
    assert np.isfinite(beta.to_numpy()).all()
    assert list(resid.columns) == genes_small


def test_pure_shared_response_is_still_detected_after_restriction():
    ref = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    delta = pd.DataFrame([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]], index=["p1", "p2"], columns=["a", "b", "c"])
    resid, beta = remove_shared(delta, ref)
    assert np.allclose(resid.to_numpy(), 0)
    assert beta.round(6).tolist() == [2.0, 1.0]
