import sys
from pathlib import Path

import pytest
import torch
from experiments.perturbation_direction.generation.generate import GuidedVelocity, cell_path, save_count_cells

from scldm.nnets import DiT


def tiny_dit(strategy, vocab):
    torch.manual_seed(0)
    dit = DiT(
        n_embed=32,
        n_embed_input=8,
        seq_len=4,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        bias=True,
        norm_layer="layernorm",
        multiple_of=4,
        layernorm_eps=1e-6,
        class_vocab_sizes=vocab,
        cfg_dropout_prob=0.5,
        condition_strategy=strategy,
    ).eval()
    # break the zero-init of adaLN/final layers so outputs depend on the condition
    with torch.no_grad():
        for p in dit.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return dit


@pytest.mark.parametrize(
    ("strategy", "vocab", "w"),
    [
        ("joint", {"cell_line": 3, "gene": 5}, 1.0),
        ("joint", {"cell_line": 3, "gene": 5}, 2.5),
        ("mutually_exclusive", {"clusters": 4}, 1.0),
        ("mutually_exclusive", {"clusters": 4}, 3.0),
    ],
)
@torch.no_grad()
def test_guided_velocity_matches_repo_cfg(strategy, vocab, w):
    dit = tiny_dit(strategy, vocab)
    b = 3
    x = torch.randn(b, 4, 8)
    t = torch.rand(b)
    cond = {k: torch.randint(0, n, (b,)) for k, n in vocab.items()}
    guidance = dict.fromkeys(vocab, w)

    ref = dit.forward_with_cfg(
        torch.cat([x, x]), torch.cat([t, t]), {k: torch.cat([v, v]) for k, v in cond.items()}, cfg_scale=guidance
    )[b:]
    ours = GuidedVelocity(dit, guidance)(x, t, cond)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)
    # sanity: the condition actually matters
    other = {k: (v + 1) % vocab[k] for k, v in cond.items()}
    assert not torch.allclose(ours, GuidedVelocity(dit, guidance)(x, t, other))


def test_cell_counts_are_saved_as_actual_cells(tmp_path):
    import anndata as ad
    import numpy as np

    counts = np.array([[0, 2], [3, 1]], dtype=np.float32)
    path = cell_path(tmp_path / "profiles.parquet", "a/b")
    assert path.parent.name == "cells"
    assert "/a/b" not in str(path)
    save_count_cells(counts, ["g1", "g2"], {"id": "a/b", "gene": "p", "cell_line": "hepg2"}, path)
    result = ad.read_h5ad(path)
    np.testing.assert_array_equal(result.X.toarray(), counts)
    assert result.obs.gene.tolist() == ["p", "p"]
    with pytest.raises(ValueError, match="integer counts"):
        save_count_cells(counts + 0.5, ["g1", "g2"], {"id": "p"}, path)
