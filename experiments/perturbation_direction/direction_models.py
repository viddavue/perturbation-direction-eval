"""Trained direction classifiers, ported from perturbational_effect_pw/src/baseline_model.

Each model predicts P(target gene goes up | perturbation, target gene, cell line). Architectures follow the
original repo (a per-target-gene prior and an MLP on embedding interaction features). One
structural change: perturbations and target genes get separate embedding tables, because here they are derived
from different data (a perturbation from its effect profile, a target gene from its responses; see
learned_baselines.py). Both tables have the same dimension, so the original feature recipe
[e_p, e_t, e_p * e_t, |e_p - e_t|] is kept unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD


def svd_embeddings(features: pd.DataFrame, dim: int = 128, seed: int = 42) -> pd.DataFrame:
    """Rows of ``features`` -> standardized -> TruncatedSVD -> L2-normalized embeddings (as in the original repo)."""
    X = features.to_numpy(dtype=np.float64)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    n_comp = min(dim, X.shape[0] - 1, X.shape[1] - 1)
    emb = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    if n_comp < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - n_comp)))
    return pd.DataFrame(emb.astype(np.float32), index=features.index)


class EmbeddingTable(nn.Module):
    """Index 0 is <UNK> (zero vector); rows 1.. are initialized from pretrained embeddings and trainable."""

    def __init__(self, embeddings: pd.DataFrame):
        super().__init__()
        self.index = {k: i + 1 for i, k in enumerate(embeddings.index)}
        weights = np.vstack([np.zeros((1, embeddings.shape[1]), np.float32), embeddings.to_numpy(np.float32)])
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights), freeze=False, padding_idx=0)
        self.dim = embeddings.shape[1]

    def lookup(self, keys: pd.Series) -> torch.Tensor:
        return torch.as_tensor(keys.map(self.index).fillna(0).to_numpy(np.int64))

    def forward(self, idx):
        return self.embedding(idx)


class MLPModel(nn.Module):
    """MLP on [e_p, e_t, e_p * e_t, |e_p - e_t|, e_c] (original MLPPerturbationModel)."""

    def __init__(
        self,
        pert: EmbeddingTable,
        target: EmbeddingTable,
        n_cells: int,
        hidden: int = 256,
        dropout: float = 0.2,
        d_cell: int = 32,
    ):
        super().__init__()
        self.pert, self.target = pert, target
        self.cell = nn.Embedding(n_cells + 1, d_cell, padding_idx=0)
        d = pert.dim
        self.net = nn.Sequential(
            nn.Linear(4 * d + d_cell, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p, t, c):
        ep, et = self.pert(p), self.target(t)
        return self.net(torch.cat([ep, et, ep * et, (ep - et).abs(), self.cell(c)], -1)).squeeze(-1)


def train_torch(
    model: nn.Module,
    p,
    t,
    c,
    y,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    log=print,
) -> nn.Module:
    """Binary cross-entropy training on (perturbation, target, cell) index triples."""
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(y)
    data = [x.to(device) for x in (p, t, c, torch.as_tensor(y, dtype=torch.float32))]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = loss_fn(model(data[0][idx], data[1][idx], data[2][idx]), data[3][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        log(f"  epoch {epoch + 1}/{epochs}: train BCE {total / n:.4f}")
    return model.eval()


@torch.no_grad()
def predict_torch(model: nn.Module, p, t, c, batch_size: int = 65536, device: str = "cpu") -> np.ndarray:
    model.to(device).eval()
    out = [
        torch.sigmoid(
            model(p[s : s + batch_size].to(device), t[s : s + batch_size].to(device), c[s : s + batch_size].to(device))
        ).cpu()
        for s in range(0, len(p), batch_size)
    ]
    return torch.cat(out).numpy()


class PriorModel:
    """P(up | target gene) from training labels, Laplace-smoothed (original PriorBaseline)."""

    def fit(self, df: pd.DataFrame) -> PriorModel:
        self.global_prior = float(df["label"].mean())
        g = df.groupby("target")["label"].agg(["sum", "count"])
        self.priors = ((g["sum"] + 1.0) / (g["count"] + 2.0)).to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return df["target"].map(self.priors).fillna(self.global_prior).to_numpy(np.float32)
