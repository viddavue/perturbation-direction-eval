"""Trained direction classifiers, ported from perturbational_effect_pw/src/baseline_model.

Each model predicts P(target gene goes up | perturbation, target gene, cell line). Architectures follow the
original repo (prior, low-rank bilinear, MLP, gradient-boosted trees on embedding interaction features). One
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
from sklearn.ensemble import HistGradientBoostingClassifier


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


class BilinearModel(nn.Module):
    """s(p, t, c) = sum_k U(e_p)_k V(e_t)_k sigmoid(W e_c)_k + b_p + b_t + b_0  (original BilinearModel)."""

    def __init__(self, pert: EmbeddingTable, target: EmbeddingTable, n_cells: int, rank: int = 32, d_cell: int = 32):
        super().__init__()
        self.pert, self.target = pert, target
        self.cell = nn.Embedding(n_cells + 1, d_cell, padding_idx=0)
        self.U = nn.Linear(pert.dim, rank, bias=False)
        self.V = nn.Linear(target.dim, rank, bias=False)
        self.cell_modulation = nn.Linear(d_cell, rank)
        self.pert_bias = nn.Embedding(len(pert.index) + 1, 1, padding_idx=0)
        self.target_bias = nn.Embedding(len(target.index) + 1, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, p, t, c):
        w = torch.sigmoid(self.cell_modulation(self.cell(c)))
        inter = (self.U(self.pert(p)) * self.V(self.target(t)) * w).sum(-1, keepdim=True)
        return (inter + self.pert_bias(p) + self.target_bias(t) + self.global_bias).squeeze(-1)


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


class GBDTModel:
    """HistGradientBoostingClassifier on [e_p, e_t, e_p * e_t, |e_p - e_t|, one-hot context].

    Deviation from the original LightGBMBaseline: the original features had no cell line, so the model could not tell
    the target line's responses from other lines' and averaged over them; a one-hot context is appended.
    """

    def __init__(
        self,
        pert_emb: pd.DataFrame,
        target_emb: pd.DataFrame,
        contexts: list[str],
        max_iter: int = 150,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        sample_size: int | None = 200_000,
        seed: int = 42,
    ):
        self.pert_emb, self.target_emb, self.sample_size, self.seed = pert_emb, target_emb, sample_size, seed
        self.contexts = pd.Index(contexts)
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth, random_state=seed
        )

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        def table(emb, keys):
            full = np.vstack([np.zeros((1, emb.shape[1]), np.float32), emb.to_numpy(np.float32)])
            return full[emb.index.get_indexer(keys) + 1]  # unknown -> zero vector

        ep, et = table(self.pert_emb, df["pert"]), table(self.target_emb, df["target"])
        onehot = np.eye(len(self.contexts), dtype=np.float32)[self.contexts.get_indexer(df["context"])]
        return np.concatenate([ep, et, ep * et, np.abs(ep - et), onehot], axis=1)

    def fit(self, df: pd.DataFrame) -> GBDTModel:
        if self.sample_size is not None and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.seed)
        self.model.fit(self._features(df), df["label"].to_numpy(np.int64))
        return self

    def predict_proba(self, df: pd.DataFrame, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate(
            [
                self.model.predict_proba(self._features(df.iloc[s : s + batch_size]))[:, 1]
                for s in range(0, len(df), batch_size)
            ]
        )


class PriorModel:
    """P(up | target gene) from training labels, Laplace-smoothed (original PriorBaseline)."""

    def fit(self, df: pd.DataFrame) -> PriorModel:
        self.global_prior = float(df["label"].mean())
        g = df.groupby("target")["label"].agg(["sum", "count"])
        self.priors = ((g["sum"] + 1.0) / (g["count"] + 2.0)).to_dict()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return df["target"].map(self.priors).fillna(self.global_prior).to_numpy(np.float32)
