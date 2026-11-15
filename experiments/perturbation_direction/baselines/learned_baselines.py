"""Trained direction classifiers (ported from perturbational_effect_pw) as baselines for the HepG2 evaluation.

Training data (no test information):
  * labels: for every training perturbation in every context (incl. the target context's training perturbations),
    the genes it significantly changes (Welch t-test vs that context's controls, BH < alpha, |delta| > min_lfc),
    labelled up (1) / down (0); the perturbed gene itself is excluded. Test perturbations are never labelled in the
    target context (they have no training cells there; asserted).
  * target-gene embeddings: SVD of each gene's responses across all training perturbations and contexts.
  * perturbation embeddings: SVD of each perturbation's effect profile in the OTHER contexts only, so training and
    test perturbations are represented the same way (test perturbations have no target-context profile).

Models: prior and mlp, always both (see direction_models.py). Each writes delta_<model>.parquet with
P(up) - 0.5 for the test perturbations x genes in the target context, which evaluate.py scores like any other
effect prediction (only the sign is meaningful).

Example:
    python experiments/perturbation_direction/baselines/learned_baselines.py --train-h5ad $TRAIN --truth-dir $O/truth \
        --pert-key gene --control non-targeting --context-key cell_line --target-context hepg2 \
        --out-dir $O/learned
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from anndata.io import read_elem

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.baselines.direction_models import (
    EmbeddingTable,
    MLPModel,
    PriorModel,
    predict_torch,
    svd_embeddings,
    train_torch,
)
from experiments.perturbation_direction.evaluation.ground_truth import group_stats, load_cells, lognorm, welch_degs


def context_effects_and_labels(args, context: str, genes: list[str], exclude: set[str], out: pathlib.Path):
    """Effects (pert x gene) and long DEG labels for one context; cached in out/train_{effects,labels}_<ctx>."""
    f_eff, f_lab = out / f"train_effects_{context}.parquet", out / f"train_labels_{context}.parquet"
    if f_eff.exists() and f_lab.exists() and not args.recompute:
        return pd.read_parquet(f_eff), pd.read_parquet(f_lab)
    adata = load_cells(args.train_h5ad, args.layer, {args.context_key: context}, {})[:, genes].copy()
    adata.obs[args.pert_key] = adata.obs[args.pert_key].astype(str)
    sizes = adata.obs[args.pert_key].value_counts()
    if sizes.get(args.control, 0) < args.min_cells:
        return None, None
    perts = sorted(p for p in sizes.index if p != args.control and sizes[p] >= args.min_cells)
    leaked = sorted(set(perts) & exclude)
    if leaked:
        raise RuntimeError(f"{len(leaked)} test perturbations have training cells in {context}: {leaked[:5]}")
    mean, var, n = group_stats(lognorm(adata), args.pert_key)
    delta, padj = welch_degs(mean, var, n, args.control, perts, args.alpha, args.min_lfc)
    deg = (padj < args.alpha) & (delta.abs() > args.min_lfc)
    long = delta.where(deg).stack().rename("delta").reset_index()
    long.columns = ["pert", "target", "delta"]
    long = long[long["pert"] != long["target"]]
    labels = pd.DataFrame(
        {
            "pert": long["pert"],
            "target": long["target"],
            "context": context,
            "label": (long["delta"] > 0).astype(np.int8),
        }
    )
    delta.to_parquet(f_eff)
    labels.to_parquet(f_lab)
    return delta, labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5ad", required=True)
    ap.add_argument("--layer", default="X")
    ap.add_argument("--truth-dir", required=True, help="ground_truth.py output: test perturbations and genes")
    ap.add_argument("--pert-key", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--context-key", required=True)
    ap.add_argument("--target-context", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-lfc", type=float, default=0.1)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="ignore cached training effects/labels")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_parquet(pathlib.Path(args.truth_dir) / "delta_true.parquet")
    test_perts, genes = list(truth.index), list(truth.columns)

    with h5py.File(args.train_h5ad, "r") as f:
        contexts = sorted(set(read_elem(f["obs"])[args.context_key].astype(str)))
    effects, labels = {}, []
    for c in contexts:
        exclude = set(test_perts) if c == args.target_context else set()
        eff, lab = context_effects_and_labels(args, c, genes, exclude, out)
        if eff is None:
            log(f"{c}: too few control cells, skipped")
            continue
        effects[c] = eff
        labels.append(lab)
        log(f"{c}: {len(eff)} training perturbations, {len(lab)} DEG labels ({lab['label'].mean():.2f} up)")
    labels = pd.concat(labels, ignore_index=True)
    others = [c for c in effects if c != args.target_context]

    # target genes: responses across all training perturbations and contexts
    target_feat = pd.concat([effects[c].T.add_prefix(f"{c}|") for c in effects], axis=1).fillna(0.0)
    target_emb = svd_embeddings(target_feat, args.embedding_dim, args.seed)
    # perturbations: effect profiles in the other contexts only (identical representation for train and test)
    all_perts = sorted(set().union(*(effects[c].index for c in others)))
    pert_feat = pd.concat([effects[c].reindex(all_perts).add_prefix(f"{c}|") for c in others], axis=1).fillna(0.0)
    pert_emb = svd_embeddings(pert_feat, args.embedding_dim, args.seed)
    missing = sorted(set(test_perts) - set(pert_emb.index))
    log(
        f"embeddings: {len(target_emb)} target genes, {len(pert_emb)} perturbations "
        f"({len(missing)} test perturbations without an embedding: {missing[:5]})"
    )

    # prediction grid: every test perturbation x gene in the target context
    grid = pd.DataFrame(
        {
            "pert": np.repeat(test_perts, len(genes)),
            "target": np.tile(genes, len(test_perts)),
            "context": args.target_context,
        }
    )
    cells = {c: i + 1 for i, c in enumerate(sorted(labels["context"].unique()))}
    info = {
        "n_labels": len(labels),
        "labels_per_context": labels["context"].value_counts().to_dict(),
        "up_fraction": float(labels["label"].mean()),
        "test_perts_without_embedding": missing,
        "models": {},
    }

    def write(name, proba):
        pred = pd.DataFrame(proba.reshape(len(test_perts), len(genes)) - 0.5, index=test_perts, columns=genes)
        pred.to_parquet(out / f"delta_{name}.parquet")
        info["models"][name] = {"mean_p_up": float(proba.mean())}
        log(f"wrote {out / f'delta_{name}.parquet'}")

    log("training prior")
    write("prior", PriorModel().fit(labels).predict_proba(grid))

    log("training mlp")
    # seed before construction: the MLP's initial weights were drawn from an unseeded generator,
    # so delta_mlp.parquet differed between processes even with the same --seed
    torch.manual_seed(args.seed)
    pt, tt = EmbeddingTable(pert_emb), EmbeddingTable(target_emb)
    model = MLPModel(pt, tt, n_cells=len(cells))

    def idx(df):
        return (
            pt.lookup(df["pert"]),
            tt.lookup(df["target"]),
            torch.as_tensor(df["context"].map(cells).fillna(0).to_numpy(np.int64)),
        )

    train_torch(
        model,
        *idx(labels),
        labels["label"].to_numpy(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        log=log,
    )
    write("mlp", predict_torch(model, *idx(grid), device=args.device))

    (out / "learned_info.json").write_text(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
