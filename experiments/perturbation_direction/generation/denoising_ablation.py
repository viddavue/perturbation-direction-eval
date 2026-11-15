"""Separate observed/scVI effect estimates from detection filtering on a fixed DEG mask.

Observed log2-FC is the ratio of arithmetic mean per-cell CP10k (not pooled counts).
The scVI estimator is posterior median logFC, so even this contrast includes estimator
differences; it is not a proof that denoising alone causes any change. Predictions are
unchanged effect matrices; effect-size errors require compatible log2-FC predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scipy import sparse

from experiments.perturbation_direction.evaluation.ground_truth import group_means, load_cells, parse_context
from experiments.perturbation_direction.evaluation.metrics import evaluate


def observed_effects(adata, perts, key, control, pseudocount=1e-3):
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive (CP10k units)")
    matrix = sparse.csr_matrix(adata.X, dtype=np.float64)
    lib = np.asarray(matrix.sum(1)).ravel()
    if (lib <= 0).any():
        raise ValueError("zero-library cells must be resolved before this experiment")
    normalized = adata.copy()
    normalized.X = matrix.multiply((1e4 / lib)[:, None]).tocsr()
    means = group_means(normalized, key)
    return np.log2((means.loc[perts] + pseudocount) / (means.loc[control] + pseudocount))


def score_variants(methods, observed, denoised, deg, detection, threshold=0.3):
    genes = observed.columns.intersection(denoised.columns).intersection(deg.columns)
    perts = observed.index.intersection(denoised.index).intersection(deg.index)
    for pred in methods.values():
        genes = genes.intersection(pred.columns)
        perts = perts.intersection(pred.index)
    if not len(genes) or not len(perts):
        raise ValueError("empty common panel or gene universe")
    base = deg.loc[perts, genes].astype(bool)
    keep = detection.loc[perts, genes] >= threshold
    results = []
    for estimator, truth in (("observed", observed), ("scvi", denoised)):
        truth = truth.loc[perts, genes]
        if not np.isfinite(truth.to_numpy()).all():
            raise ValueError("non-finite target")
        for filtered, mask in ((False, base), (True, base & keep)):
            for name, pred in methods.items():
                pred = pred.loc[perts, genes]
                if not np.isfinite(pred.to_numpy()).all():
                    raise ValueError(f"non-finite predictions for {name}")
                scores = evaluate(pred, truth, mask, exclude_genes={p: [p] for p in perts})
                scores["estimator"], scores["filtered"], scores["method"] = estimator, filtered, name
                results.append(scores.reset_index())
    return pd.concat(results, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5ad", required=True)
    parser.add_argument("--test-h5ad", required=True)
    parser.add_argument("--context", nargs="*", default=["cell_line=hepg2"])
    parser.add_argument("--pert-key", default="gene")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--scvi-dir", type=Path, required=True)
    parser.add_argument("--method", nargs="+", required=True, help="NAME=delta.parquet")
    parser.add_argument("--pseudocount", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    deg = pd.read_parquet(args.truth_dir / "deg.parquet")
    context = parse_context(args.context)
    pert = load_cells(args.test_h5ad, "X", context, {args.pert_key: list(deg.index)})
    ctrl = load_cells(args.train_h5ad, "X", context, {args.pert_key: [args.control]})
    adata = ad.concat([pert[:, deg.columns], ctrl[:, deg.columns]], index_unique="-")
    obs = observed_effects(adata, list(deg.index), args.pert_key, args.control, args.pseudocount)
    den = pd.read_parquet(args.scvi_dir / "delta_unmasked.parquet")
    nz1 = pd.read_parquet(args.scvi_dir / "detection_perturbed.parquet")
    nz2 = pd.read_parquet(args.scvi_dir / "detection_control.parquet")
    detection = pd.DataFrame(np.minimum(nz1, nz2), index=nz1.index, columns=nz1.columns)
    methods = {k: pd.read_parquet(v) for k, v in (x.split("=", 1) for x in args.method)}
    scores = score_variants(methods, obs, den, deg, detection)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "per_perturbation.csv", index=False)
    # Paired cohort: retain only perturbations scoreable in EVERY arm and method.
    paired = scores.assign(ok=(scores.n_deg >= 5) & scores.balanced_acc.notna()).groupby("perturbation").ok.all()
    selected = scores[scores.perturbation.isin(paired.index[paired])]
    summary = selected.groupby(["estimator", "filtered", "method"]).agg(
        n_paired=("perturbation", "size"),
        balanced_acc=("balanced_acc", "median"),
        spearman_deg=("spearman_deg", "median"),
        n_deg=("n_deg", "median"),
    )
    summary.to_csv(args.out_dir / "summary_paired.csv")
    obs.to_parquet(args.out_dir / "delta_observed_log2fc.parquet")
    info = {
        "pseudocount_cp10k": args.pseudocount,
        "detection_threshold": 0.3,
        "comparison": "fixed predictions; arithmetic-mean observed vs posterior-median scVI effects",
        "mask": "fixed original DEGs, optionally intersected with detection >= 0.3 in both arms",
        "paired": "same perturbations in every arm; >=5 scored DEGs and defined balanced accuracy",
    }
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
