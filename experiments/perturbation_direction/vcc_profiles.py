"""Fixed-panel retrieval using cell-eval2's official rank kernel on supplied effect matrices.

This is NOT the official counts-based VCC score: inputs here are existing effect profiles,
which may use mean-log differences or scVI logFC. No synthetic cells or significance calls
are inferred. Every method uses the SAME complete panel and gene set, including weak effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

PIN = "a2630dcbda95fc0b819bf841e56d9561d9c1e1da"


def validate(frame: pd.DataFrame):
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("duplicate perturbation or gene labels")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("non-finite effect values; missing predictions must not be zero-filled")


def align_panel(truth: pd.DataFrame, methods: dict[str, pd.DataFrame]):
    validate(truth)
    perts, genes = set(truth.index), set(truth.columns)
    for frame in methods.values():
        validate(frame)
        perts &= set(frame.index)
        genes &= set(frame.columns)
    if len(perts) < 2 or not genes:
        raise ValueError("need >=2 common perturbations and at least one common gene")
    perts, genes = sorted(perts), sorted(genes)
    return truth.loc[perts, genes], {k: v.loc[perts, genes] for k, v in methods.items()}


def project_shared(frame: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    vector = ref.reindex(frame.columns).to_numpy(dtype=float)
    if not np.isfinite(vector).all() or vector @ vector <= 0:
        raise ValueError("shared reference must be finite, aligned, and nonzero")
    values = frame.to_numpy(dtype=float)
    residual = values - np.outer(values @ vector / (vector @ vector), vector)
    # Floating-point residue of a perfectly shared response has no identifiable direction.
    tiny = np.linalg.norm(residual, axis=1) <= 1e-12 * np.maximum(np.linalg.norm(values, axis=1), 1)
    residual[tiny] = 0
    return pd.DataFrame(residual,
                        index=frame.index, columns=frame.columns)


def retrieval(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Use upstream midrank/n-1 and panel target exclusion, with an explicit zero control.

    Zero is only an algebraic adapter for already-computed effects, NOT a fabricated cell.
    """
    from cell_eval2.config import EvalConfig
    from cell_eval2.distances import pairwise_full
    from cell_eval2.metrics.discrimination import discrimination_score

    cfg = EvalConfig.from_preset("vcc2026").discrimination
    labels = np.asarray(["__effect_origin__", *truth.index], dtype=str)
    def bulk(frame):
        return labels, np.vstack([np.zeros(frame.shape[1]), frame.to_numpy(dtype=float)])
    # An explicit None map means a panel target outside our gene universe excludes nothing.
    mapping = {str(p): str(p) if p in truth.columns else None for p in truth.index}
    scores = discrimination_score(
        pred_bulk=bulk(pred), real_bulk=bulk(truth), genes=np.asarray(truth.columns),
        control="__effect_origin__", distance="cosine", rank_denominator=cfg.rank_denominator,
        tie_policy=cfg.tie_policy, exclusion_scope=cfg.exclusion_scope,
        exclude_target_gene=True, control_source="pred", target_gene_map=mapping,
    )
    values = pd.Series(scores).reindex(truth.index)
    rank = (1 - values) * (len(truth) - 1)
    # Additional retrieval diagnostics, with expected top-k credit for tied matches.
    # Upstream already excludes every panel target; apply the same feature set here.
    columns = truth.columns.difference(truth.index, sort=False)
    distances = pairwise_full(pred[columns].to_numpy(), truth[columns].to_numpy(), "cosine")
    match = np.diag(distances)
    less = (distances < match[:, None]).sum(1)
    equal = (distances == match[:, None]).sum(1)
    return pd.DataFrame({"effect_pds": values, "rank_0based_midrank": rank,
                         "cosine_distance_to_match": match,
                         "top1_tie_adjusted": np.clip(1 - less, 0, equal) / equal,
                         "top5_tie_adjusted": np.clip(5 - less, 0, equal) / equal})


def bootstrap_mean(values, seed=0, n_bootstrap=1000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--method", nargs="+", required=True, help="NAME=effect.parquet (not expression profiles)")
    parser.add_argument("--shared-ref", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = dict(item.split("=", 1) for item in args.method)
    if len(paths) != len(args.method):
        raise ValueError("duplicate method names")
    truth, methods = align_panel(pd.read_parquet(args.truth), {k: pd.read_parquet(v) for k, v in paths.items()})
    # Fix features BEFORE projecting: excluded target coordinates must not influence beta.
    excluded = sorted(set(truth.index) & set(truth.columns))
    genes = truth.columns.difference(excluded, sort=False)
    if not len(genes):
        raise ValueError("panel target exclusion leaves no genes")
    truth, methods = truth[genes], {k: v[genes] for k, v in methods.items()}
    modes = {"full": (truth, methods)}
    if args.shared_ref:
        ref = pd.read_parquet(args.shared_ref).iloc[0]
        modes["specific"] = (project_shared(truth, ref), {k: project_shared(v, ref) for k, v in methods.items()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(truth))
    # A derangement prevents self-matches; use the same shuffle for every method.
    while np.any(permutation == np.arange(len(truth))):
        permutation = rng.permutation(len(truth))
    paired = []
    for mode, (target, predictions) in modes.items():
        method_scores = {}
        for name, pred in predictions.items():
            for shuffled in (False, True):
                query = pred.iloc[permutation].copy() if shuffled else pred
                query.index = pred.index
                result = retrieval(query, target)
                label = name + ("_shuffled" if shuffled else "")
                result.index.name = "perturbation"
                result.to_csv(args.out_dir / f"{mode}_{label}.csv")
                method_scores[label] = result.effect_pds.to_numpy()
                lo, hi = bootstrap_mean(result.effect_pds, args.seed)
                rows.append({"mode": mode, "method": label, "n_panel": len(target),
                             "n_genes": target.shape[1], "mean_effect_pds": result.effect_pds.mean(),
                             "top1_tie_adjusted": result.top1_tie_adjusted.mean(),
                             "top5_tie_adjusted": result.top5_tie_adjusted.mean(),
                             "ci_low": lo, "ci_high": hi})
        for i, left in enumerate(method_scores):
            for right in list(method_scores)[i + 1:]:
                diff = method_scores[left] - method_scores[right]
                lo, hi = bootstrap_mean(diff, args.seed)
                paired.append({"mode": mode, "left": left, "right": right,
                               "mean_difference": diff.mean(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out_dir / "paired_differences.csv", index=False)
    files = {"truth": args.truth, **paths}
    if args.shared_ref:
        files["shared_ref"] = args.shared_ref
    manifest = {"cell_eval2_version": version("cell-eval2"), "expected_source_commit": PIN,
                "metric": "effect-space diagnostic, NOT official VCC counts comparator or anchored score",
                "bootstrap": "resample query perturbation scores, fixed candidate panel; not independent gene CIs",
                "seed": args.seed, "panel": list(truth.index), "genes": list(genes),
                "excluded_panel_targets": excluded,
                "inputs": {k: {"path": str(v), "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest()}
                           for k, v in files.items()}}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
