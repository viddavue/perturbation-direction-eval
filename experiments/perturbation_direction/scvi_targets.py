"""scVI-denoised log2 fold-change target, following scDEBART's ``cal_de_for_perturbseq.py``.

This produces the *observed* effect for scoring, the way scDEBART defines it, so predictions can
be scored on a denoised target as well as on the log1p-difference target.

Faithful to their code (``Preprocess_perturbseq/cal_de_for_perturbseq.py``):

    scvi.model.SCVI.setup_anndata(adata, layer="counts", labels_key="perturb_gene", batch_key="perturb_gene")
    scvi.model.SCVI(adata, n_latent=64)
    model.differential_expression(groupby=..., group1=<pert>, group2=<control>, mode="change",
                                  delta=0.25, all_stats=True, n_samples_overall=1000, batch_size=4096)
    logfc[nonzero_proportion1 < 0.3] = 0     # and likewise for arm 2
    target = lfc_median

Deliberate differences, documented rather than hidden:
  * **one DE seed, not their 50 resampled profiles per perturbation.** Here the target is
    a single estimate, so 1,000 posterior draws from one seed is used.
  * the fit covers all HepG2 cells from the train and test files (their model is per dataset and
    includes every perturbation). Test perturbations appear only as *targets*, never as training
    labels for a compared model.

Output (a drop-in truth directory for ``evaluate.py``):
    delta_true.parquet   perturbation x gene, masked scVI log2 fold change
    deg.parquet          copied from --truth-dir (same DEG mask, so only the target changes)
    diagnostic.parquet   per perturbation x gene: proba_de, detection rates, raw lfc
    info.json            provenance and settings

Example:
    python experiments/perturbation_direction/scvi_targets.py \
        --train-h5ad _artifacts/datasets/replogle_state/replogle_train.h5ad \
        --test-h5ad _artifacts/datasets/replogle_state/replogle_test.h5ad \
        --metadata-json metadata/replogle_train.json --context cell_line=hepg2 \
        --pert-key gene --control non-targeting \
        --truth-dir outputs/perturbation_direction/replogle/truth \
        --out-dir outputs/perturbation_direction/replogle/truth_scvi
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scvi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ground_truth import load_cells, parse_context


def mask_low_detection(logfc: np.ndarray, nz1: np.ndarray, nz2: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Zero out genes detected in fewer than ``threshold`` of cells in either arm (their rule)."""
    low = (nz1 < threshold) | (nz2 < threshold)
    out = np.array(logfc, dtype=np.float64, copy=True)
    out[low] = 0.0
    return out


def assemble_cells(
    train_h5ad: str,
    test_h5ad: str,
    genes: list[str],
    context: dict[str, str],
    pert_key: str,
    control: str,
) -> ad.AnnData:
    """All cells of the target context from both files: perturbed cells from test, plus train cells.

    Controls live in the train file (the Replogle test file has none), and train perturbations are
    included because scDEBART fits one scVI per dataset over every perturbation.
    """
    test = load_cells(test_h5ad, "X", context, {})
    train = load_cells(train_h5ad, "X", context, {})
    missing = sorted(set(genes) - set(train.var_names))
    if missing:
        raise ValueError(f"{len(missing)} metadata genes missing from data, e.g. {missing[:5]}")

    adata = ad.concat([test[:, genes], train[:, genes]], join="inner", index_unique="-")
    adata.obs[pert_key] = adata.obs[pert_key].astype(str)
    adata.layers["counts"] = adata.X.copy()
    if (adata.obs[pert_key] == control).sum() == 0:
        raise ValueError(f"no '{control}' cells in the target context")
    return adata


def fit_scvi(adata: ad.AnnData, pert_key: str, n_latent: int, max_epochs: int, seed: int):
    """Fit their scVI configuration: perturbation as both label and batch key."""
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, layer="counts", labels_key=pert_key, batch_key=pert_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(
        batch_size=7000,
        max_epochs=max_epochs,
        early_stopping=True,
        early_stopping_patience=5,
    )
    return model


def de_for_perturbations(
    model,
    adata: ad.AnnData,
    perts: list[str],
    pert_key: str,
    control: str,
    n_samples: int,
    batch_size: int,
    log_every: int = 50,
) -> pd.DataFrame:
    """One DE call per perturbation vs control; returns a long frame of per-gene statistics."""
    frames = []
    for i, p in enumerate(perts, 1):
        if p == control:
            continue
        df = model.differential_expression(
            groupby=pert_key,
            group1=p,
            group2=control,
            mode="change",
            delta=0.25,
            all_stats=True,
            silent=True,
            adata=None,
            filter_outlier_cells=False,
            batch_correction=False,
            n_samples_overall=n_samples,
            batch_size=batch_size,
        )
        df = df.reindex(adata.var_names)
        df["perturbation"] = p
        frames.append(df.reset_index().rename(columns={"index": "gene"}))
        if log_every and i % log_every == 0:
            print(f"  {i}/{len(perts)} perturbations done ({time.strftime('%H:%M:%S')})", flush=True)
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5ad", required=True)
    ap.add_argument("--test-h5ad", required=True)
    ap.add_argument("--metadata-json", required=True)
    ap.add_argument("--context", nargs="*", default=["cell_line=hepg2"])
    ap.add_argument("--pert-key", default="gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--truth-dir", required=True, help="existing truth dir; its DEG mask is reused")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=1000, help="their n_samples_overall")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--nonzero-threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = parse_context(args.context)
    genes = json.loads(pathlib.Path(args.metadata_json).read_text())["genes"]

    delta_ref = pd.read_parquet(pathlib.Path(args.truth_dir) / "delta_true.parquet")
    perts = list(delta_ref.index)
    deg = pd.read_parquet(pathlib.Path(args.truth_dir) / "deg.parquet")

    print(f"loading {context} cells from train and test ...", flush=True)
    adata = assemble_cells(args.train_h5ad, args.test_h5ad, genes, context, args.pert_key, args.control)
    present = set(adata.obs[args.pert_key])
    missing = [p for p in perts if p not in present]
    perts = [p for p in perts if p in present]
    print(f"{adata.n_obs} cells, {adata.n_vars} genes, {len(perts)} target perturbations "
          f"({len(missing)} absent from cells)", flush=True)
    if missing:
        print(f"  absent: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    print("fitting scVI ...", flush=True)
    t0 = time.time()
    model = fit_scvi(adata, args.pert_key, args.n_latent, args.max_epochs, args.seed)
    print(f"scVI trained in {time.time() - t0:.0f}s", flush=True)
    if args.save_model:
        model.save(out / "scvi_model", overwrite=True, save_anndata=False)

    print("computing differential expression ...", flush=True)
    long = de_for_perturbations(
        model, adata, perts, args.pert_key, args.control, args.n_samples, args.batch_size
    )
    lfc = long.pivot(index="perturbation", columns="gene", values="lfc_median")
    nz1 = long.pivot(index="perturbation", columns="gene", values="non_zeros_proportion1")
    nz2 = long.pivot(index="perturbation", columns="gene", values="non_zeros_proportion2")
    proba = long.pivot(index="perturbation", columns="gene", values="proba_de")
    lfc, nz1, nz2 = (f.reindex(index=perts, columns=genes) for f in (lfc, nz1, nz2))

    masked = mask_low_detection(lfc.to_numpy(), nz1.to_numpy(), nz2.to_numpy(), args.nonzero_threshold)
    target = pd.DataFrame(masked, index=perts, columns=genes)
    target.to_parquet(out / "delta_true.parquet")
    # Keep unmasked estimates and detection independently: replacing low-detection effects
    # with zero conflates denoising with filtering and cannot support a controlled ablation.
    lfc.to_parquet(out / "delta_unmasked.parquet")
    nz1.to_parquet(out / "detection_perturbed.parquet")
    nz2.to_parquet(out / "detection_control.parquet")
    deg.to_parquet(out / "deg.parquet")
    proba.reindex(index=perts, columns=genes).to_parquet(out / "diagnostic_proba_de.parquet")

    info = {
        "target": "scvi_log2_fold_change",
        "context": context,
        "pert_key": args.pert_key,
        "control": args.control,
        "n_cells_fit": int(adata.n_obs),
        "n_genes": len(genes),
        "n_perturbations": len(perts),
        "perturbations_absent_from_cells": missing,
        "n_latent": args.n_latent,
        "max_epochs": args.max_epochs,
        "de": {"mode": "change", "delta": 0.25, "n_samples_overall": args.n_samples, "seeds": 1},
        "nonzero_threshold": args.nonzero_threshold,
        "seed": args.seed,
        "deviation_from_scdebart": "single DE seed instead of their 50; fit covers train+test cells",
        "masked_fraction": float(
            ((nz1.to_numpy() < args.nonzero_threshold) | (nz2.to_numpy() < args.nonzero_threshold)).mean()
        ),
        "ceiling_note": "no split-half ceiling: would need two more scVI fits on cell halves",
    }
    (out / "info.json").write_text(json.dumps(info, indent=1))

    print(f"masked {100 * info['masked_fraction']:.1f}% of gene/perturbation entries to zero")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
