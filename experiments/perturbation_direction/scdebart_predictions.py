"""Convert scDEBART's test-prediction ``.npz`` into the parquet layout ``evaluate.py`` expects.

scDEBART writes, per perturbation, the predicted log2 fold change over *integer* gene ids
(``pred_logfc``, ``true_logfc``, ``true_ctrl_exp``, ``gene_ids``). Everything downstream in this
repo works with gene symbols, so this script maps integer id -> Ensembl -> symbol.

The predictions are log2 fold changes of normalised expression, i.e. the same quantity as
``truth_scvi`` (scVI ``lfc_median``), not the default ``mean log1p(CP10k)`` difference. Score them
against the target they were trained on; the sign is comparable across targets, the magnitude no.

Example:
    python experiments/perturbation_direction/scdebart_predictions.py \
        --npz outputs/perturbation_direction/scdebart_job/predictions/Nadig_HepG2_oursplit_test_gene_perturb_predictions.npz \
        --ens2geneid _artifacts/datasets/replogle_state/../ens2geneid.pkl \
        --out outputs/perturbation_direction/replogle/scdebart_pred.parquet
"""

from __future__ import annotations

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd


def load_npz_dicts(path: str | pathlib.Path) -> dict[str, dict]:
    """Read the four per-perturbation dictionaries from the npz saved by finetune_perturbseq."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k].item() for k in ("pred_logfc", "true_logfc", "true_ctrl_exp", "gene_ids")}


def to_frame(pred: dict, gene_ids: dict, id2ens: dict[int, str], ens2symbol: dict[str, str]) -> pd.DataFrame:
    """pred logFC per perturbation x gene symbol.

    Rows are keyed by the perturbation's Ensembl id (the npz keys are perturbation gene ids), and
    columns by gene symbol. Genes with no symbol are dropped rather than mislabelled.
    """
    pert_ens = {id2ens.get(int(k), str(k)): k for k in pred}
    columns: dict[str, np.ndarray] = {}
    index: list[str] = []
    for pert_ens_id, key in sorted(pert_ens.items()):
        ids = np.asarray(gene_ids[key][0]).ravel()
        vals = np.asarray(pred[key][0]).ravel().astype(np.float64)
        symbols = [ens2symbol.get(id2ens.get(int(g), ""), None) for g in ids]
        keep = [i for i, s in enumerate(symbols) if s is not None]
        if not keep:
            continue
        index.append(pert_ens_id)
        for i in keep:
            columns.setdefault(symbols[i], {})
            columns[symbols[i]][pert_ens_id] = vals[i]
    frame = pd.DataFrame(columns).T
    return frame.reindex(columns=sorted({p for p in index})).T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ens2geneid", required=True, help="scDEBART data/ens2geneid.pkl (int id -> Ensembl)")
    ap.add_argument("--symbol-map", required=True, help="pickled symbol -> Ensembl dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pert-is-symbol", action="store_true", help="keep symbol keys instead of Ensembl")
    args = ap.parse_args(argv)

    id2ens = {int(k): v for k, v in pickle.loads(pathlib.Path(args.ens2geneid).read_bytes()).items()}
    sym2ens = pickle.loads(pathlib.Path(args.symbol_map).read_bytes())
    ens2symbol = {ens: sym for sym, ens in sym2ens.items()}

    d = load_npz_dicts(args.npz)
    frame = to_frame(d["pred_logfc"], d["gene_ids"], id2ens, ens2symbol)
    if args.pert_is_symbol:
        frame.index = [ens2symbol.get(i, i) for i in frame.index]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
    print(f"{frame.shape[0]} perturbations x {frame.shape[1]} genes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
