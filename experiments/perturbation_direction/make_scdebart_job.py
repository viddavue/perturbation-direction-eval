"""Package a scDEBART fine-tune job that uses *this* repo's Replogle HepG2 split.

Why this exists
---------------
scDEBART's released HepG2 predictions are scored on their own random 80/10/10 split. Of our
372 test perturbations, 241 are training perturbations there, so their predictions are not
comparable to ours. To compare on our split we re-fine-tune their model with their own script,
handing it a split file built here.

What the server needs (all produced by this script except the big HF downloads):
  split.json          Ensembl-ID train/val/test lists covering our 1,784-gene universe
  train_val_test_<seed>.pkl  the same thing in the pickle format their finetune script reads
  hvg_geneids_1784.pkl  passed as --num_HVG 1784 so their script selects our gene universe
  job.sh              the deepspeed invocation, with every path filled in

Downloads (from the HF dataset Jieun-S/scDEBART, see job.sh for the exact commands):
  cellxgene/pretrain_output/best_pretrained_scDEBART_model_weights.pt        ~0.5 GB
  data/BioBERT_gene_embedding_dict.pkl                                       ~62 MB
  data/c2_all_2024_1_membership_ens2pathway.tsv                              ~316 MB
  data/ens2geneid.pkl                                                        ~0.4 MB
  perturb_seq/Nadig_HepG2/DE_results/de_masked_high_zero_logfc_to_zero.h5    ~10 GB
  perturb_seq/Nadig_HepG2/DE_results/hvg_geneids_5000.pkl                    small
  <their repo>                                                                cloned

Example:
    python experiments/perturbation_direction/make_scdebart_job.py \
        --lfc-pkl /tmp/opencode/scdebart_probe/hf/perturb_seq/Nadig_HepG2/DE_results/DE_true_logFC_dataframe_masked_low_nonzero_to_zero_dataframe.pkl \
        --ens2geneid /tmp/opencode/scdebart_probe/hf/data/ens2geneid.pkl \
        --out-dir /tmp/opencode/scdebart_job
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

# Perturbations seen in the HepG2 training split that cannot be our test set. If any of our test
# perturbations has HepG2 training cells, the split is not the one our scLDM researchers used.
TRAIN_JSON = "metadata/replogle_train.json"
TEST_JSON = "metadata/replogle_test.json"


def symbol_map_from_biomart(symbols: list[str]) -> dict[str, str]:
    """Fallback: query Ensembl through pybiomart (needs network)."""
    import pybiomart as pbm

    dataset = pbm.Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
    df = dataset.query(attributes=["external_gene_name", "ensembl_gene_id"])
    df.columns = ["symbol", "ensembl"]
    df = df.dropna(subset=["symbol"]).drop_duplicates("symbol")
    return dict(zip(df["symbol"], df["ensembl"], strict=False))


def build_universe(symbols: list[str], sym2ens: dict[str, str], available: set[str]) -> tuple[list[str], list[str]]:
    """Return (ensembl universe, unmapped or unavailable symbols), restricted to ``available``."""
    unmapped = []
    universe = set()
    for symbol in symbols:
        ens = sym2ens.get(symbol)
        if ens is None or ens not in available:
            unmapped.append(symbol)
        else:
            universe.add(ens)
    return sorted(universe), unmapped


def split_perturbations(
    test_perts: list[str],
    sym2ens: dict[str, str],
    lfc_index: set[str],
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[list[str], list[str], list[str], dict]:
    """Split our test perturbations into test / val / train, all disjoint.

    ``train`` and ``val`` are *other* perturbations of the same dataset, never our test ones, so a
    held-out test perturbation is not a training perturbation of the model being compared.
    """
    import numpy as np

    mapped = [sym2ens[p] for p in test_perts if p in sym2ens and sym2ens[p] in lfc_index]
    test = sorted(set(mapped))
    remaining = sorted(lfc_index - set(test))
    rng = np.random.default_rng(seed)
    remaining = list(rng.permutation(remaining))
    n_val = int(round(val_fraction * len(test)))
    val = sorted(remaining[:n_val])
    train = sorted(remaining[n_val:])
    info = {
        "test": len(test),
        "val": len(val),
        "train": len(train),
        "unmapped_test_perts": sorted(set(test_perts) - {p for p in test_perts if p in sym2ens}),
        "test_perts_not_in_de_file": sorted(p for p in test_perts if p in sym2ens and sym2ens[p] not in lfc_index),
    }
    return train, val, test, info


def check_no_leakage(train_h5ad: Path, context: tuple[str, str], test_perts: list[str]) -> dict:
    """Test perturbations that carry target-context training cells would make the comparison invalid.

    The metadata json's ``gene`` vocabulary lists every perturbation in the file, not the ones seen
    in the target context, so the check has to read ``obs`` from the h5ad itself.
    """
    if not train_h5ad.exists():
        return {"n_test_perts": len(test_perts), "n_in_target_train": None, "leaked": [], "skipped": True}
    import h5py
    from anndata.io import read_elem

    with h5py.File(train_h5ad, "r") as f:
        obs = read_elem(f["obs"])
    key, value = context
    obs = obs[obs[key].astype(str) == value]
    train_perts = set(obs["gene"].astype(str))
    overlap = sorted(set(test_perts) & train_perts)
    return {"n_test_perts": len(test_perts), "n_in_target_train": len(overlap), "leaked": overlap[:20]}


def write_job_script(
    out: Path,
    seed: int,
    n_genes: int,
    pretrain_dir: str,
    de_dir: str,
    repo_dir: str,
    split_dir: Path,
) -> Path:
    """Write job.sh with the exact command their README documents, for our split and gene set."""
    script = f"""#!/usr/bin/env bash
# scDEBART fine-tune on this repo's Replogle HepG2 split. Generated by make_scdebart_job.py.
#
# One-time downloads: run fetch_scdebart_data.sh (handles a missing huggingface_hub CLI).
#
# hvg_geneids_5000.pkl is NOT published for Nadig_HepG2 (only norman ships one), so their script
# cannot be run as-is. Copy the gene list produced here next to the DE file:
#   cp hvg_geneids_{n_genes}.pkl {de_dir}/hvg_geneids_{n_genes}.pkl
set -euo pipefail

cd {repo_dir}

deepspeed --num_gpus $NGPUS Finetune_perturbseq/finetune_perturbseq.py \\
  --dataset Nadig_HepG2_oursplit \\
  --de_result_dir {de_dir} \\
  --pretrained_model_dir {pretrain_dir} \\
  --outputdir ./perturbseq/Nadig_HepG2_oursplit/finetuned_model/seed_{seed} \\
  --deepspeed_config Finetune_perturbseq/deepspeed_config_fortuning.json \\
  --train_val_test_dict_path {split_dir / f"train_val_test_{seed}.pkl"} \\
  --num_HVG {n_genes} \\
  --pert_type INH
"""
    path = out / "job.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lfc-pkl", required=True, help="scDEBART DE_true_logFC...pkl; its columns/rows are the DE gene/pert universe")
    ap.add_argument("--ens2geneid", required=True, help="scDEBART data/ens2geneid.pkl")
    ap.add_argument("--symbol-map", default=None, help="optional pickled symbol->ensembl dict; else queried from Ensembl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--scdebart-repo", default="./scDEBART", help="where job.sh cd's before running deepspeed")
    ap.add_argument("--pretrained-dir", default="./cellxgene/pretrain_output", help="dir holding the pretrained weights")
    ap.add_argument("--de-dir", default="./perturbseq/Nadig_HepG2/DE_results", help="dir holding the 10 GB DE h5")
    ap.add_argument("--train-h5ad", default="_artifacts/datasets/replogle_state/replogle_train.h5ad")
    ap.add_argument("--context", default="cell_line=hepg2", help="target context for the leakage check")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.repo_root)

    test_meta = json.loads((root / TEST_JSON).read_text())
    our_genes = test_meta["genes"]
    our_perts = test_meta["labels"]["gene"]

    lfc = pd.read_pickle(args.lfc_pkl)
    available = set(lfc.columns)
    lfc_index = set(lfc.index)
    ens2idx = pd.read_pickle(args.ens2geneid)
    if not all(isinstance(k, int) for k in list(ens2idx)[:5]):
        raise ValueError("ens2geneid.pkl is expected to map integer id -> Ensembl")
    id2ens = dict(ens2idx)

    if args.symbol_map:
        sym2ens = pickle.loads(Path(args.symbol_map).read_bytes())
    else:
        sym2ens = symbol_map_from_biomart(sorted(set(our_genes) | set(our_perts)))

    universe, unmapped = build_universe(our_genes, sym2ens, available)
    train, val, test, split_info = split_perturbations(our_perts, sym2ens, lfc_index, args.seed)

    known_ids = set(id2ens.values())
    dropped_genes = [g for g in universe if g not in known_ids]
    universe = [g for g in universe if g in known_ids]
    test = [p for p in test if p in known_ids]
    train = [p for p in train if p in known_ids]
    val = [p for p in val if p in known_ids]
    split_info["genes_outside_scdebart_vocab"] = dropped_genes
    split_info["test_perts_outside_scdebart_vocab"] = sorted(
        p for p in our_perts if sym2ens.get(p) in lfc_index and sym2ens[p] not in known_ids
    )
    split_info["n_test_perts_repo"] = len(our_perts)

    leakage = check_no_leakage(
        root / args.train_h5ad, tuple(args.context.split("=", 1)), our_perts
    )
    if leakage["n_in_target_train"]:
        raise ValueError(
            f"{leakage['n_in_target_train']} test perturbations have training cells in the target context: "
            f"{leakage['leaked'][:5]}"
        )

    (out / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source": {"test": TEST_JSON, "train": TRAIN_JSON},
                "n_genes": len(universe),
                "unmapped_genes": unmapped,
                **split_info,
                "leakage_check": leakage,
            },
            indent=1,
        )
    )
    (out / f"train_val_test_{args.seed}.pkl").write_bytes(pickle.dumps({"train": train, "val": val, "test": test}))
    (out / f"hvg_geneids_{len(universe)}.pkl").write_bytes(pickle.dumps(universe))
    script = write_job_script(
        out,
        seed=args.seed,
        n_genes=len(universe),
        pretrain_dir=args.pretrained_dir,
        de_dir=args.de_dir,
        repo_dir=args.scdebart_repo,
        split_dir=out.resolve(),
    )

    print(f"universe genes : {len(universe)} of {len(our_genes)} ({len(unmapped)} unmapped/absent)")
    print(f"split          : train {len(train)} | val {len(val)} | test {len(test)}")
    print(f"leakage check  : {leakage['n_in_target_train']} of {leakage['n_test_perts']} test perts in target train")
    print(f"wrote          : {sorted(p.name for p in out.iterdir())}")
    print(f"run on server  : NGPUS=<n> bash {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
