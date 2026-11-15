"""Fail-fast audit of data, metadata, size-factor tables and checkpoint before a perturbation evaluation.

Reads only obs/var and a sample of the expression matrix, so it is cheap even for large files.
Prints a report and exits non-zero if a blocking problem is found.

Example (Replogle):
    python experiments/perturbation_direction/data/audit.py \
        --train-h5ad .../train_asSTATE_hvg.h5ad --test-h5ad .../test_asSTATE_hvg.h5ad \
        --train-json metadata/replogle_train.json --test-json metadata/replogle_test.json \
        --pert-key gene --control non-targeting --context-key cell_line --target-context hepg2 \
        --size-factor-pkl _artifacts/resubmission/replogle_log_size_factor_munew.pkl \
        --ckpt _artifacts/resubmission/replogle_negative_binomial.ckpt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import anndata as ad
import h5py
import numpy as np
import torch

PROBLEMS: list[str] = []


def check(ok: bool, msg: str, blocking: bool = True):
    tag = "OK  " if ok else ("FAIL" if blocking else "WARN")
    print(f"[{tag}] {msg}")
    if not ok and blocking:
        PROBLEMS.append(msg)


def matrix_sample(path: str, layer: str, n: int = 200_000) -> np.ndarray:
    with h5py.File(path, "r") as f:
        node = f["X"] if layer == "X" else f["layers"][layer]
        return node["data"][:n] if isinstance(node, h5py.Group) else node[: min(node.shape[0], 200)].ravel()


def audit_file(name: str, path: str, layer: str, meta: dict, train_labels: dict, keys: list[str]):
    print(f"\n== {name}: {path}")
    a = ad.read_h5ad(path, backed="r")
    print(f"   {a.n_obs} cells x {a.n_vars} genes; obs: {list(a.obs.columns)}")
    vals = matrix_sample(path, layer)
    check(
        np.allclose(vals, np.round(vals)) and vals.min() >= 0, f"{layer} holds raw counts (sample max {vals.max():.1f})"
    )
    genes = np.asarray(meta["genes"])
    var = a.var_names.astype(str).to_numpy()
    check(set(genes) <= set(var), f"all {len(genes)} metadata genes present in var_names")
    check(
        len(var) == len(genes) and np.all(var == genes),
        "var_names in the same order as metadata (info only)",
        blocking=False,
    )
    for k in keys:
        check(k in a.obs, f"obs column '{k}' present")
        if k in a.obs:
            unknown = sorted(set(a.obs[k].astype(str)) - set(map(str, train_labels[k])))
            check(not unknown, f"all '{k}' labels in TRAIN vocabulary ({len(unknown)} unknown: {unknown[:5]})")
    obs = a.obs.copy()
    a.file.close()
    return obs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5ad", required=True)
    ap.add_argument("--test-h5ad", required=True)
    ap.add_argument("--layer", default="X")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--pert-key", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--context-key", required=True)
    ap.add_argument("--target-context", required=True)
    ap.add_argument("--size-factor-pkl")
    ap.add_argument("--ckpt")
    ap.add_argument("--min-cells", type=int, default=40)
    args = ap.parse_args(argv)

    train_meta = json.loads(pathlib.Path(args.train_json).read_text())
    test_meta = json.loads(pathlib.Path(args.test_json).read_text())
    keys = [args.context_key, args.pert_key]
    tl = train_meta["labels"]
    check(train_meta["genes"] == test_meta["genes"], "train/test metadata gene lists identical")

    tr = audit_file("train", args.train_h5ad, args.layer, train_meta, tl, keys)
    te = audit_file("test", args.test_h5ad, args.layer, train_meta, tl, keys)

    print(f"\n== split structure (target context: {args.target_context})")
    for o in (tr, te):
        for k in keys:
            o[k] = o[k].astype(str)
    tr_t = tr[tr[args.context_key] == args.target_context]
    te_t = te[te[args.context_key] == args.target_context]
    n_ctrl_train = int((tr_t[args.pert_key] == args.control).sum())
    n_ctrl_test = int((te_t[args.pert_key] == args.control).sum())
    print(f"   control cells in target context: train={n_ctrl_train}, test={n_ctrl_test}")
    check(n_ctrl_train + n_ctrl_test >= 2 * args.min_cells, "enough target-context control cells")

    test_perts = set(te_t[args.pert_key]) - {args.control}
    train_perts_t = set(tr_t[args.pert_key]) - {args.control}
    overlap = test_perts & train_perts_t
    print(f"   test perturbations: {len(test_perts)}; target-context perturbations in train: {len(train_perts_t)}")
    check(not overlap, f"no test perturbation also has target-context cells in train (overlap: {len(overlap)})")
    other = tr[tr[args.context_key] != args.target_context]
    seen_elsewhere = other[other[args.pert_key].isin(test_perts)].groupby(args.pert_key)[args.context_key].nunique()
    print(
        f"   test perturbations seen in other contexts in train: {len(seen_elsewhere)}/{len(test_perts)} "
        f"(contexts per perturbation: {seen_elsewhere.describe()[['min', '50%', 'max']].to_dict()})"
    )
    sizes = te_t[args.pert_key].value_counts().drop(args.control, errors="ignore")
    print(
        f"   test cells per perturbation: min={sizes.min()}, median={int(sizes.median())}, max={sizes.max()}; "
        f"{int((sizes < args.min_cells).sum())} perturbations below {args.min_cells}"
    )
    ctrl_other = other[other[args.pert_key] == args.control][args.context_key].value_counts().to_dict()
    print(f"   control cells in other contexts: {ctrl_other}")

    if args.size_factor_pkl:
        print(f"\n== size factors: {args.size_factor_pkl}")
        sf = pickle.load(open(args.size_factor_pkl, "rb"))
        table = sf.get(f"{args.context_key}_{args.pert_key}") or next(iter(sf.values()))
        need = [f"{args.target_context}_{p}" for p in [args.control, *sorted(test_perts)]]
        miss = [k for k in need if k not in table]
        check(
            not miss,
            f"size-factor stats for all generated conditions ({len(miss)} missing: {miss[:5]})",
            blocking=False,
        )
        held = [k for k in need[1:] if k in table]
        check(
            not held,
            f"size-factor table contains {len(held)} held-out test combinations "
            "(stats likely computed on test data; only matters for count sampling)",
            blocking=False,
        )

    if args.ckpt:
        print(f"\n== checkpoint: {args.ckpt}")
        from scldm._utils import remap_pickle

        sd = torch.load(args.ckpt, map_location="cpu", pickle_module=remap_pickle, weights_only=False)
        print(f"   epoch={sd.get('epoch')}, global_step={sd.get('global_step')}")
        sd = sd.get("state_dict", sd)
        ema = {k[len("ema_model.ema_model.") :]: v for k, v in sd.items() if k.startswith("ema_model.ema_model.")}
        raw = {k[len("diffusion_model.") :]: v for k, v in sd.items() if k.startswith("diffusion_model.")}
        check(bool(ema), f"EMA weights present ({len(ema)} tensors)", blocking=False)
        if ema:
            common = [k for k in raw if k in ema and raw[k].is_floating_point()]
            rel = np.mean([float((ema[k] - raw[k]).norm() / (raw[k].norm() + 1e-12)) for k in common])
            print(
                f"   mean relative difference EMA vs raw diffusion weights: {rel:.4f} "
                "(> 0 means the choice matters; sample() uses raw weights)"
            )

    print("\n" + ("AUDIT PASSED" if not PROBLEMS else f"AUDIT FAILED ({len(PROBLEMS)} blocking problems)"))
    sys.exit(1 if PROBLEMS else 0)


if __name__ == "__main__":
    main()
