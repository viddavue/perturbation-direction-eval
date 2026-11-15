"""Build scLDM-style Replogle train/test h5ads from the STATE Replogle-Nadig file.

Source: https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint (replogle.h5ad).
That file stores X as a dense float32 matrix of log1p-normalised expression (643k cells x 6642 genes;
normalisation was done before gene subsetting). Raw counts are recovered per cell as
expm1(x) / s, where s is the smallest non-zero expm1(x) in the row (= one count); every row is checked
to give integers, and the count total must not exceed obs UMI_count. A row whose counts share a common
factor (e.g. all even) is not identifiable this way; such rows would show up as an unusually low
recovered/UMI_count ratio, whose distribution is reported in prepare_info.json.

Output (counts in X, sparse int32, the 2000 genes of metadata/replogle_train.json in that order):
  replogle_test.h5ad   cells of --target-context whose perturbation is in the test json (STATE-style
                       held-out set; for scLDM: hepg2 x 372 genes)
  replogle_train.h5ad  all other cells (incl. target-context controls and its other perturbations)

Memory stays bounded: rows are read in contiguous chunks and parts are concatenated on disk.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.experimental import concat_on_disk
from scipy import sparse


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a], dtype=object)


def read_obs(f: h5py.File, columns: list[str]) -> pd.DataFrame:
    obs = f["obs"]
    out = {}
    for c in columns:
        node = obs[c]
        if isinstance(node, h5py.Group):  # categorical
            out[c] = _dec(node["categories"][:])[node["codes"][:]]
        else:
            vals = node[:]
            out[c] = _dec(vals) if vals.dtype.kind in "SO" else vals
    return pd.DataFrame(out, index=_dec(obs[obs.attrs["_index"]][:]).astype(str))


def counts_from_lognorm(block: np.ndarray, max_divisor: int = 4, tol: float = 1e-2):
    """Recover integer counts row-wise; returns (counts float64, ok mask)."""
    e = np.expm1(block.astype(np.float64))
    e_pos = np.where(e > 0, e, np.inf)
    unit = e_pos.min(axis=1, keepdims=True)
    unit[~np.isfinite(unit)] = 1.0  # all-zero row
    counts = np.zeros_like(e)
    ok = np.zeros(len(e), dtype=bool)
    for d in range(1, max_divisor + 1):  # smallest count in a row may be >1
        todo = ~ok
        if not todo.any():
            break
        c = e[todo] / (unit[todo] / d)
        good = np.abs(c - np.round(c)).max(axis=1) < tol
        idx = np.flatnonzero(todo)[good]
        counts[idx] = np.round(c[good])
        ok[idx] = True
    return counts, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--target-context", default="hepg2")
    ap.add_argument("--context-key", default="cell_line")
    ap.add_argument("--pert-key", default="gene")
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out_dir)
    parts_dir = out / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    train_meta = json.loads(pathlib.Path(args.train_json).read_text())
    test_genes = set(json.loads(pathlib.Path(args.test_json).read_text())["labels"][args.pert_key])
    genes = list(train_meta["genes"])

    with h5py.File(args.src, "r") as f:
        var_names = _dec(f["var"][f["var"].attrs["_index"]][:]).astype(str)
        pos = pd.Index(var_names).get_indexer(genes)
        if (pos < 0).any():
            raise ValueError(f"{int((pos < 0).sum())} model genes missing from source")
        cols = [
            c
            for c in (args.context_key, args.pert_key, "UMI_count", "gem_group", "sgID_AB", "mitopercent")
            if c in f["obs"]
        ]
        obs = read_obs(f, cols)
        for k in (args.context_key, args.pert_key):
            unknown = set(obs[k]) - set(train_meta["labels"][k])
            if unknown:
                raise ValueError(f"labels outside training vocab for {k}: {sorted(unknown)[:5]}")
            obs[k] = pd.Categorical(obs[k], categories=train_meta["labels"][k])
        is_test = (obs[args.context_key].astype(str) == args.target_context).to_numpy() & obs[args.pert_key].astype(
            str
        ).isin(test_genes).to_numpy()
        print(f"{len(obs)} cells: test {is_test.sum()}, train {(~is_test).sum()}")

        X = f["X"]
        n = X.shape[0]
        n_bad, over_umi, parts, ratios = 0, 0, {"train": [], "test": []}, []
        for k, start in enumerate(range(0, n, args.chunk)):
            stop = min(start + args.chunk, n)
            counts, ok = counts_from_lognorm(X[start:stop])
            n_bad += int((~ok).sum())
            if "UMI_count" in obs:
                umi = obs["UMI_count"].to_numpy()[start:stop]
                over_umi += int((counts.sum(1) > umi * 1.001).sum())
                ratios.append(counts.sum(1) / np.maximum(umi, 1))
            sub = sparse.csr_matrix(counts[:, pos].astype(np.int32))
            for split, m in (("test", is_test[start:stop]), ("train", ~is_test[start:stop])):
                if m.any():
                    p = parts_dir / f"{split}_{k:05d}.h5ad"
                    ad.AnnData(X=sub[m], obs=obs.iloc[start:stop][m], var=pd.DataFrame(index=genes)).write_h5ad(p)
                    parts[split].append(str(p))
            print(f"rows {stop}/{n}  non-integer rows so far: {n_bad}  rows above UMI_count: {over_umi}", flush=True)

    if n_bad:
        raise RuntimeError(f"{n_bad} rows could not be converted to integer counts")
    for split in ("test", "train"):
        dst = out / f"replogle_{split}.h5ad"
        concat_on_disk(parts[split], dst, axis=0, join="inner", merge="same")
        print(f"wrote {dst}")
    shutil.rmtree(parts_dir)
    (out / "prepare_info.json").write_text(
        json.dumps(
            {
                "source": args.src,
                "n_test": int(is_test.sum()),
                "n_train": int((~is_test).sum()),
                "rows_above_umi": over_umi,
                "target_context": args.target_context,
                "recovered_over_umi_quantiles": {
                    str(q): float(np.quantile(np.concatenate(ratios), q)) for q in (0.0, 0.001, 0.01, 0.5, 0.99)
                }
                if ratios
                else None,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
