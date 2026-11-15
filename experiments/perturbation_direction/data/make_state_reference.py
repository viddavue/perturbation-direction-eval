"""A per-cell reference table of the STATE Replogle-Nadig file, for the identity gates in verify_merge.py.

Reads the STATE matrix (``replogle.h5ad``: dense float32 ``X = log1p(1e4 * c / L)`` over 6,642 genes,
``L`` the cell's count total over ALL genes of its raw screen file) in row blocks, together with the
integer counts that prepare_replogle_state.py recovered from it (``replogle_train.h5ad`` and
``replogle_test.h5ad``: CSR int32 over the 2,000 model genes, obs index = STATE cell barcode).

One row per STATE cell in ``state_reference.parquet``:

  cell_id          STATE obs index (cell barcode, e.g. ``AAACCCAAGAATAGTC-3``)
  state_row        row in the STATE file
  cell_line, gene, gem_group, UMI_count   STATE obs
  split            ``train`` / ``test``: which STATE-derived count file holds the cell
  lib2000          sum of the cell's counts over the 2,000 model genes
  lib_full_state   the library ``L`` implied by STATE X: ``1e4 * sum_j c_j / sum_j expm1(x_j)`` over the
                   2,000 model genes (NaN when the cell has no count there)
  lib_implied_rel_range  (max - min) / lib_full_state of the per-gene ``1e4 * c_j / expm1(x_j)``: ~1e-7
                   when one ``L`` explains every gene of the cell (float32 rounding), large otherwise
  nz_mismatch      model genes where ``c_j > 0`` and ``x_j > 0`` disagree (must be 0)
  counts_hash      sha1 hex of the int32 little-endian dense 2,000-vector in metadata gene order

``state_reference.json`` records the parquet's md5, the inputs' sizes (and the STATE file's sha256
unless ``--no-sha256``), the git commit, runtime and summary statistics.

The STATE-derived count files must hold the STATE cells in STATE row order (as prepare_replogle_state.py
writes them); this is checked, and every STATE cell must be in exactly one of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.data.audit_raw_screen import quantiles, read_column, read_index, to_jsonable

COLUMNS = (
    "cell_id",
    "state_row",
    "cell_line",
    "gene",
    "gem_group",
    "UMI_count",
    "split",
    "lib2000",
    "lib_full_state",
    "lib_implied_rel_range",
    "nz_mismatch",
    "counts_hash",
)


def counts_hash(counts: np.ndarray) -> list[str]:
    """sha1 hex per row of a dense count matrix, hashed as int32 little-endian (metadata gene order)."""
    c = np.ascontiguousarray(counts)
    if c.dtype != np.dtype("<i4"):
        if not np.array_equal(c, np.rint(c)) or (c.size and (c.min() < 0 or c.max() > np.iinfo(np.int32).max)):
            raise ValueError("counts_hash expects non-negative integer counts that fit int32")
        c = np.ascontiguousarray(c.astype("<i4"))
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in c]


def file_digest(path: str | pathlib.Path, algo: str = "md5", chunk: int = 64 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while buf := fh.read(chunk):
            h.update(buf)
    return h.hexdigest()


def git_commit(repo: pathlib.Path | None = None) -> dict:
    repo = repo or pathlib.Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def csr_rows(node: h5py.Group, start: int, stop: int, n_cols: int) -> np.ndarray:
    """Dense int64 copy of rows [start, stop) of an on-disk CSR group."""
    out = np.zeros((stop - start, n_cols), dtype=np.int64)
    if stop <= start:
        return out
    ip = node["indptr"][start : stop + 1].astype(np.int64)
    data = node["data"][ip[0] : ip[-1]]
    idx = node["indices"][ip[0] : ip[-1]]
    rows = np.repeat(np.arange(stop - start), np.diff(ip))
    out[rows, idx] = data
    return out


def derived_positions(state_ids: np.ndarray, derived_ids: np.ndarray, name: str) -> np.ndarray:
    """Row of each STATE cell in a derived file (-1 if absent); the order must follow STATE row order."""
    if not pd.Index(derived_ids).is_unique:
        raise ValueError(f"{name}: duplicated cell ids")
    pos = pd.Index(derived_ids).get_indexer(state_ids)
    hit = pos[pos >= 0]
    if len(hit) != len(derived_ids):
        raise ValueError(f"{name}: {len(derived_ids) - len(hit)} cells are not in the STATE file")
    if len(hit) and not np.array_equal(hit, np.arange(len(derived_ids))):
        raise ValueError(f"{name}: cells are not in STATE row order; this script relies on that")
    return pos


def build(
    state: str,
    train: str,
    test: str,
    genes: list[str],
    *,
    block: int = 10_000,
    line_key: str = "cell_line",
    pert_key: str = "gene",
    log_every: int = 10,
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    with h5py.File(state, "r") as fs, h5py.File(train, "r") as ftr, h5py.File(test, "r") as fte:
        sids = read_index(fs["obs"])
        svar = read_index(fs["var"])
        spos = pd.Index(svar).get_indexer(genes)
        if (spos < 0).any():
            raise ValueError(f"{int((spos < 0).sum())} model genes missing from {state}")
        obs = pd.DataFrame(
            {c: read_column(fs["obs"], c) for c in (line_key, pert_key, "gem_group", "UMI_count")}, index=sids
        )
        files = {"train": ftr, "test": fte}
        pos = {}
        for split, f in files.items():
            if list(read_index(f["var"])) != list(genes):
                raise ValueError(f"{split} count file: var is not the metadata genes in order")
            ids = read_index(f["obs"])
            pos[split] = derived_positions(sids, ids, split)
            for c in (line_key, pert_key):  # the derived files carry STATE's labels unchanged
                got = read_column(f["obs"], c).astype(str)
                want = obs[c].to_numpy().astype(str)[pos[split] >= 0][np.argsort(pos[split][pos[split] >= 0])]
                if not np.array_equal(got, want):
                    raise ValueError(f"{split} count file: obs[{c!r}] differs from STATE")
        in_train, in_test = pos["train"] >= 0, pos["test"] >= 0
        if (in_train & in_test).any() or (~in_train & ~in_test).any():
            raise ValueError(
                f"{int((in_train & in_test).sum())} STATE cells in both count files, "
                f"{int((~in_train & ~in_test).sum())} in neither"
            )

        n, m = len(sids), len(genes)
        lib2000 = np.zeros(n, dtype=np.int64)
        lib_full = np.full(n, np.nan)
        rel_range = np.full(n, np.nan)
        nz_mismatch = np.zeros(n, dtype=np.int64)
        hashes: list[str] = []
        X = fs["X"]
        if not isinstance(X, h5py.Dataset):
            raise ValueError("STATE X is expected to be a dense dataset")
        for k, start in enumerate(range(0, n, block)):
            stop = min(start + block, n)
            x = X[start:stop][:, spos].astype(np.float64)
            c = np.zeros((stop - start, m), dtype=np.int64)
            for split, f in files.items():
                sel = np.flatnonzero(pos[split][start:stop] >= 0)
                if len(sel):
                    a, b = pos[split][start + sel[0]], pos[split][start + sel[-1]] + 1
                    c[sel] = csr_rows(f["X"], a, b, m)  # contiguous: order checked above
            e = np.expm1(x)
            lib2000[start:stop] = c.sum(1)
            se = e.sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                lf = np.where(se > 0, 1e4 * c.sum(1) / se, np.nan)
                per_gene = np.where((c > 0) & (e > 0), 1e4 * c / np.where(e > 0, e, 1.0), np.nan)
                has = np.isfinite(per_gene).any(1)
                rr = np.full(stop - start, np.nan)
                rr[has] = (np.nanmax(per_gene[has], 1) - np.nanmin(per_gene[has], 1)) / lf[has]
            lib_full[start:stop] = lf
            rel_range[start:stop] = rr
            nz_mismatch[start:stop] = ((c > 0) != (x > 0)).sum(1)
            hashes.extend(counts_hash(c))
            if k % log_every == 0 or stop == n:
                print(f"rows {stop}/{n}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(
        {
            "cell_id": sids,
            "state_row": np.arange(n, dtype=np.int64),
            "cell_line": obs[line_key].astype(str).to_numpy(),
            "gene": obs[pert_key].astype(str).to_numpy(),
            "gem_group": obs["gem_group"].to_numpy(),
            "UMI_count": obs["UMI_count"].to_numpy(),
            "split": np.where(in_test, "test", "train"),
            "lib2000": lib2000,
            "lib_full_state": lib_full,
            "lib_implied_rel_range": rel_range,
            "nz_mismatch": nz_mismatch,
            "counts_hash": hashes,
        }
    )[list(COLUMNS)]
    summary = {
        "rows": int(n),
        "seconds_matrix_pass": round(time.time() - t0, 1),
        "per_line_split": {
            f"{line}/{split}": int(v) for (line, split), v in df.groupby(["cell_line", "split"]).size().items()
        },
        "lib_full_state": quantiles(lib_full),
        "lib_full_state_nan": int(np.isnan(lib_full).sum()),
        "lib_implied_rel_range": quantiles(rel_range),
        "lib_full_over_lib2000": quantiles(lib_full / np.maximum(lib2000, 1)),
        "nz_mismatch_cells": int((nz_mismatch > 0).sum()),
        "nz_mismatch_total": int(nz_mismatch.sum()),
        "duplicate_counts_hash": int(df["counts_hash"].duplicated().sum()),
    }
    return df, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="STATE replogle.h5ad")
    ap.add_argument("--train", required=True, help="STATE-derived replogle_train.h5ad (counts)")
    ap.add_argument("--test", required=True, help="STATE-derived replogle_test.h5ad (counts)")
    ap.add_argument("--metadata-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block", type=int, default=10_000)
    ap.add_argument("--no-sha256", action="store_true", help="skip hashing the 22 GB STATE file")
    args = ap.parse_args(argv)

    t0 = time.time()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    genes = list(json.loads(pathlib.Path(args.metadata_json).read_text())["genes"])
    df, summary = build(args.state, args.train, args.test, genes, block=args.block)
    pq = out / "state_reference.parquet"
    df.to_parquet(pq, index=False)

    inputs = {}
    for name, p in (
        ("state", args.state),
        ("train", args.train),
        ("test", args.test),
        ("metadata", args.metadata_json),
    ):
        entry = {"path": str(p), "size": pathlib.Path(p).stat().st_size}
        if name != "state" or not args.no_sha256:
            t1 = time.time()
            entry["sha256"] = file_digest(p, "sha256")
            entry["sha256_seconds"] = round(time.time() - t1, 1)
        inputs[name] = entry
    info = {
        "output": str(pq),
        "output_md5": file_digest(pq, "md5"),
        "output_size": pq.stat().st_size,
        "columns": list(COLUMNS),
        "inputs": inputs,
        "git": git_commit(),
        "block": args.block,
        **summary,
        "seconds_total": round(time.time() - t0, 1),
    }
    (out / "state_reference.json").write_text(json.dumps(to_jsonable(info), indent=1))
    print(f"wrote {pq} ({summary['rows']} rows) in {info['seconds_total']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
