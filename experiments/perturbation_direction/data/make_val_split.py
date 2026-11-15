"""Carve a validation set out of the held-out perturbations, before any hyperparameter choice.

Selecting guidance/EMA on the same perturbations used for the headline would tune on the test
set. This splits the panel once, deterministically, stratified by reference effect strength so the
validation set is not dominated by strong perturbations.

The split is a protocol artifact: commit it, and use it for selection only. The report's test
numbers should come from the complement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Only scorable perturbations form strata: a perturbation with <5 reference DEGs carries no
# direction signal (evaluate.py drops it), so spending validation budget on it is wasted.
# Those perturbations stay in the test set, where they enter the retrieval panel but not the
# scored medians.
BINS = ((5, 20), (20, 100), (100, 500), (500, np.inf))


def quota(n_val: int, sizes: list[int]) -> list[int]:
    """Largest-remainder allocation with at least one member per non-empty stratum."""
    total = sum(sizes)
    if n_val >= total:
        return list(sizes)
    if n_val < sum(1 for s in sizes if s > 0):
        raise ValueError(f"cannot give every stratum a member with n_val={n_val}")
    ideal = [n_val * s / total for s in sizes]
    base = [int(np.floor(x)) for x in ideal]
    for i, size in enumerate(sizes):
        if size > 0 and base[i] == 0:
            base[i] = 1
    while sum(base) > n_val:  # trim the largest allocations first
        i = int(np.argmax([b if sizes[j] > 0 else -1 for j, b in enumerate(base)]))
        base[i] -= 1
    remainders = sorted(
        ((ideal[i] - base[i], i) for i, size in enumerate(sizes) if size > 0 and base[i] < size),
        reverse=True,
    )
    for _, i in remainders:
        if sum(base) >= n_val:
            break
        base[i] += 1
    return base


def build_split(deg: pd.DataFrame, n_val: int, seed: int) -> tuple[list[str], list[str], dict]:
    counts = deg.sum(1).astype(int)
    perts = list(counts.index)
    if n_val < 1 or n_val >= len(perts):
        raise ValueError(f"n_val must be in [1, {len(perts) - 1}]")
    rng = np.random.default_rng(seed)
    strata = {
        f"{low}-{high if np.isfinite(high) else '+'}": [p for p in sorted(perts) if low <= counts[p] < high]
        for low, high in BINS
    }
    names = list(strata)
    sizes = [len(strata[name]) for name in names]
    picks = quota(n_val, sizes)
    val: list[str] = []
    for name, k in zip(names, picks, strict=True):
        members = strata[name]
        if k == 0:
            continue
        shuffled = rng.permutation(len(members))
        val.extend(sorted(members[i] for i in shuffled[:k]))
    test = sorted(set(perts) - set(val))
    detail = {
        "strata": {
            name: {"available": len(strata[name]), "validation": int(sum(p in val for p in strata[name]))}
            for name in names
        },
        "deg_counts": {p: int(counts[p]) for p in perts},
    }
    return sorted(val), test, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="json path")
    args = ap.parse_args(argv)

    deg = pd.read_parquet(Path(args.truth_dir) / "deg.parquet")
    val, test, detail = build_split(deg, args.n_val, args.seed)
    if set(val) & set(test) or len(val) + len(test) != len(deg):
        raise AssertionError("split must be disjoint and cover the panel")
    payload = {
        "purpose": "validation-only selection of guidance/EMA; test numbers come from 'test'",
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
        "source": str(Path(args.truth_dir) / "deg.parquet"),
        "binning": ["5-19", "20-99", "100-499", "500+"],
        "note": "perturbations with <5 reference DEGs are never selected for validation",
        "strata": detail["strata"],
        "deg_counts": detail["deg_counts"],
        "val": val,
        "test": test,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"validation {len(val)}, test {len(test)}")
    for name, info in detail["strata"].items():
        print(f"  {name:>10}: {info['validation']}/{info['available']} in validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
