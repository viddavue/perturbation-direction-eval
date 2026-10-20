"""Compare two generation runs as a reproducibility replicate.

The Replogle runs use identical settings and seeds, so a replicate is expected to be identical
unless the code path, checkpoint, or library versions drifted. This reports the size of any
difference instead of asserting equality, so a real drift is visible rather than a failure.

Example:
    python experiments/perturbation_direction/replicate_check.py \
        --a outputs/perturbation_direction/replogle/pred/w1.parquet \
        --b outputs/perturbation_direction/replogle/pred/fresh_w1_cells64_20261020.parquet \
        --out outputs/perturbation_direction/replogle/replicate_w1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if set(a.index) != set(b.index) or set(a.columns) != set(b.columns):
        raise ValueError(
            f"not comparable: {len(set(a.index) ^ set(b.index))} differing conditions, "
            f"{len(set(a.columns) ^ set(b.columns))} differing genes"
        )
    b = b.reindex(index=a.index, columns=a.columns)
    left, right = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
    diff = np.abs(left - right)
    per_condition = pd.DataFrame(
        {"max_abs_diff": diff.max(axis=1), "mean_abs_diff": diff.mean(axis=1)},
        index=a.index,
    )
    # correlation is undefined for a constant row; record it as NaN rather than 0
    with np.errstate(invalid="ignore", divide="ignore"):
        per_condition["pearson"] = [
            np.corrcoef(left[i], right[i])[0, 1] if left[i].std() > 0 and right[i].std() > 0 else np.nan
            for i in range(len(a))
        ]
    summary = {
        "n_conditions": int(len(a)),
        "n_genes": int(a.shape[1]),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "fraction_exactly_equal": float((left == right).mean()),
        "n_conditions_exactly_equal": int((diff.max(axis=1) == 0).sum()),
        "min_pearson": float(np.nanmin(per_condition.pearson)) if per_condition.pearson.notna().any() else None,
    }
    return per_condition, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-6, help="report, but do not fail, above this")
    args = ap.parse_args(argv)

    per_condition, summary = compare(pd.read_parquet(args.a), pd.read_parquet(args.b))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_condition.to_csv(out / "per_condition.csv")
    summary["identical_within_tolerance"] = summary["max_abs_diff"] <= args.tolerance
    summary["a"], summary["b"] = args.a, args.b
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"conditions {summary['n_conditions']}, genes {summary['n_genes']}")
    print(f"max |diff| {summary['max_abs_diff']:.3e}; exactly equal {summary['fraction_exactly_equal']:.4f} of values")
    print(f"identical within {args.tolerance:g}: {summary['identical_within_tolerance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
