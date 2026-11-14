"""Restrict a truth directory to one side of a committed split.

The validation split makes the 312 test perturbations the primary cohort and the full 372 a
superset sensitivity check. `evaluate.py` takes a truth *directory*, so restricting the cohort
means writing one — this does that without touching the original.

Outputs mirror the input (delta_true, deg, padj and the split-half files if present, ceiling.csv,
info.json) with rows restricted to the requested perturbations, and record which split produced them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROW_FILES = ("delta_true.parquet", "deg.parquet", "padj.parquet", "delta_half_a.parquet",
             "delta_half_b.parquet", "deg_half_a.parquet", "padj_half_a.parquet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--which", default="test", choices=["test", "val"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    src, out = Path(args.truth_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split_json).read_text())
    keep = list(split[args.which])

    written = []
    for name in ROW_FILES:
        path = src / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        missing = sorted(set(keep) - set(frame.index))
        if missing:
            raise ValueError(f"{name}: {len(missing)} requested perturbations absent, e.g. {missing[:5]}")
        frame.loc[keep].to_parquet(out / name)
        written.append(name)

    if "delta_true.parquet" not in written:
        raise FileNotFoundError(f"{src} has no delta_true.parquet")

    if (src / "ceiling.csv").exists():
        pd.read_csv(src / "ceiling.csv", index_col=0).loc[keep].to_csv(out / "ceiling.csv")
    info = json.loads((src / "info.json").read_text())
    info["subset"] = {"which": args.which, "n": len(keep), "split_json": args.split_json,
                      "source": str(src)}
    info["n_cells"] = {p: info["n_cells"][p] for p in keep if p in info.get("n_cells", {})}
    if "n_degs" in info:
        info["n_degs"] = {p: info["n_degs"][p] for p in keep if p in info["n_degs"]}
    (out / "info.json").write_text(json.dumps(info, indent=1))

    print(f"{args.which}: {len(keep)} perturbations -> {out}")
    print("wrote:", ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
