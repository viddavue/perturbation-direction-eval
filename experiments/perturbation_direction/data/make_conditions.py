"""Write a generate.py conditions CSV from a test metadata json (no expression data needed).

Lets generation start before the data is downloaded/prepared; evaluate.py later scores only the
perturbations that ground_truth.py keeps (enough real cells).

Example:
    python experiments/perturbation_direction/data/make_conditions.py --test-json metadata/replogle_test.json \
        --pert-key gene --control non-targeting --context cell_line=hepg2 --out conditions.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--pert-key", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--context", nargs="*", default=[], help="key=value, e.g. cell_line=hepg2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    perts = [
        p for p in json.loads(pathlib.Path(args.test_json).read_text())["labels"][args.pert_key] if p != args.control
    ]
    cond = pd.DataFrame({"id": [args.control, *perts], args.pert_key: [args.control, *perts]})
    for item in args.context:
        k, v = item.split("=", 1)
        cond[k] = v
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cond.to_csv(args.out, index=False)
    print(f"wrote {args.out}: control + {len(perts)} perturbations")


if __name__ == "__main__":
    main()
