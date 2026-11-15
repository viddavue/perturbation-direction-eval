"""Choose guidance/EMA from the validation split only, then record the decision.

The sweep generates the control plus the validation perturbations; this scores each config on
those and picks one. The test perturbations are never read here. The chosen config is written to
a json file so the final test run is a single, pre-registered choice rather than a search.

Selection rule (fixed in advance):
  1. maximise median balanced accuracy on validation perturbations with >=5 reference DEGs;
  2. if the top two are within ``--tie-tolerance``, prefer the LOWER guidance weight, then raw
     weights over EMA. Higher guidance costs more sampling time, so a tie should not cost it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.evaluation.metrics import evaluate

CONTROL = "non-targeting"


def delta_from_profiles(profiles: pd.DataFrame, control: str = CONTROL) -> pd.DataFrame:
    if control not in profiles.index:
        raise KeyError(f"control {control!r} missing from profiles")
    return (profiles - profiles.loc[control]).drop(index=[control])


def score_config(pred: pd.DataFrame, truth: pd.DataFrame, deg: pd.DataFrame, perts: list[str]) -> dict:
    rows = pred.index.intersection(perts)
    if len(rows) == 0:
        raise ValueError("no validation perturbations in the prediction")
    keep = [p for p in rows if p in truth.index and p in deg.index]
    scores = evaluate(pred.loc[keep], truth, deg, exclude_genes={p: [p] for p in keep})
    scored = scores[scores["n_deg"] >= 5]
    return {
        "n_scored": int(len(scored)),
        "balanced_acc_median": float(scored["balanced_acc"].median()) if len(scored) else float("nan"),
        "sign_acc_median": float(scored["sign_acc"].median()) if len(scored) else float("nan"),
        "auroc_median": float(scored["auroc"].median()) if len(scored) else float("nan"),
        "spearman_deg_median": float(scored["spearman_deg"].median()) if len(scored) else float("nan"),
    }


def parse_name(stem: str) -> tuple[float, bool]:
    """'w2_ema' -> (2.0, True); 'w1' -> (1.0, False)."""
    parts = stem.split("_")
    weight = float(parts[0].lstrip("w"))
    return weight, "ema" in parts[1:]


def choose(table: pd.DataFrame, tie_tolerance: float) -> str:
    if table.empty or table["balanced_acc_median"].isna().all():
        raise ValueError("no config produced a score")
    best = table["balanced_acc_median"].max()
    near = table[table["balanced_acc_median"] >= best - tie_tolerance]
    return near.sort_values(["guidance", "ema"]).index[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tie-tolerance", type=float, default=0.005)
    args = ap.parse_args(argv)

    truth = pd.read_parquet(Path(args.truth_dir) / "delta_true.parquet")
    deg = pd.read_parquet(Path(args.truth_dir) / "deg.parquet")
    perts = json.loads(Path(args.val_split).read_text())["val"]

    rows = {}
    for path in sorted(Path(args.sweep_dir).glob("*.parquet")):
        weight, ema = parse_name(path.stem)
        pred = delta_from_profiles(pd.read_parquet(path))
        rows[path.stem] = {**score_config(pred, truth, deg, perts), "guidance": weight, "ema": ema}
    table = pd.DataFrame(rows).T
    table["balanced_acc_median"] = table["balanced_acc_median"].astype(float)
    chosen = choose(table, args.tie_tolerance)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chosen": chosen,
        "selection": "validation perturbations only; test perturbations never read",
        "rule": "max median balanced accuracy; ties within tolerance prefer lower guidance, then raw",
        "tie_tolerance": args.tie_tolerance,
        "n_val": len(perts),
        "sweep_dir": str(args.sweep_dir),
        "table": table.reset_index().rename(columns={"index": "config"}).to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(table.sort_values("balanced_acc_median", ascending=False).round(4).to_string())
    print(f"\nchosen: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
