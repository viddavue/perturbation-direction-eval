"""Audit the actual scDEBART prediction tokens, not a potentially stale gene-list pickle.

An inhibited gene outside the input sequence receives no -10 indicator. This report flags
those predictions; it does not infer that different baseline inputs must produce identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_predictions(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as data:
        genes = data["gene_ids"].item()
        pred = data["pred_logfc"].item()
        ctrl = data["true_ctrl_exp"].item()
    rows = []
    for key, replicates in genes.items():
        if not (len(replicates) == len(pred[key]) == len(ctrl[key])):
            raise ValueError(f"replicate count mismatch for {key}")
        for i, ids in enumerate(replicates):
            ids = np.asarray(ids).ravel()
            values = np.asarray(pred[key][i]).ravel()
            baseline = np.asarray(ctrl[key][i]).ravel()
            if len(ids) != len(values) or len(ids) != len(baseline):
                raise ValueError(f"gene/value length mismatch for {key}")
            rows.append({
                "perturbation_id": int(key), "replicate": i,
                "n_genes": len(ids), "target_token_present": bool(int(key) in ids),
                "baseline_hash": hashlib.sha256(ids.tobytes() + baseline.tobytes()).hexdigest(),
                "prediction_hash": hashlib.sha256(ids.tobytes() + values.tobytes()).hexdigest(),
                "finite_prediction": bool(np.isfinite(values).all()),
            })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = audit_predictions(args.npz)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "conditioning.csv", index=False)
    summary = {
        "input": str(args.npz), "sha256": hashlib.sha256(args.npz.read_bytes()).hexdigest(),
        "n_perturbations": int(frame.perturbation_id.nunique()),
        "n_profiles": len(frame), "n_target_token_present": int(frame.target_token_present.sum()),
        "n_target_token_absent": int((~frame.target_token_present).sum()),
        "warning": "Absent targets get no perturbation indicator; baseline expression may still differ. "
        "Do not interpret their scores as a clean test of perturbation-conditioned prediction.",
    }
    (args.out_dir / "info.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
