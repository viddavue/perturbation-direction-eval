"""Turn an `evaluate.py` output directory into the curated tables the report cites.

`outputs/` is generated and gitignored; this writes small, commit-able tables to `results/` and
records the provenance needed to know which code and inputs produced them.

Example:
    python experiments/perturbation_direction/evaluation/freeze_results.py \
        --eval-dir outputs/perturbation_direction/legacy/replogle/eval_freeze_20261021 \
        --tag replogle_hepgl_hvg2000 \
        --results-dir results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.evaluation.stats import pairwise_tests, summarize

REPORTS = {"": "summary", "_specific": "summary_specific"}


def read_per_perturbation(eval_dir: Path, suffix: str) -> dict[str, pd.DataFrame]:
    folder = eval_dir / f"per_perturbation{suffix}"
    if not folder.is_dir():
        raise FileNotFoundError(f"missing {folder}")
    frames = {}
    for path in sorted(folder.glob("*.csv")):
        frames[path.stem] = pd.read_csv(path, index_col=0)
    if not frames:
        raise ValueError(f"no per-perturbation tables in {folder}")
    return frames


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--metric", default="balanced_acc")
    ap.add_argument("--min-degs", type=int, default=5)
    ap.add_argument("--reference", default="scLDM_w1", help="family for Holm correction; '' for all pairs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = args.reference or None

    written = []
    for suffix, label in REPORTS.items():
        frames = read_per_perturbation(eval_dir, suffix)
        summary = summarize(frames, metric=args.metric, min_degs=args.min_degs, n_boot=args.n_boot, seed=args.seed)
        tests = pairwise_tests(
            frames,
            metric=args.metric,
            min_degs=args.min_degs,
            reference=reference,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        path = out / f"{args.tag}_{label}.csv"
        summary.to_csv(path)
        written.append(path)
        test_path = out / f"{args.tag}_{label}_paired.csv"
        tests.to_csv(test_path, index=False)
        written.append(test_path)

        # the raw published summary is the point estimate the report quotes; keep it alongside
        raw = pd.read_csv(eval_dir / f"summary{'_specific' if suffix else ''}.csv", index_col=0)
        raw_path = out / f"{args.tag}_{label}_raw.csv"
        raw.to_csv(raw_path)
        written.append(raw_path)

    inputs = sorted(p for p in eval_dir.glob("summary*.csv"))
    manifest = {
        "tag": args.tag,
        "eval_dir": str(eval_dir),
        "git_commit": git_commit(),
        "metric": args.metric,
        "min_degs": args.min_degs,
        "reference": reference,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed, "unit": "perturbations", "interval": "percentile, 95%"},
        "test": "two-sided Wilcoxon signed-rank on paired perturbations; Holm within family",
        "inputs": {str(p): sha256(p) for p in inputs},
        "tables": [p.name for p in written],
    }
    (out / f"{args.tag}.provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(written)} tables to {out}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
