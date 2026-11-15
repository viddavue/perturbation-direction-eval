#!/usr/bin/env bash
# Guidance/EMA sweep on the VALIDATION split only, so the test set stays untouched.
#
# Each config generates the control + the 60 validation perturbations. Selection is done
# afterwards from the resulting parquet files; nothing here reads the test perturbations.
#
# Resumable: a config whose parquet exists is skipped, and generate.py resumes per condition,
# so this survives a node outage (as happened when student09 went down mid-w2).
#
# Usage:
#   GPUS=0,1,2 bash run_sweep_replogle.sh
#   JOB_ROOT=$PWD PYTHON=$PWD/.venv/bin/python GPUS=0 bash run_sweep_replogle.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/replogle_runners.env"

COND=${COND:-outputs/perturbation_direction/replogle/val_conditions.csv}
SWEEP_DIR=${SWEEP_DIR:-outputs/perturbation_direction/replogle/sweep}
cd "$JOB_ROOT"
[ -f "$COND" ] || { echo "missing validation conditions file: $JOB_ROOT/$COND" >&2; exit 2; }

for W in 1 2 4; do
  for EMA in "" "--ema"; do
    NAME="w${W}${EMA:+_ema}"
    OUT="$SWEEP_DIR/${NAME}.parquet"
    if [ -f "$OUT" ]; then
      echo "== skip ${NAME} (exists)"
      continue
    fi
    echo "== ${NAME}: $(date +%H:%M:%S)"
    ARGS=(--config "$CONFIG" --ckpt "$CKPT" --metadata-json "$METADATA" --conditions "$COND"
          --n-cells 64 --batch-size 32 --library-size 4850
          --guidance "$W" --seed 0)
    if [ -n "$EMA" ]; then
      ARGS+=("$EMA")
    fi
    run_generation "$OUT" "${ARGS[@]}" || { echo "FAILED ${NAME}"; exit 1; }
  done
done
echo "SWEEP_DONE $(date +%H:%M:%S)"
