#!/usr/bin/env bash
# Final test runs at the config chosen on the validation split (w2 + EMA), 3 seeds.
#
# Numerics are fixed by seed and batch index, so GPU COUNT may change freely between runs but
# BATCH SIZE may not: generate.py seeds noise per batch index b, so 64 cells at batch 32 is two
# blocks (b=0,1) while batch 16 is four. Changing it would silently change every cell's noise.
#
# A per-run timeout bounds a wedged worker so it cannot sit on a node indefinitely.
#
# Usage:
#   GPUS=0,1,2 bash run_seeds_replogle.sh                      # university node
#   JOB_ROOT=$PWD PYTHON=$PWD/.venv/bin/python GPUS=0 bash run_seeds_replogle.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/replogle_runners.env"

SEEDS=$(normalize_seeds "${SEEDS:-0,1,2}")
COND=${COND:-outputs/perturbation_direction/replogle/test_conditions.csv}
cd "$JOB_ROOT"
[ -f "$COND" ] || { echo "missing conditions file: $JOB_ROOT/$COND" >&2; exit 2; }

for S in $SEEDS; do
  OUT="$PRED_DIR/test312_w2_ema_seed${S}.parquet"
  if [ -f "$OUT" ]; then
    echo "== skip seed ${S} (exists)"
    continue
  fi
  echo "== seed ${S}: $(date +%H:%M:%S) on GPUs ${GPUS}"
  ARGS=(--config "$CONFIG" --ckpt "$CKPT" --metadata-json "$METADATA" --conditions "$COND"
        --n-cells 64 --batch-size 32 --library-size 4850
        --guidance 2 --ema --seed "$S")
  run_generation "$OUT" "${ARGS[@]}" || { echo "FAILED seed ${S}"; exit 1; }
done
echo "SEEDS_DONE $(date +%H:%M:%S)"
