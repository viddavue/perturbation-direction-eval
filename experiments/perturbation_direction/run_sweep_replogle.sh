#!/usr/bin/env bash
# Guidance/EMA sweep on the VALIDATION split only, so the test set stays untouched.
#
# Each config generates the control + the 60 validation perturbations. Selection is done
# afterwards from the resulting parquet files; nothing here reads the test perturbations.
#
# Usage: GPUS=0,1,2,3 bash run_sweep_replogle.sh
#
# Resumable: a config whose parquet exists is skipped, and generate.py resumes per condition,
# so this survives a node outage (as happened when student09 went down mid-w2).
set -euo pipefail

B=/system/user/studentwork/fleischa/scldm_generation_20261020
GPUS=${GPUS:-0,1,2,3,4}
PY=/system/apps/studentenv/fleischa/miniconda3/envs/scldm/bin/python
cd "$B"

for W in 1 2 4; do
  for EMA in "" "--ema"; do
    NAME="w${W}${EMA:+_ema}"
    OUT="outputs/perturbation_direction/replogle/sweep/${NAME}.parquet"
    if [ -f "$OUT" ]; then
      echo "== skip ${NAME} (exists)"
      continue
    fi
    echo "== ${NAME}: $(date +%H:%M:%S)"
    "$PY" -u experiments/perturbation_direction/generate.py \
      --config _artifacts/resubmission/replogle_negative_binomial.yaml \
      --ckpt _artifacts/resubmission/replogle_negative_binomial.ckpt \
      --metadata-json metadata/replogle_train.json \
      --conditions outputs/perturbation_direction/replogle/val_conditions.csv \
      --n-cells 64 --batch-size 32 --library-size 4850 \
      --guidance "$W" $EMA --seed 0 --gpus "$GPUS" \
      --out "$OUT" || { echo "FAILED ${NAME}"; exit 1; }
  done
done
echo "SWEEP_DONE $(date +%H:%M:%S)"
