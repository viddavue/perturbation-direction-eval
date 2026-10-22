#!/usr/bin/env bash
# Final test runs at the config chosen on the validation split (w2 + EMA), 3 seeds.
#
# Numerics are fixed by seed and batch index, so GPU COUNT may change freely between runs but
# BATCH SIZE may not: generate.py seeds noise per batch index b, so 64 cells at batch 32 is two
# blocks (b=0,1) while batch 16 is four. Changing it would silently change every cell's noise.
#
# Server safety: fewer GPUs than available (leaves headroom), capped thread pools, and a per-run
# timeout so a wedged worker cannot sit on the node indefinitely.
#
# Usage: GPUS=0,1,2,3 bash run_seeds_replogle.sh
set -euo pipefail

B=/system/user/studentwork/fleischa/scldm_generation_20261020
PY=/system/apps/studentenv/fleischa/miniconda3/envs/scldm/bin/python
GPUS=${GPUS:-0,1,2,3}
SEEDS=${SEEDS:-0 1 2}
COND=$B/outputs/perturbation_direction/replogle/test_conditions.csv
cd "$B"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for S in $SEEDS; do
  OUT="outputs/perturbation_direction/replogle/pred/test312_w2_ema_seed${S}.parquet"
  if [ -f "$OUT" ]; then
    echo "== skip seed ${S} (exists)"
    continue
  fi
  echo "== seed ${S}: $(date +%H:%M:%S) on GPUs ${GPUS}"
  timeout 14400 "$PY" -u experiments/perturbation_direction/generate.py \
    --config _artifacts/resubmission/replogle_negative_binomial.yaml \
    --ckpt _artifacts/resubmission/replogle_negative_binomial.ckpt \
    --metadata-json metadata/replogle_train.json \
    --conditions "$COND" \
    --n-cells 64 --batch-size 32 --library-size 4850 \
    --guidance 2 --ema --seed "$S" --gpus "$GPUS" \
    --out "$OUT" || { echo "FAILED seed ${S}"; exit 1; }
done
echo "SEEDS_DONE $(date +%H:%M:%S)"
