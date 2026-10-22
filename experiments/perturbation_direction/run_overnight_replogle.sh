#!/usr/bin/env bash
# Overnight Replogle run on one node, at the config chosen on the validation split (w2 + EMA).
#
# Stage A: three generation seeds on the 312-perturbation test panel (64 cells) -> seed variance
#          for the main table.
# Stage B: one full-panel run with 128 cells and --save-cells -> the cell-level DE fidelity
#          experiment. More cells than stage A because a DE test at 32-64 cells scored only
#          1/9 perturbations for NMAE in the pilot.
#
# Numerics: seed and batch size fix the noise (per batch index), so GPU COUNT may differ between
# runs but BATCH SIZE may not. All runs here use batch 32.
#
# Each stage logs its own exit code and is resumable; a wedged run is bounded by `timeout`.
#
# Usage: GPUS=0,1,2 bash run_overnight_replogle.sh
set -uo pipefail

B=/system/user/studentwork/fleischa/scldm_generation_20261020
PY=/system/apps/studentenv/fleischa/miniconda3/envs/scldm/bin/python
GPUS=${GPUS:-0,1,2}
cd "$B"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=_artifacts/resubmission/replogle_negative_binomial.yaml
CKPT=_artifacts/resubmission/replogle_negative_binomial.ckpt
COND=outputs/perturbation_direction/replogle/test_conditions.csv

gen() {  # gen <out> <n_cells> <seed> <extra flags...>
  local out="$1" cells="$2" seed="$3"; shift 3
  if [ -f "$out" ]; then echo "== skip $(basename "$out") (exists)"; return 0; fi
  echo "== $(basename "$out"): start $(date +%H:%M:%S) on GPUs ${GPUS}"
  timeout 21600 "$PY" -u experiments/perturbation_direction/generate.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --metadata-json metadata/replogle_train.json \
    --conditions "$COND" \
    --n-cells "$cells" --batch-size 32 --library-size 4850 \
    --guidance 2 --ema --seed "$seed" --gpus "$GPUS" \
    --out "$out" "$@"
  local rc=$?
  echo "== $(basename "$out"): end $(date +%H:%M:%S) rc=${rc}"
  return $rc
}

echo "### STAGE A: seeds  $(date +%H:%M:%S)"
for S in 0 1 2; do
  gen "outputs/perturbation_direction/replogle/pred/test312_w2_ema_seed${S}.parquet" 64 "$S"
done
echo "STAGE_A_DONE $(date +%H:%M:%S)"

echo "### STAGE B: cell-level $(date +%H:%M:%S)"
gen "outputs/perturbation_direction/replogle/pred/test312_w2_ema_cells128.parquet" 128 0 --save-cells
echo "STAGE_B_DONE $(date +%H:%M:%S)"
echo "OVERNIGHT_DONE $(date +%H:%M:%S)"
