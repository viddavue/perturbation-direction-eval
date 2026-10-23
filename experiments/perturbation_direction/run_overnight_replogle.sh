#!/usr/bin/env bash
# Overnight Replogle run on one node, at the config chosen on the validation split (w2 + EMA).
#
# Stage A: three generation seeds on the 312-perturbation test panel (64 cells) -> seed variance
#          for the main table.
# Stage B: one full-panel run with 128 cells and --save-cells -> the cell-level DE fidelity
#          experiment. More cells than stage A because a DE test at 32-64 cells scored only
#          1/9 perturbations for NMAE in the pilot.
#
# Batch size stays 32 in both stages: seed and batch index fix the per-cell noise.
# Each stage is resumable (existing outputs are skipped) and bounded by `timeout`.
#
# Usage:
#   GPUS=0,1,2 bash run_overnight_replogle.sh          # university node
#   JOB_ROOT=$PWD PYTHON=$PWD/.venv/bin/python GPUS=0 bash run_overnight_replogle.sh   # this PC
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/replogle_runners.env"

SEEDS=${SEEDS:-0 1 2}
COND=${COND:-$JOB_ROOT/outputs/perturbation_direction/replogle/test_conditions.csv}
[ -f "$COND" ] || { echo "missing conditions file: $COND" >&2; exit 2; }
cd "$JOB_ROOT"

gen() {  # gen <out> <n_cells> <seed> <extra flags...>
  local out="$1" cells="$2" seed="$3"; shift 3
  if [ -f "$out" ]; then echo "== skip $(basename "$out") (exists)"; return 0; fi
  echo "== $(basename "$out"): start $(date +%H:%M:%S) on GPUs ${GPUS}"
  local args=(--config "$CONFIG" --ckpt "$CKPT" --metadata-json "$METADATA" --conditions "$COND"
              --n-cells "$cells" --batch-size 32 --library-size 4850
              --guidance 2 --ema --seed "$seed" --gpus "$GPUS" --out "$out" "$@")
  timeout 21600 "$PYTHON" -u "$GENERATE" "${args[@]}"
  local rc=$?
  echo "== $(basename "$out"): end $(date +%H:%M:%S) rc=${rc}"
  return $rc
}

echo "### STAGE A: seeds  $(date +%H:%M:%S)"
for S in $SEEDS; do
  gen "$PRED_DIR/test312_w2_ema_seed${S}.parquet" 64 "$S"
done
echo "STAGE_A_DONE $(date +%H:%M:%S)"

echo "### STAGE B: cell-level $(date +%H:%M:%S)"
gen "$PRED_DIR/test312_w2_ema_cells128.parquet" 128 0 --save-cells
echo "STAGE_B_DONE $(date +%H:%M:%S)"
echo "OVERNIGHT_DONE $(date +%H:%M:%S)"
