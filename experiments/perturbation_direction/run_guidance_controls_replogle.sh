#!/usr/bin/env bash
# Guidance controls on the test-312 cohort, scored identically to the w2+EMA main result.
#
#   w0  : v = u, the unconditional prediction. Identical for every condition, so the predicted
#         effect is exactly zero -- the "predict nothing" floor the metric must place near 0.5.
#         Not the same as w1: guidance weight 1 is the plain conditional prediction.
#   w1  : v = c, the conditional prediction with no classifier-free guidance, so "without
#         guidance" in the usual sense.
#
# Both run on the same conditions, batch size and seed as w2_ema, so the only difference is
# guidance. Batch size must stay 32: noise is seeded per batch index.
#
# Usage: GPUS=1,2,4 bash run_guidance_controls_replogle.sh
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/replogle_runners.env"

COND=${COND:-outputs/perturbation_direction/replogle/test_conditions.csv}
cd "$JOB_ROOT"
[ -f "$COND" ] || { echo "missing conditions file: $JOB_ROOT/$COND" >&2; exit 2; }

run() {  # run <weight> <ema flag>
  local w="$1" ema="$2"
  local name="test312_w${w}"
  [ -n "$ema" ] && name="${name}_ema"
  local out="$PRED_DIR/${name}.parquet"
  if [ -f "$out" ]; then echo "== skip ${name} (exists)"; return 0; fi
  echo "== ${name}: start $(date +%H:%M:%S) on GPUs ${GPUS}"
  local args=(--config "$CONFIG" --ckpt "$CKPT" --metadata-json "$METADATA" --conditions "$COND"
              --n-cells 64 --batch-size 32 --library-size 4850
              --guidance "$w" --seed 0)
  [ -n "$ema" ] && args+=(--ema)
  run_generation "$out" "${args[@]}"
  local rc=$?
  echo "== ${name}: end $(date +%H:%M:%S) rc=${rc}"
  return $rc
}

echo "### guidance controls $(date +%H:%M:%S)"
run 0 "" || { echo "FAILED w0"; exit 1; }
run 1 "" || { echo "FAILED w1"; exit 1; }
echo "GUIDANCE_CONTROLS_DONE $(date +%H:%M:%S)"
