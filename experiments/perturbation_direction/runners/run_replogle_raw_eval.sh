#!/usr/bin/env bash
# Raw Replogle rebuild, evaluation stage: the prepared filtered/all files -> the side-by-side main table.
#
# Everything is read from the config (experiments/configs/perturbation_direction/replogle_raw_eval.yaml); extra
# arguments are OmegaConf overrides (e.g. out_dir=/some/where controls=null). Steps, in order (P = prepared_dir,
# O = out_dir, primary version first):
#
#   truth_<v>, subset_{val,test}_<v>  ground_truth.py on P/replogle_test_raw_<v> with the target line's controls
#                                     from P/replogle_train_raw_<v> (all test perturbations), then subset_truth.py
#                                     -> O/truth_<v>_val, O/truth_<v>_test (the split's val / test perturbations)
#   a5_truth, a5_truth_test           GATE. verify_merge.py truth: the primary version's truth (all, and its test
#                                     subset) must equal the frozen STATE-based truth
#   stage_sweep, select_sweep,        GATE. The configured sweep files (staged one by one: the generation dir also
#   select_check                      holds *_sample files) rescored by select_sweep.py on O/truth_<primary>_val;
#                                     side_by_side.py check-selection: it must choose selection.expected
#   tierb_<v>, tierb_test_<v>         verify_merge.py compare: primary vs each other version (a report)
#   baselines_<v>, learned_<v>        baselines.py (generic, transfer, kNN) and learned_baselines.py (prior, MLP),
#                                     refitted on P/replogle_train_raw_<v> (--recompute: no stale caches)
#   eval_<v>                          evaluate.py on O/truth_<v>_test: the preds (scLDM seeds) + the five baselines,
#                                     --exclude-target, --specific-ref O/baselines_<v>/delta_generic.parquet; three
#                                     reports: full effect, perturbation-specific, changed vs unchanged genes
#                                     (evaluate.unchanged_* / min_unchanged; evaluate.direction_only is left out of it)
#   controls_<v>                      optional: the reference + the configured generation controls -> O/eval_controls_<v>
#   side_by_side[, _controls]         side_by_side.py table -> <results_dir>/<results_tag>[_controls]_*
#
# Each step logs to O/logs/<step>.log, stops the run on failure (non-zero exit), and writes O/logs/<step>.done,
# whose first line is "hash=<sha256>". Resume (step_hash.py, as in run_replogle_raw_data.sh): the hash covers the
# resolved config, the step's command line, the sha256 of its script and of every pipeline module the script
# imports, its small input files (metadata JSON, split, frozen selection/summaries) and the size/mtime of its
# large ones (the prepared h5ads, sweep/pred/control parquet, the frozen truth), and the .done contents of the
# steps upstream of it. A step is skipped while its hash is unchanged and reruns when it differs (the log names
# the components that changed; O/logs/<step>.hash.json holds them). The run time that side_by_side.py records
# is masked, so a rerun that starts at another time does not rerun the tables. A step that runs first deletes its
# own marker, the markers of every step downstream of it and DONE_raw_eval. A marker written by an earlier runner
# (a timestamp, no hash) is stale, so that step and its downstream steps rerun. FORCE=1 reruns all.
# Inputs are also pinned run-wide: O/logs/inputs.sha256 holds the resolved config, the sha256 of every input file
# outside O (prepared manifest.json, metadata, split, frozen truth/selection/summaries, sweep/pred/control files)
# and the size/mtime of the prepared h5ads. When a rerun finds them changed (e.g. the data stage was rerun), it
# says what changed and deletes every .done marker, so everything reruns; when that stamp itself is missing while
# markers exist, the markers count as stale and everything reruns too.
# CPU throughout, except learned_baselines.py, which uses CUDA when available (learned.device or DEVICE=cpu|cuda).
#
# Usage (paths in the config are relative to the repository root; the script cds there):
#   bash experiments/perturbation_direction/runners/run_replogle_raw_eval.sh                                 # student03
#   tmux new -s raweval 'bash experiments/perturbation_direction/runners/run_replogle_raw_eval.sh; bash'
#   PYTHON=$PWD/.venv/bin/python bash experiments/perturbation_direction/runners/run_replogle_raw_eval.sh \
#     prepared_dir=... generation_dir=outputs/perturbation_direction/replogle
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_ROOT=${JOB_ROOT:-$(cd "$HERE/../../.." && pwd)}
PYTHON=${PYTHON:-/system/apps/studentenv/fleischa/miniconda3/envs/scldm/bin/python}
CONFIG=${CONFIG:-experiments/configs/perturbation_direction/replogle_raw_eval.yaml}
FORCE=${FORCE:-0}
DEVICE=${DEVICE:-}
E=experiments/perturbation_direction
cd "$JOB_ROOT"
OVERRIDES=("$@")
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # recorded in the provenance of the tables

# resolved config -> shell variables + one tab-separated line per sweep / pred / control file
CFG_DUMP=$("$PYTHON" "$E/evaluation/side_by_side.py" config --config "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
eval "$(grep '^CFG_' <<< "$CFG_DUMP")"
mapfile -t SWEEP < <(grep $'^SWEEP\t' <<< "$CFG_DUMP" || true)
mapfile -t PREDS < <(grep $'^PRED\t' <<< "$CFG_DUMP" || true)
mapfile -t CONTROLS < <(grep $'^CONTROL\t' <<< "$CFG_DUMP" || true)
read -r -a VERSIONS <<< "$CFG_VERSIONS"
read -r -a DIRECTION_ONLY <<< "$CFG_DIRECTION_ONLY"  # methods left out of the changed-vs-unchanged report
# the resolved config as step_hash.py sees it (provenance-only keys such as md5_source left out)
CFG_DIGEST=$("$PYTHON" - "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"} <<'PY'
import sys
from experiments.perturbation_direction.evaluation.side_by_side import load_eval_config
from experiments.perturbation_direction.data.step_hash import config_digest
print(config_digest(load_eval_config(sys.argv[1], sys.argv[2:])))
PY
)
P=$CFG_PREPARED_DIR
O=$CFG_OUT_DIR
PRIMARY=$CFG_PRIMARY
LOG_DIR="$O/logs"
ts() { date +%Y-%m-%dT%H:%M:%S; }
h5() { echo "$P/replogle_$1_raw_$2.h5ad"; }  # h5 <split> <version>

# data dependencies: step -> the steps whose .done enters its hash (a gate writes nothing another step reads)
SIDE_UPSTREAM="" SIDE_CONTROL_UPSTREAM=""
declare -A UPSTREAM=([stage_sweep]="")
for v in "${VERSIONS[@]}"; do
  UPSTREAM[truth_$v]=""
  UPSTREAM[subset_val_$v]="truth_$v"
  UPSTREAM[subset_test_$v]="truth_$v"
done
UPSTREAM[a5_truth]="truth_$PRIMARY"
UPSTREAM[a5_truth_test]="subset_test_$PRIMARY"
UPSTREAM[select_sweep]="stage_sweep subset_val_$PRIMARY"  # rescored on the primary version's val truth
UPSTREAM[select_check]="select_sweep"
for v in "${VERSIONS[@]}"; do
  if [ "$v" != "$PRIMARY" ]; then
    UPSTREAM[tierb_$v]="truth_$PRIMARY truth_$v"
    UPSTREAM[tierb_test_$v]="subset_test_$PRIMARY subset_test_$v"
    SIDE_UPSTREAM+=" tierb_$v tierb_test_$v"
  fi
  UPSTREAM[baselines_$v]="truth_$v"
  UPSTREAM[learned_$v]="truth_$v"
  UPSTREAM[eval_$v]="subset_test_$v baselines_$v learned_$v"
  UPSTREAM[controls_$v]="subset_test_$v baselines_$v"
  SIDE_UPSTREAM+=" eval_$v"
  SIDE_CONTROL_UPSTREAM+=" controls_$v"
done
UPSTREAM[side_by_side]="a5_truth a5_truth_test select_check$SIDE_UPSTREAM"
UPSTREAM[side_by_side_controls]=$SIDE_CONTROL_UPSTREAM

downstream_of() {  # every step whose upstream chain contains $1 (one per line; may repeat)
  local t
  for t in "${!UPSTREAM[@]}"; do
    case " ${UPSTREAM[$t]} " in *" $1 "*) echo "$t"; downstream_of "$t" ;; esac
  done
}

invalidate() {  # $1 is about to run: it, everything downstream of it and DONE_raw_eval stop being done
  local t
  rm -f "$LOG_DIR/$1.done" "$O/DONE_raw_eval"
  for t in $(downstream_of "$1" | sort -u); do
    if [ -f "$LOG_DIR/$t.done" ]; then
      rm -f "$LOG_DIR/$t.done"
      echo "        invalidated $t (downstream of $1)"
    fi
  done
}

# run_step <name> [--code <script> | --file <small file> | --stat <large file> | --mask-argv <value>]... -- <command...>
# log, stop on failure, write a .done marker. --code/--file/--stat/--mask-argv are the hash inputs of step_hash.py
# (--mask-argv: a volatile argv value such as the run timestamp, left out of the hash); upstream markers come
# from UPSTREAM.
run_step() {
  local name="$1"; shift
  local -a hargs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
      --code|--file|--stat|--mask-argv) hargs+=("$1" "$2"); shift 2 ;;
      *) echo "run_step $name: unknown option $1" >&2; exit 2 ;;
    esac
  done
  shift  # --
  local up
  for up in ${UPSTREAM[$name]-}; do hargs+=(--upstream "$LOG_DIR/$up.done"); done
  local marker="$LOG_DIR/$name.done" info="$LOG_DIR/$name.hash.json" next="$LOG_DIR/$name.hash.next.json"
  local hash reason first
  hash=$("$PYTHON" "$E/data/step_hash.py" hash --config-digest "$CFG_DIGEST" --info "$next" ${hargs[@]+"${hargs[@]}"} -- "$@")
  if [ "$FORCE" = 1 ]; then
    reason="FORCE=1"
  elif [ ! -f "$marker" ]; then
    reason="not done"
  else
    first=$(head -n 1 "$marker")
    if [ "$first" = "hash=$hash" ]; then
      rm -f "$next"
      echo "[$(ts)] skip $name (done, hash ${hash:0:12}, $(sed -n 's/^finished=//p' "$marker" | head -n 1))"
      return 0
    elif [[ "$first" != hash=* ]]; then
      reason="marker without a step hash (an earlier runner's timestamp): $first"
    else
      reason="hash changed: $("$PYTHON" "$E/data/step_hash.py" diff "$info" "$next")"
    fi
  fi
  invalidate "$name"
  echo "[$(ts)] start $name ($reason) -> $LOG_DIR/$name.log"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    printf 'hash=%s\nfinished=%s\n' "$hash" "$(ts)" > "$marker"
    mv -f "$next" "$info"
    echo "[$(ts)] ok    $name"
    tail -n 3 "$LOG_DIR/$name.log" | sed 's/^/        /'
  else
    local rc=$?
    echo "[$(ts)] FAIL  $name (exit $rc); last lines of $LOG_DIR/$name.log:" >&2
    tail -n 30 "$LOG_DIR/$name.log" >&2
    exit "$rc"
  fi
}

# --- inputs: everything must exist before the first step (optional controls are skipped instead) -------------
missing=()
need() { [ -e "$1" ] || missing+=("$1"); }
if [ "$CFG_REQUIRE_DATA_DONE" = 1 ]; then need "$P/DONE_raw_data"; fi
need "$P/manifest.json"
for v in "${VERSIONS[@]}"; do need "$(h5 train "$v")"; need "$(h5 test "$v")"; done
need "$CFG_TRAIN_JSON"
need "$CFG_SPLIT_JSON"
need "$CFG_FROZEN_TRUTH/delta_true.parquet"
need "$CFG_FROZEN_TRUTH_TEST/delta_true.parquet"
for f in "$CFG_SEL_FROZEN_JSON" "$CFG_FROZEN_SUMMARY" "$CFG_FROZEN_SUMMARY_SPECIFIC"; do
  if [ -n "$f" ]; then need "$f"; fi
done
for line in "${SWEEP[@]}" "${PREDS[@]}"; do
  IFS=$'\t' read -r _ _ path <<< "$line"
  need "$path"
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "missing inputs (relative to $JOB_ROOT):" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "the data stage must have finished (DONE_raw_data); override paths as key=value (see the config)" >&2
  exit 2
fi
mkdir -p "$LOG_DIR" "$O/verify" "$O/selection"

stamp_inputs() {  # what the steps read from outside out_dir, in a fixed order
  local v f line path
  printf '%s\n' "$CFG_DUMP"
  for v in "${VERSIONS[@]}"; do stat -c 'stat %s %Y %n' "$(h5 train "$v")" "$(h5 test "$v")"; done
  local files=("$P/manifest.json" "$CFG_TRAIN_JSON" "$CFG_SPLIT_JSON" "$CFG_FROZEN_TRUTH"/* "$CFG_FROZEN_TRUTH_TEST"/*)
  for f in "$CFG_SEL_FROZEN_JSON" "$CFG_FROZEN_SUMMARY" "$CFG_FROZEN_SUMMARY_SPECIFIC"; do
    if [ -n "$f" ]; then files+=("$f"); fi
  done
  for line in "${SWEEP[@]}" "${PREDS[@]}" ${CONTROLS[@]+"${CONTROLS[@]}"}; do
    IFS=$'\t' read -r _ _ path <<< "$line"
    files+=("$path")
  done
  for f in "${files[@]}"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "missing $f"; fi
  done
}
STAMP="$LOG_DIR/inputs.sha256"
NEW_STAMP=$(stamp_inputs)
MARKERS=("$LOG_DIR"/*.done)
if [ ! -e "${MARKERS[0]}" ]; then MARKERS=(); fi  # no marker yet: the glob did not match
if [ ! -f "$STAMP" ] && [ "${#MARKERS[@]}" -gt 0 ]; then
  echo "[$(ts)] the inputs stamp $STAMP is missing while ${#MARKERS[@]} step markers exist:" \
    "the markers are stale, every step reruns"
  rm -f ${MARKERS[@]+"${MARKERS[@]}"} "$O/DONE_raw_eval"
elif [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$NEW_STAMP" ]; then
  echo "[$(ts)] inputs changed since the last run of $O; every step reruns. Changed:"
  diff <(cat "$STAMP") <(printf '%s\n' "$NEW_STAMP") | grep '^[<>]' | sed 's/^/        /' || true
  rm -f ${MARKERS[@]+"${MARKERS[@]}"} "$O/DONE_raw_eval"
fi
printf '%s\n' "$NEW_STAMP" > "$STAMP"

# per-step hash inputs: small files by content (--file), large ones by size and mtime (--stat)
PRED_ARGS=()
PRED_STATS=()
REF_ARG=
SWEEP_INPUTS=()
SWEEP_STATS=()
for line in "${PREDS[@]}"; do
  IFS=$'\t' read -r _ name path <<< "$line"
  PRED_ARGS+=("$name=$path")
  PRED_STATS+=(--stat "$path")
  if [ "$name" = "$CFG_REFERENCE" ]; then REF_ARG="$name=$path"; fi
done
CONTROL_ARGS=()
CONTROL_STATS=()
for line in ${CONTROLS[@]+"${CONTROLS[@]}"}; do
  IFS=$'\t' read -r _ name path <<< "$line"
  if [ -f "$path" ]; then
    CONTROL_ARGS+=("$name=$path")
    CONTROL_STATS+=(--stat "$path")
  else
    echo "[$(ts)] note: control $name skipped (missing $path)"
  fi
done
for line in "${SWEEP[@]}"; do
  IFS=$'\t' read -r _ name path <<< "$line"
  SWEEP_INPUTS+=(--input "sweep_$name=$path")
  SWEEP_STATS+=(--stat "$path")
done
FROZEN_TRUTH_STATS=()
FROZEN_TEST_STATS=()
for f in delta_true.parquet padj.parquet deg.parquet info.json; do  # what verify_merge.py truth reads
  FROZEN_TRUTH_STATS+=(--stat "$CFG_FROZEN_TRUTH/$f")
  FROZEN_TEST_STATS+=(--stat "$CFG_FROZEN_TRUTH_TEST/$f")
done
SEL_FILES=()
if [ -n "$CFG_SEL_FROZEN_JSON" ]; then SEL_FILES+=(--file "$CFG_SEL_FROZEN_JSON"); fi
SUMMARY_FILES=()
for f in "$CFG_FROZEN_SUMMARY" "$CFG_FROZEN_SUMMARY_SPECIFIC"; do
  if [ -n "$f" ]; then SUMMARY_FILES+=(--file "$f"); fi
done
OVERRIDE_ARGS=()
for o in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do OVERRIDE_ARGS+=(--override "$o"); done
DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then
  DEVICE_ARGS=(--device "$DEVICE")
elif [ "$CFG_L_DEVICE" != auto ]; then
  DEVICE_ARGS=(--device "$CFG_L_DEVICE")
fi

echo "[$(ts)] raw eval stage: config=$CONFIG overrides=[${OVERRIDES[*]-}] prepared=$P out_dir=$O versions=[${VERSIONS[*]}]"

GT_ARGS=(--pert-key "$CFG_PERT_KEY" --control "$CFG_CONTROL" --context "$CFG_CONTEXT_KEY=$CFG_TARGET_CONTEXT"
         --metadata-json "$CFG_TRAIN_JSON" --min-cells "$CFG_GT_MIN_CELLS" --alpha "$CFG_GT_ALPHA"
         --min-lfc "$CFG_GT_MIN_LFC" --seed "$CFG_GT_SEED")
truth_steps() {  # truth_steps <version>
  local v="$1"
  run_step "truth_$v" --code "$E/evaluation/ground_truth.py" --file "$CFG_TRAIN_JSON" \
    --stat "$(h5 test "$v")" --stat "$(h5 train "$v")" -- \
    "$PYTHON" -u "$E/evaluation/ground_truth.py" --pert-h5ad "$(h5 test "$v")" --ctrl-h5ad "$(h5 train "$v")" \
    "${GT_ARGS[@]}" --out-dir "$O/truth_$v"
  for which in val test; do
    run_step "subset_${which}_$v" --code "$E/evaluation/subset_truth.py" --file "$CFG_SPLIT_JSON" -- \
      "$PYTHON" -u "$E/evaluation/subset_truth.py" --truth-dir "$O/truth_$v" \
      --split-json "$CFG_SPLIT_JSON" --which "$which" --out-dir "$O/truth_${v}_$which"
  done
}

stage_sweep() {  # link the configured sweep files, and only those, into one directory for select_sweep.py
  # runs as a run_step condition, where set -e is off: every command checks its own status
  local line name path dir
  rm -rf "$O/selection/sweep" && mkdir -p "$O/selection/sweep" || return 1
  for line in "${SWEEP[@]}"; do
    IFS=$'\t' read -r _ name path <<< "$line"
    dir=$(cd "$(dirname "$path")" && pwd) || return 1
    ln -s "$dir/$(basename "$path")" "$O/selection/sweep/$name.parquet" || return 1
    echo "$name -> $dir/$(basename "$path")"
  done
}

# 1. primary truth and the A5 gate (the frozen STATE truth, all perturbations and the test subset)
truth_steps "$PRIMARY"
run_step a5_truth --code "$E/data/verify_merge.py" "${FROZEN_TRUTH_STATS[@]}" -- \
  "$PYTHON" -u "$E/data/verify_merge.py" truth --ours "$O/truth_$PRIMARY" \
  --reference "$CFG_FROZEN_TRUTH" --out "$O/verify/a5_truth.json" --tol "$CFG_A5_TOL"
run_step a5_truth_test --code "$E/data/verify_merge.py" "${FROZEN_TEST_STATS[@]}" -- \
  "$PYTHON" -u "$E/data/verify_merge.py" truth --ours "$O/truth_${PRIMARY}_test" \
  --reference "$CFG_FROZEN_TRUTH_TEST" --out "$O/verify/a5_truth_test.json" --tol "$CFG_A5_TOL"

# 2. validation check: the frozen guidance choice, rescored on the primary version's val truth
run_step stage_sweep ${SWEEP_STATS[@]+"${SWEEP_STATS[@]}"} -- stage_sweep
run_step select_sweep --code "$E/generation/select_sweep.py" --file "$CFG_SPLIT_JSON" -- \
  "$PYTHON" -u "$E/generation/select_sweep.py" --sweep-dir "$O/selection/sweep" \
  --truth-dir "$O/truth_${PRIMARY}_val" --val-split "$CFG_SPLIT_JSON" --tie-tolerance "$CFG_SEL_TIE_TOL" \
  --out "$O/selection/selection_$PRIMARY.json"
SEL_FROZEN_ARGS=()
if [ -n "$CFG_SEL_FROZEN_JSON" ]; then SEL_FROZEN_ARGS=(--frozen "$CFG_SEL_FROZEN_JSON"); fi
run_step select_check --code "$E/evaluation/side_by_side.py" ${SEL_FILES[@]+"${SEL_FILES[@]}"} -- \
  "$PYTHON" -u "$E/evaluation/side_by_side.py" check-selection \
  --rescored "$O/selection/selection_$PRIMARY.json" --expected "$CFG_SEL_EXPECTED" \
  ${SEL_FROZEN_ARGS[@]+"${SEL_FROZEN_ARGS[@]}"} --out "$O/selection/check.json"

# 3. the other versions' truth, and how they differ from the primary (Tier B, a report)
TIERB_INPUTS=()
for v in "${VERSIONS[@]}"; do
  if [ "$v" = "$PRIMARY" ]; then continue; fi
  truth_steps "$v"
  run_step "tierb_$v" --code "$E/data/verify_merge.py" -- "$PYTHON" -u "$E/data/verify_merge.py" compare \
    --filtered "$O/truth_$PRIMARY" --all "$O/truth_$v" \
    --out-dir "$O/verify/${PRIMARY}_vs_$v" --min-degs "$CFG_MIN_DEGS"
  run_step "tierb_test_$v" --code "$E/data/verify_merge.py" -- "$PYTHON" -u "$E/data/verify_merge.py" compare \
    --filtered "$O/truth_${PRIMARY}_test" --all "$O/truth_${v}_test" \
    --out-dir "$O/verify/${PRIMARY}_vs_${v}_test" --min-degs "$CFG_MIN_DEGS"
  TIERB_INPUTS+=(--input "tierb_$v=$O/verify/${PRIMARY}_vs_$v/filtered_vs_all.json"
                 --input "tierb_test_$v=$O/verify/${PRIMARY}_vs_${v}_test/filtered_vs_all.json")
done

# 4. baselines, refitted on each version's own train file
for v in "${VERSIONS[@]}"; do
  run_step "baselines_$v" --code "$E/baselines/baselines.py" --stat "$(h5 train "$v")" -- \
    "$PYTHON" -u "$E/baselines/baselines.py" --train-h5ad "$(h5 train "$v")" --pert-key "$CFG_PERT_KEY" \
    --control "$CFG_CONTROL" --context-key "$CFG_CONTEXT_KEY" --target-context "$CFG_TARGET_CONTEXT" \
    --truth-dir "$O/truth_$v" --min-cells "$CFG_BASE_MIN_CELLS" --recompute --out-dir "$O/baselines_$v"
  run_step "learned_$v" --code "$E/baselines/learned_baselines.py" --stat "$(h5 train "$v")" -- \
    "$PYTHON" -u "$E/baselines/learned_baselines.py" --train-h5ad "$(h5 train "$v")" \
    --truth-dir "$O/truth_$v" --pert-key "$CFG_PERT_KEY" --control "$CFG_CONTROL" --context-key "$CFG_CONTEXT_KEY" \
    --target-context "$CFG_TARGET_CONTEXT" --alpha "$CFG_L_ALPHA" --min-lfc "$CFG_L_MIN_LFC" \
    --min-cells "$CFG_L_MIN_CELLS" --embedding-dim "$CFG_L_EMB_DIM" --epochs "$CFG_L_EPOCHS" \
    --batch-size "$CFG_L_BATCH" --seed "$CFG_L_SEED" ${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"} --recompute \
    --out-dir "$O/learned_$v"
done

# 5. evaluate each version on its own test truth
EVAL_COMMON=(--exclude-target --min-degs "$CFG_MIN_DEGS" --eps "$CFG_EPS" --seed "$CFG_EVAL_SEED"
             --unchanged-max-effect "$CFG_UNCH_MAX_EFFECT" --unchanged-min-padj "$CFG_UNCH_MIN_PADJ"
             --min-unchanged "$CFG_MIN_UNCHANGED")
if [ "${#DIRECTION_ONLY[@]}" -gt 0 ]; then EVAL_COMMON+=(--direction-only "${DIRECTION_ONLY[@]}"); fi
EVAL_DIRS=()
CONTROL_DIRS=()
# the tables also depend on each eval dir's changed-vs-unchanged summary: hashed by content ("missing" when an
# eval dir has no such report), next to the eval steps' markers in UPSTREAM
EVAL_UNCHANGED=()
CONTROL_UNCHANGED=()
for v in "${VERSIONS[@]}"; do
  B="$O/baselines_$v"; L="$O/learned_$v"
  run_step "eval_$v" --code "$E/evaluation/evaluate.py" "${PRED_STATS[@]}" -- \
    "$PYTHON" -u "$E/evaluation/evaluate.py" --truth-dir "$O/truth_${v}_test" --pred "${PRED_ARGS[@]}" \
    --baseline "knn=$B/delta_knn.parquet" "mlp=$L/delta_mlp.parquet" "generic=$B/delta_generic.parquet" \
    "transfer=$B/delta_transfer.parquet" "prior=$L/delta_prior.parquet" \
    --specific-ref "$B/delta_generic.parquet" "${EVAL_COMMON[@]}" --out-dir "$O/eval_$v"
  EVAL_DIRS+=(--eval "$v=$O/eval_$v")
  EVAL_UNCHANGED+=(--file "$O/eval_$v/summary_unchanged.csv")
  if [ "${#CONTROL_ARGS[@]}" -gt 0 ]; then
    run_step "controls_$v" --code "$E/evaluation/evaluate.py" "${CONTROL_STATS[@]}" -- \
      "$PYTHON" -u "$E/evaluation/evaluate.py" --truth-dir "$O/truth_${v}_test" \
      --pred "$REF_ARG" "${CONTROL_ARGS[@]}" --specific-ref "$B/delta_generic.parquet" "${EVAL_COMMON[@]}" \
      --out-dir "$O/eval_controls_$v"
    CONTROL_DIRS+=(--eval "$v=$O/eval_controls_$v")
    CONTROL_UNCHANGED+=(--file "$O/eval_controls_$v/summary_unchanged.csv")
  fi
done

# 6. the side-by-side tables (+ provenance) in results_dir
SBS_COMMON=(--primary "$PRIMARY" --reference "$CFG_REFERENCE" --metric "$CFG_METRIC"
            --unchanged-metric "$CFG_UNCH_METRIC" --min-degs "$CFG_MIN_DEGS" --min-unchanged "$CFG_MIN_UNCHANGED"
            --n-boot "$CFG_N_BOOT" --seed "$CFG_STATS_SEED" --out-dir "$CFG_RESULTS_DIR" --timestamp "$STARTED"
            --config "$CONFIG" ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"}
            --input "manifest=$P/manifest.json" --input "inputs_stamp=$STAMP" --input "a5_truth=$O/verify/a5_truth.json"
            --input "a5_truth_test=$O/verify/a5_truth_test.json" --input "selection_check=$O/selection/check.json"
            ${TIERB_INPUTS[@]+"${TIERB_INPUTS[@]}"} "${SWEEP_INPUTS[@]}")
FROZEN_ARGS=()
if [ -n "$CFG_FROZEN_SUMMARY" ]; then FROZEN_ARGS+=(--frozen-summary "full=$CFG_FROZEN_SUMMARY"); fi
if [ -n "$CFG_FROZEN_SUMMARY_SPECIFIC" ]; then FROZEN_ARGS+=(--frozen-summary "specific=$CFG_FROZEN_SUMMARY_SPECIFIC"); fi
SBS_HASH=(--code "$E/evaluation/side_by_side.py" --file "$P/manifest.json" --mask-argv "$STARTED")
run_step side_by_side "${SBS_HASH[@]}" ${SUMMARY_FILES[@]+"${SUMMARY_FILES[@]}"} \
  ${EVAL_UNCHANGED[@]+"${EVAL_UNCHANGED[@]}"} -- \
  "$PYTHON" -u "$E/evaluation/side_by_side.py" table "${EVAL_DIRS[@]}" "${SBS_COMMON[@]}" \
  ${FROZEN_ARGS[@]+"${FROZEN_ARGS[@]}"} --tag "$CFG_RESULTS_TAG" \
  --title "Replogle HepG2 test perturbations, raw rebuild: ${VERSIONS[*]} side by side"
TABLES="$CFG_RESULTS_DIR/${CFG_RESULTS_TAG}_side_by_side.md"
if [ "${#CONTROL_DIRS[@]}" -gt 0 ]; then
  run_step side_by_side_controls "${SBS_HASH[@]}" ${CONTROL_UNCHANGED[@]+"${CONTROL_UNCHANGED[@]}"} -- \
    "$PYTHON" -u "$E/evaluation/side_by_side.py" table "${CONTROL_DIRS[@]}" "${SBS_COMMON[@]}" \
    --tag "${CFG_RESULTS_TAG}_controls" --title "Replogle HepG2 test perturbations, raw rebuild: generation controls"
  TABLES="$TABLES $CFG_RESULTS_DIR/${CFG_RESULTS_TAG}_controls_side_by_side.md"
fi

COMMIT=$(git -C "$JOB_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
printf 'finished %s\ncommit %s\nversions %s\nchosen %s\ntables %s\nprovenance %s\n' "$(ts)" "$COMMIT" \
  "${VERSIONS[*]}" "$CFG_SEL_EXPECTED" "$TABLES" "$CFG_RESULTS_DIR/$CFG_RESULTS_TAG.provenance.json" > "$O/DONE_raw_eval"
echo "[$(ts)] RAW_EVAL_DONE -> $O/DONE_raw_eval"
