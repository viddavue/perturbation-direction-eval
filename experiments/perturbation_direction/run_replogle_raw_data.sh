#!/usr/bin/env bash
# Raw Replogle rebuild, data stage: checksums -> audit every raw screen -> prepare (filtered + all) -> identity gates A1-A4.
#
# Everything is read from the config (experiments/configs/perturbation_direction/replogle_raw.yaml);
# extra arguments are OmegaConf overrides passed to every step (e.g. out_dir=/some/where block=10000).
# Each step logs to <out_dir>/logs/<step>.log and stops the run on failure (non-zero exit).
#
# Steps
#   checksums         recompute every raw file's md5 (MD5_JOBS files at a time, default 4) and compare it with
#                     the config md5 and the .md5 sidecar -> <out_dir>/checksums.json
#   audit_<screen>    per-screen audit (audit_raw_screen.py) -> <out_dir>/audit/
#   prepare           both versions + manifest.json (records md5_computed from checksums.json)
#   verify_identity   Tier A gates A1-A4 against STATE -> <out_dir>/verify/identity.json
#
# Resume. A finished step writes <out_dir>/logs/<step>.done whose first line is "hash=<sha256>" over
# (step_hash.py): the resolved config (provenance-only keys such as md5_source left out), the step's command
# line (without the overrides, which act through the resolved config), the sha256 of its script and of every
# pipeline module the script imports, small input files (metadata JSONs, .md5 sidecars, state_reference), the
# size/mtime of the raw files (checksums only), and the .done contents of its upstream steps. A step is
# skipped while that hash is unchanged and rerun when it differs (the log names the components that changed;
# <step>.hash.json holds them). A step that runs first deletes its own .done, the .done of every step
# downstream of it, and DONE_raw_data. FORCE=1 reruns every step.
#
# Upstream / downstream. Data flows only prepare -> verify_identity. checksums and audit_<screen> are gates:
# they write reports no step reads, so they are upstream of nothing, and a failing gate stops the run before
# prepare. The raw bytes the audits and prepare read are pinned by the config md5 (part of every hash) and by
# the .md5 sidecars (hashed by content), and checked by the checksums gate, which reruns whenever a raw file's
# size or mtime changes. Package versions are not hashed: FORCE=1 after changing the environment.
#
# Legacy markers (the earlier runner wrote a timestamp, no hash) are stale for the gates and verify_identity,
# which rerun. A legacy prepare.done is adopted (kept, its hash filled in, nothing rebuilt) when
# `step_hash.py adopt-prepare` finds manifest.json consistent with the current config, knockdown-filter
# module, raw files (checksums.json md5, sizes, mtimes) and outputs (sizes, mtimes); otherwise prepare
# reruns. A legacy marker records no code hash, so adoption cannot prove which prepare code wrote the files;
# verify_identity re-checks the filtered outputs against STATE. ADOPT_LEGACY=0 rebuilds instead.
#
# The run ends with <out_dir>/DONE_raw_data (commit, time, versions, the hash of every step).
#
# Inputs that must exist: every screen's raw file plus its .md5 (written when the download finished;
# MD5_WAIT=<seconds> polls for a missing one), and state_reference (built locally by
# make_state_reference.py from the 22 GB STATE file, then copied to the same relative path here).
#
# Usage (paths in the config are relative to the repository root; the script cds there):
#   bash experiments/perturbation_direction/run_replogle_raw_data.sh                      # student03
#   PYTHON=$PWD/.venv/bin/python bash experiments/perturbation_direction/run_replogle_raw_data.sh raw_dir=... out_dir=...
#   tmux new -s rawdata 'bash experiments/perturbation_direction/run_replogle_raw_data.sh; bash'
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_ROOT=${JOB_ROOT:-$(cd "$HERE/../.." && pwd)}
PYTHON=${PYTHON:-/system/apps/studentenv/fleischa/miniconda3/envs/scldm/bin/python}
CONFIG=${CONFIG:-experiments/configs/perturbation_direction/replogle_raw.yaml}
MD5_WAIT=${MD5_WAIT:-0}
MD5_JOBS=${MD5_JOBS:-4}
FORCE=${FORCE:-0}
ADOPT_LEGACY=${ADOPT_LEGACY:-1}
E=experiments/perturbation_direction
cd "$JOB_ROOT"
OVERRIDES=("$@")

# resolved config -> shell variables + one tab-separated line per screen
CFG_DUMP=$("$PYTHON" - "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"} <<'PY'
import shlex, sys
sys.path.insert(0, "experiments/perturbation_direction")
from prepare_replogle_raw import load_config
from step_hash import config_digest
cfg = load_config(sys.argv[1], sys.argv[2:])
for k in ("raw_dir", "out_dir", "train_json", "test_json", "state_reference", "pert_key", "control", "gene_col", "block"):
    print(f"CFG_{k.upper()}={shlex.quote(str(cfg[k]))}")
print(f"CFG_MIN_CONTROLS={int(cfg.audit.min_controls)}")
print(f"CFG_VERSIONS={shlex.quote(' '.join(cfg.versions))}")
print(f"CFG_DIGEST={config_digest(cfg)}")
for s in cfg.screens:
    print(f"SCREEN\t{s.name}\t{s.cell_line}\t{s.file}")
PY
)
eval "$(grep '^CFG_' <<< "$CFG_DUMP")"
mapfile -t SCREENS < <(grep $'^SCREEN\t' <<< "$CFG_DUMP")

LOG_DIR="$CFG_OUT_DIR/logs"
mkdir -p "$LOG_DIR" "$CFG_OUT_DIR/audit"
ts() { date +%Y-%m-%dT%H:%M:%S; }

# data dependencies: step -> upstream steps whose .done enters its hash (gates feed no step)
declare -A UPSTREAM=([verify_identity]="prepare")
STEPS=()

downstream_of() {  # every step whose upstream chain contains $1 (one per line; may repeat)
  local t
  for t in "${!UPSTREAM[@]}"; do
    case " ${UPSTREAM[$t]} " in *" $1 "*) echo "$t"; downstream_of "$t" ;; esac
  done
}

invalidate() {  # $1 is about to run: it, everything downstream of it and DONE_raw_data stop being done
  local t
  rm -f "$LOG_DIR/$1.done" "$CFG_OUT_DIR/DONE_raw_data"
  for t in $(downstream_of "$1" | sort -u); do
    if [ -f "$LOG_DIR/$t.done" ]; then
      rm -f "$LOG_DIR/$t.done"
      echo "        invalidated $t (downstream of $1)"
    fi
  done
}

adopt_prepare() {  # <hash>: keep a legacy prepare.done whose outputs are consistent with manifest.json
  local hash="$1" marker="$LOG_DIR/prepare.done" legacy
  legacy=$(head -n 1 "$marker")
  if "$PYTHON" "$E/step_hash.py" adopt-prepare --config "$CONFIG" --out "$LOG_DIR/prepare.adopt.json" \
      ${OVERRIDES[@]+"${OVERRIDES[@]}"} > "$LOG_DIR/prepare.adopt.log" 2>&1; then
    printf 'hash=%s\nfinished=%s\nadopted=%s legacy marker kept, outputs consistent with manifest.json (%s)\n' \
      "$hash" "$legacy" "$(ts)" "$LOG_DIR/prepare.adopt.json" > "$marker"
    mv -f "$LOG_DIR/prepare.hash.next.json" "$LOG_DIR/prepare.hash.json"
    echo "[$(ts)] adopt prepare (legacy marker of $legacy; nothing rebuilt; checks: $LOG_DIR/prepare.adopt.json)"
    return 0
  fi
  echo "[$(ts)] legacy prepare.done is not adoptable:"
  sed 's/^/        /' "$LOG_DIR/prepare.adopt.log"
  return 1
}

# run_step <name> [--overrides] [--code <script> | --file <small file> | --stat <large file>]... -- <command...>
# --overrides appends the config overrides to the command; they enter the hash through CFG_DIGEST only.
run_step() {
  local name="$1"; shift
  local -a hargs=() extra=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
      --overrides) extra=(${OVERRIDES[@]+"${OVERRIDES[@]}"}); shift ;;
      --code|--file|--stat) hargs+=("$1" "$2"); shift 2 ;;
      *) echo "run_step $name: unknown option $1" >&2; exit 2 ;;
    esac
  done
  shift  # --
  local up
  for up in ${UPSTREAM[$name]-}; do hargs+=(--upstream "$LOG_DIR/$up.done"); done
  STEPS+=("$name")
  local marker="$LOG_DIR/$name.done" info="$LOG_DIR/$name.hash.json" next="$LOG_DIR/$name.hash.next.json"
  local hash reason first
  hash=$("$PYTHON" "$E/step_hash.py" hash --config-digest "$CFG_DIGEST" --info "$next" ${hargs[@]+"${hargs[@]}"} -- "$@")
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
      reason="legacy marker without a hash: $first"
      if [ "$name" = prepare ] && [ "$ADOPT_LEGACY" = 1 ]; then
        if adopt_prepare "$hash"; then return 0; fi
        reason="$reason; not adoptable, see $LOG_DIR/prepare.adopt.log"
      fi
    else
      reason="hash changed: $("$PYTHON" "$E/step_hash.py" diff "$info" "$next")"
    fi
  fi
  invalidate "$name"
  echo "[$(ts)] start $name ($reason) -> $LOG_DIR/$name.log"
  if "$@" ${extra[@]+"${extra[@]}"} > "$LOG_DIR/$name.log" 2>&1; then
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

wait_for_md5() {  # a download is complete only once its non-empty .md5 exists
  local f="$1" waited=0
  until [ -s "$f.md5" ]; do
    if [ "$waited" -ge "$MD5_WAIT" ]; then
      echo "[$(ts)] FAIL  $f.md5 missing or empty (download not finished?); set MD5_WAIT=<s> to wait" >&2
      exit 3
    fi
    sleep 30; waited=$((waited + 30))
  done
}

[ -f "$CFG_STATE_REFERENCE" ] || {
  echo "missing state_reference: $JOB_ROOT/$CFG_STATE_REFERENCE (make it locally with make_state_reference.py, then copy it here)" >&2
  exit 2
}

echo "[$(ts)] raw data stage: config=$CONFIG overrides=[${OVERRIDES[*]-}] out_dir=$CFG_OUT_DIR config_digest=${CFG_DIGEST:0:12}"

# 1. recompute every raw file's md5 (gate)
CK_INPUTS=() MD5_FILES=()
for line in "${SCREENS[@]}"; do
  IFS=$'\t' read -r _ name cell_line file <<< "$line"
  wait_for_md5 "$CFG_RAW_DIR/$file"
  CK_INPUTS+=(--stat "$CFG_RAW_DIR/$file" --file "$CFG_RAW_DIR/$file.md5")
  MD5_FILES+=(--file "$CFG_RAW_DIR/$file.md5")
done
run_step checksums --overrides --code "$E/verify_merge.py" "${CK_INPUTS[@]}" -- \
  "$PYTHON" -u "$E/verify_merge.py" checksums --config "$CONFIG" --jobs "$MD5_JOBS"

# 2. audit every screen (gate, read-only)
for line in "${SCREENS[@]}"; do
  IFS=$'\t' read -r _ name cell_line file <<< "$line"
  run_step "audit_$name" --code "$E/audit_raw_screen.py" --file "$CFG_TRAIN_JSON" \
    --file "$CFG_RAW_DIR/$file.md5" -- \
    "$PYTHON" -u "$E/audit_raw_screen.py" --h5ad "$CFG_RAW_DIR/$file" --cell-line "$cell_line" \
    --metadata-json "$CFG_TRAIN_JSON" --out "$CFG_OUT_DIR/audit/audit_$name.json" --pert-key "$CFG_PERT_KEY" \
    --control "$CFG_CONTROL" --gene-col "$CFG_GENE_COL" --min-controls "$CFG_MIN_CONTROLS" --block "$CFG_BLOCK"
done

# 3. harmonise: knockdown filter, 2,000 model genes, vocab restriction, split, both versions
run_step prepare --overrides --code "$E/prepare_replogle_raw.py" --file "$CFG_TRAIN_JSON" --file "$CFG_TEST_JSON" \
  "${MD5_FILES[@]}" -- \
  "$PYTHON" -u "$E/prepare_replogle_raw.py" --config "$CONFIG"

# 4. Tier A identity gates A1-A4 (primary version vs STATE)
PREPARED_STATS=()
for v in $CFG_VERSIONS; do
  for split in train test; do PREPARED_STATS+=(--stat "$CFG_OUT_DIR/replogle_${split}_raw_${v}.h5ad"); done
done
# the prepared files and manifest are hashed too, so outputs rewritten outside the runner re-trigger the gates
run_step verify_identity --overrides --code "$E/verify_merge.py" --file "$CFG_STATE_REFERENCE" \
  --file "$CFG_OUT_DIR/manifest.json" "${PREPARED_STATS[@]}" -- \
  "$PYTHON" -u "$E/verify_merge.py" identity --config "$CONFIG"

COMMIT=$(git -C "$JOB_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
{
  printf 'finished %s\ncommit %s\nversions %s\nmanifest %s\nchecksums %s\nidentity %s\nconfig_digest %s\n' \
    "$(ts)" "$COMMIT" "$CFG_VERSIONS" "$CFG_OUT_DIR/manifest.json" "$CFG_OUT_DIR/checksums.json" \
    "$CFG_OUT_DIR/verify/identity.json" "$CFG_DIGEST"
  for s in "${STEPS[@]}"; do printf 'step %s %s\n' "$s" "$(head -n 1 "$LOG_DIR/$s.done")"; done
} > "$CFG_OUT_DIR/DONE_raw_data"
echo "[$(ts)] RAW_DATA_DONE -> $CFG_OUT_DIR/DONE_raw_data"
