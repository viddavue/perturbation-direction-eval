# Running the Replogle evaluation on a GPU server

Generation is the expensive step, and it is done: the frozen predictions are reused, not regenerated.
Everything else (data, ground truth, baselines, evaluation) runs on CPU.

**Replogle is run from the raw screens now** (sections 4 and 5): `run_replogle_raw_data.sh` builds the
train/test files in two versions, `filtered` (STATE's knockdown filter reproduced from raw; the primary) and
`all` (no filter), and `run_replogle_raw_eval.sh` evaluates both end to end into one side-by-side main table.
The STATE-file route that produced the frozen `results/replogle_test312_w2ema_3seeds_*` tables is kept at the
end as a legacy note.

Paths below are relative to the repo root. `E=experiments/perturbation_direction`.

## 1. Code and environment (once)

```bash
git clone <your-remote> scldm && cd scldm && git checkout perturbation-direction-eval
# or: rsync -a --exclude .venv --exclude _artifacts --exclude outputs <pc>:PycharmProjects/perturbation-direction-eval/ scldm/

# uv: curl -LsSf https://astral.sh/uv/install.sh | sh    (if not installed)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install --override experiments/perturbation_direction/overrides.txt -e ".[test]" scanpy pyarrow h5py \
  "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git@a3720244bf0cfc4950e89ad583ceb0f86ed07199"
python -m pytest -q tests            # expect: ~247 passed, 5 skipped (2026-11-10)
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

`overrides.txt` pins anndata, scvi-tools, scanpy and cellarium-ml to versions that work together; the
repo's own pins conflict. torch 2.7 installs with CUDA 12.6 wheels by default. If the last command prints
`0` GPUs or fails, the server's NVIDIA driver is too old for those wheels. In that case, install the
torch 2.7.0 build for the CUDA version that `nvidia-smi` reports, for example
`uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118`.

**GPU pinning.** Other users may occupy GPUs. Pin with `CUDA_VISIBLE_DEVICES`, e.g.
`CUDA_VISIBLE_DEVICES=2,3` to restrict to specific devices. On student07, never use GPUs 0 and 3.

**Launcher-script pattern for tmux.** Start long runs in a tmux session so they survive SSH
disconnection. Use `tmux new -s <name> 'bash <script>; bash'` (the trailing `bash` keeps the
session open on failure). Avoid `pkill -f` patterns that match the ssh command line itself;
use `tmux kill-session -t <name>` or kill by PID instead.

## 2. Model checkpoint (~1 GB)

```bash
python $E/fetch_artifacts.py --match replogle --exclude gaussian     # -> _artifacts/resubmission/
```

## 3. Generation (multi-GPU)

**The main table's generations already exist and are reused:** the guidance/EMA sweep on the 60 validation
perturbations (`run_sweep_replogle.sh` -> `sweep/{w1,w1_ema,w2,w2_ema,w4,w4_ema}.parquet`) and the chosen
config, w2 + EMA, on the 312 test perturbations with seeds 0-2 (`run_seeds_replogle.sh` /
`run_overnight_replogle.sh` -> `pred/test312_w2_ema_seed{0,1,2}.parquet`, plus the `cells128`, `w1` and `w0`
controls). On student03 they are in
`/system/user/studentwork/fleischa/scldm_generation_20261020/outputs/perturbation_direction/replogle/`
(sha256 identical to the local copies under `outputs/perturbation_direction/replogle/`). The split is
`experiments/perturbation_direction/splits/replogle_val60.json`. The recipe below is the generic multi-GPU one.

The conditions come straight from the test metadata (HepG2 control + 372 test perturbations), so this
step can start right away.

```bash
O=outputs/perturbation_direction/replogle; A=_artifacts/resubmission
python $E/make_conditions.py --test-json metadata/replogle_test.json --pert-key gene \
  --control non-targeting --context cell_line=hepg2 --out $O/conditions_all.csv

gen() {  # usage: gen <guidance> <ema flag or ""> <gpu list>
  python $E/generate.py --config $A/replogle_negative_binomial.yaml --ckpt $A/replogle_negative_binomial.ckpt \
    --metadata-json metadata/replogle_train.json --conditions $O/conditions_all.csv \
    --n-cells 64 --batch-size 32 --library-size 4850 --guidance $1 $2 --gpus $3 \
    --out $O/pred/w$1${2:+_ema}.parquet
}
tmux new -s gen    # keeps running after you disconnect (detach: Ctrl-b d; reattach: tmux a -t gen)
gen 1 "" 0,1,2,3          # main result: raw weights, as LatentDiffusion.sample() uses
gen 1 --ema 0,1,2,3
gen 2 "" 0,1,2,3
gen 4 "" 0,1,2,3
```

- **Workers.** `--gpus 0,1,2,3` starts one worker per GPU. Each worker's log is in
  `$O/pred/<name>.parts/shardXXXofYYY.log`, and shows a per-condition time and an ETA. When all
  workers finish, the shards are merged into `$O/pred/<name>.parquet`, plus `<name>_sample.parquet`.
- **Resuming.** Every condition is saved as soon as it finishes. If something dies, rerun the same
  command, even with a different GPU list, and finished conditions are skipped. Changing a setting
  (`--guidance`, `--n-cells`, ...) under the same `--out` is refused, so use a new `--out` instead.
- **Batch size 32 is right.** The GPU is already compute-bound at 32, so larger batches don't speed
  things up per cell.
- **Expected time.** On an RTX 3060, one batch of 32 cells takes about 11 s at w=1 and about 22 s at
  w>1. One configuration is 373 conditions × 2 batches, so about 2.3 h at w=1 and about 4.6 h at w>1 on
  one such GPU, divided by the number of GPUs. Datacenter GPUs are several times faster.
- **Reproducibility.** Results don't depend on how many GPUs are used. Every condition uses the same
  noise for batch b, whichever shard runs it, so the output is bit-identical to a single-GPU run.

SLURM alternative (one array task per shard, then merge):

```bash
#SBATCH --array=0-7 --gres=gpu:1
srun python $E/generate.py <same args as above, without --gpus> --shard $SLURM_ARRAY_TASK_ID --num-shards 8
# afterwards, once all tasks have finished:
python $E/generate.py <same args> --merge-only
```

## 4. Data: raw rebuild (CPU)

Builds the Replogle train/test files from the four raw screens instead of the STATE matrix, in two
versions: `filtered` (STATE's knockdown filter reproduced from raw, the primary) and `all` (no filter).
Config: `experiments/configs/perturbation_direction/replogle_raw.yaml` (raw/out dirs, screens with
URL/size/md5, filter parameters). Any key can be overridden as `key=value` on every command below.

```bash
# once, on the PC (reads the 22 GB STATE file in row blocks, ~4 min, 1.5 GB RAM):
python $E/make_state_reference.py --state _artifacts/datasets/replogle_state/replogle.h5ad \
  --train _artifacts/datasets/replogle_state/replogle_train.h5ad --test _artifacts/datasets/replogle_state/replogle_test.h5ad \
  --metadata-json metadata/replogle_train.json --out-dir outputs/perturbation_direction/replogle_raw/state_reference
rsync -a outputs/perturbation_direction/replogle_raw/state_reference/ <server>:<repo>/outputs/perturbation_direction/replogle_raw/state_reference/

# on student03, from the repo root (checksums -> audit x4 -> prepare -> identity gates A1-A4; per-step hash resume,
# FORCE=1 reruns all, MD5_JOBS=<n> parallel md5s; DONE marker at the end; see the runner's header):
tmux new -s rawdata 'bash experiments/perturbation_direction/run_replogle_raw_data.sh; bash'
cat /system/user/studentwork/fleischa/replogle_raw_prepared/DONE_raw_data
```

The runner's steps, one by one:

```bash
python $E/verify_merge.py checksums --config experiments/configs/perturbation_direction/replogle_raw.yaml
python $E/audit_raw_screen.py --h5ad <raw>.h5ad --cell-line hepg2 --metadata-json metadata/replogle_train.json --out audit_hepg2.json
python $E/prepare_replogle_raw.py --config experiments/configs/perturbation_direction/replogle_raw.yaml
python $E/verify_merge.py identity --config experiments/configs/perturbation_direction/replogle_raw.yaml
```

- **Outputs** (`out_dir`): `replogle_{train,test}_raw_{filtered,all}.h5ad` (CSR int32 counts over the 2,000
  model genes; obs `cell_id`, `gene`, `cell_line`, `screen`, `gem_group`, `UMI_count`, `library_full`,
  `kd_keep`), `manifest.json` (inputs with md5, commit, config, counts per version/split/line, filter and
  split checks), `knockdown_report.json`, `test_target_cells.csv`, `audit/`, `verify/identity.json`, `logs/`.
- **Gates.** The audit fails on non-integer or negative counts, fewer than 300 controls, a zero-library cell,
  or a model gene missing or duplicated. prepare fails if a test target is a HepG2 training perturbation,
  has no donor line, or has fewer than 30 cells in `filtered`. identity fails unless the `filtered` cells,
  labels, counts (sha1), full-file libraries (the same integer) and train/test split all equal STATE's; a
  duplicated cell id or a line that falls back to key matching (without `--allow-key-mapping`) fails too.
  Cells are matched by barcode; 777 STATE barcodes (92 Jurkat, 263 K562, 422 RPE1) carry the `-1` that
  anndata's `obs_names_make_unique` added when STATE merged screens with repeated barcodes, and are matched
  after removing it (`verify/identity.json` -> `mapping`).
- **Resources.** Each raw file is read twice (knockdown pass, data pass) in 20,000-row blocks (~2 GB RAM).
  Parts are concatenated on disk, so `out_dir` briefly needs about twice the final size. Measured on student03:
  14.5 GB final for both versions (train 6.1 + 7.8 GB, test 0.3 + 0.3 GB); prepare took 17 min for all four screens.
- **ADOPT_LEGACY and per-step hashes.** The data runner (`run_replogle_raw_data.sh`) uses `step_hash.py`
  for incremental resume. A finished step writes `<out_dir>/logs/<step>.done` whose first line is
  `hash=<sha256>` over the resolved config (provenance-only keys stripped), the step's command line,
  the sha256 of its script and all transitively imported pipeline modules, small input files (metadata
  JSONs, `.md5` sidecars, `state_reference`), raw file size/mtime (checksums only), and the upstream
  steps' `.done` contents. Each step also writes `<step>.hash.json` with the full components. A step
  is skipped while its hash is unchanged. `ADOPT_LEGACY=1` (default) keeps a legacy `prepare.done`
  (no hash, a timestamp from an earlier runner) when `step_hash.py adopt-prepare` finds
  `manifest.json` consistent with the current config, knockdown-filter module, raw files and output
  sizes/mtimes; otherwise prepare reruns. A legacy marker records no code hash, so adoption cannot
  prove which code wrote the files; `verify_identity` always re-checks the filtered outputs against
  STATE. Set `ADOPT_LEGACY=0` to rebuild instead.

## 5. Evaluation: both versions end to end (CPU)

`run_replogle_raw_eval.sh` turns the prepared files into the side-by-side main table. Config:
`experiments/configs/perturbation_direction/replogle_raw_eval.yaml` (prepared/out/results dirs, versions,
ground-truth and baseline settings, the frozen truth and selection, the prediction files); any key can be
overridden as `key=value`. As in the data runner: one log per step in `<out_dir>/logs/`, the run stops at
the first failing step, and ends with `<out_dir>/DONE_raw_eval`. Resume is simpler than the data runner's
per-step hashes: a step with a `.done` marker is skipped, and `<out_dir>/logs/inputs.sha256` pins what the
stage reads from outside `out_dir` (resolved config, prepared `manifest.json` and h5ad size/mtime, metadata,
split, frozen truth/selection/summaries, every prediction file). When a rerun finds any of it changed, for
example after the data stage was rerun, it prints the difference and reruns every step. Code changes are
not detected: use `FORCE=1`.

```bash
# prerequisites on student03: the data stage finished (DONE_raw_data), and the frozen selection table, which
# lives in outputs/ (not in git), is at the configured path (or run with selection.frozen_json=null):
#   rsync -a outputs/perturbation_direction/replogle/selection.json <server>:<repo>/outputs/perturbation_direction/replogle/
tmux new -s raweval 'bash experiments/perturbation_direction/run_replogle_raw_eval.sh; bash'
cat /system/user/studentwork/fleischa/replogle_raw_prepared/eval/DONE_raw_eval
cat results/replogle_raw_test312_w2ema_3seeds_side_by_side.md
# on the PC (generations under outputs/):
PYTHON=$PWD/.venv/bin/python bash experiments/perturbation_direction/run_replogle_raw_eval.sh \
  prepared_dir=<prepared dir> generation_dir=outputs/perturbation_direction/replogle
```

Per version `v` (primary first), with `P` = `prepared_dir` and `O` = `out_dir` (default `P/eval`):

| step | what | gate? |
|---|---|---|
| `truth_<v>`, `subset_{val,test}_<v>` | `ground_truth.py` on `P/replogle_test_raw_<v>.h5ad`, HepG2 controls from `P/replogle_train_raw_<v>.h5ad`, all 372; then `subset_truth.py` -> `O/truth_<v>_val` (60), `O/truth_<v>_test` (312) | |
| `a5_truth`, `a5_truth_test` | `verify_merge.py truth`: the `filtered` truth (372 and 312) equals the frozen STATE-based truth (delta, padj to 1e-9, DEG masks, cell counts) | **yes** |
| `stage_sweep`, `select_sweep`, `select_check` | the six sweep files rescored on `O/truth_filtered_val` by `select_sweep.py`; `side_by_side.py check-selection` requires the frozen choice (`w2_ema`) and records the table vs the frozen `selection.json` | **yes** |
| `tierb_<v>`, `tierb_test_<v>` | `verify_merge.py compare`: filtered vs all (per-perturbation correlation, slope, DEG Jaccard, panel sizes) | report |
| `baselines_<v>`, `learned_<v>` | `baselines.py` (generic, transfer, kNN) and `learned_baselines.py` (prior, MLP; seeded), refitted on `P/replogle_train_raw_<v>.h5ad` | |
| `eval_<v>` | `evaluate.py` on `O/truth_<v>_test`: `scLDM_s0..2` + knn, mlp, generic, transfer, prior; `--exclude-target`, `--specific-ref O/baselines_<v>/delta_generic.parquet` | |
| `controls_<v>` | optional: `scLDM_s0` + the generation controls (`cells128`, `w1`, `w0`; missing files are skipped) -> `O/eval_controls_<v>` | |
| `side_by_side[_controls]` | `side_by_side.py table` -> `results/replogle_raw_test312_w2ema_3seeds[_controls]_*` | |

The sweep files are linked one by one into `O/selection/sweep/`: the generation dir also holds
`*_sample.parquet` files, which `select_sweep.py` would score as extra configs if pointed at that directory.

**The side-by-side table** (`results/<tag>_side_by_side.md`, `.csv`, `_specific.csv`, `_long.csv`,
`_panel.csv`, `_vs_frozen.csv`, `<tag>.provenance.json`):

- Rows: scLDM seeds 0-2, kNN, MLP, generic, transfer, prior, the three shuffled controls, the split-half
  ceiling. Column groups: `filtered (primary)` | `all`, for the full effect and for the perturbation-specific
  component.
- Medians, 95% percentile bootstrap CIs (2,000 resamples over perturbations) and paired tests are computed on
  the **shared panel**: perturbations with >= 5 DEGs (target excluded) in **both** versions, per report.
  Each version's own panel (n and median, what `freeze_results.py` would report) is listed beside it.
- Paired: two-sided Wilcoxon signed-rank vs `scLDM_s0` with the bootstrap CI of the median difference
  (`scLDM_s0 − method`); raw and Holm p, Holm within each (report, version) column. Statistics are `stats.py`'s.
- `_vs_frozen.csv`: the filtered own panel against the frozen STATE-based tables. With A5 passed it should
  be 0 for every method whose inputs did not change; the MLP is now seeded, so its row may differ.
- Provenance: git commit and dirty flag, sha256 of every input (prepared `manifest.json`, truth files,
  prediction and baseline parquets, per-perturbation tables, A5 / selection / Tier B reports, sweep files),
  the resolved config, the scripts' sha256, and the run's start time.

The runner's steps by hand (what it runs, with the default config):

```bash
P=/system/user/studentwork/fleischa/replogle_raw_prepared; O=$P/eval
G=/system/user/studentwork/fleischa/scldm_generation_20261020/outputs/perturbation_direction/replogle
F=outputs/perturbation_direction/replogle; S=experiments/perturbation_direction/splits/replogle_val60.json
for V in filtered all; do
  python $E/ground_truth.py --pert-h5ad $P/replogle_test_raw_$V.h5ad --ctrl-h5ad $P/replogle_train_raw_$V.h5ad \
    --pert-key gene --control non-targeting --context cell_line=hepg2 --metadata-json metadata/replogle_train.json \
    --min-cells 15 --out-dir $O/truth_$V
  for W in val test; do
    python $E/subset_truth.py --truth-dir $O/truth_$V --split-json $S --which $W --out-dir $O/truth_${V}_$W
  done
done
python $E/verify_merge.py truth --ours $O/truth_filtered --reference $F/truth --out $O/verify/a5_truth.json
python $E/verify_merge.py truth --ours $O/truth_filtered_test --reference $F/truth_test312 --out $O/verify/a5_truth_test.json
mkdir -p $O/selection/sweep && for C in w1 w1_ema w2 w2_ema w4 w4_ema; do ln -s $G/sweep/$C.parquet $O/selection/sweep/; done
python $E/select_sweep.py --sweep-dir $O/selection/sweep --truth-dir $O/truth_filtered_val --val-split $S \
  --out $O/selection/selection_filtered.json
python $E/side_by_side.py check-selection --rescored $O/selection/selection_filtered.json --expected w2_ema \
  --frozen $F/selection.json --out $O/selection/check.json
python $E/verify_merge.py compare --filtered $O/truth_filtered --all $O/truth_all --out-dir $O/verify/filtered_vs_all
for V in filtered all; do
  T=$P/replogle_train_raw_$V.h5ad; B=$O/baselines_$V; L=$O/learned_$V
  python $E/baselines.py --train-h5ad $T --pert-key gene --control non-targeting --context-key cell_line \
    --target-context hepg2 --truth-dir $O/truth_$V --recompute --out-dir $B
  python $E/learned_baselines.py --train-h5ad $T --truth-dir $O/truth_$V --pert-key gene --control non-targeting \
    --context-key cell_line --target-context hepg2 --recompute --out-dir $L
  python $E/evaluate.py --truth-dir $O/truth_${V}_test --exclude-target \
    --pred scLDM_s0=$G/pred/test312_w2_ema_seed0.parquet scLDM_s1=$G/pred/test312_w2_ema_seed1.parquet \
           scLDM_s2=$G/pred/test312_w2_ema_seed2.parquet \
    --baseline knn=$B/delta_knn.parquet mlp=$L/delta_mlp.parquet generic=$B/delta_generic.parquet \
               transfer=$B/delta_transfer.parquet prior=$L/delta_prior.parquet \
    --specific-ref $B/delta_generic.parquet --out-dir $O/eval_$V
done
python $E/side_by_side.py table --eval filtered=$O/eval_filtered --eval all=$O/eval_all --primary filtered \
  --tag replogle_raw_test312_w2ema_3seeds --out-dir results --timestamp "$(date -u +%FT%TZ)" \
  --input manifest=$P/manifest.json \
  --frozen-summary full=results/replogle_test312_w2ema_3seeds_summary.csv \
  --frozen-summary specific=results/replogle_test312_w2ema_3seeds_summary_specific.csv
```

- **Baselines.** generic, transfer and kNN (`baselines.py`), prior and MLP (`learned_baselines.py`, always
  both; the MLP is seeded), the shuffled control (`evaluate.py`) and the split-half ceiling
  (`ground_truth.py`). calibrated, bilinear, GBDT and `transfer_<line>` are gone. Effects in every line are
  taken against that line's own non-targeting cells.
- **Guidance is not re-chosen.** The config was chosen once, on the 60 validation perturbations of the
  filtered (= STATE) truth, and frozen (`w2_ema`); the `select_check` gate only confirms that the rebuilt
  truth reproduces that choice.
- **Resources.** One process per step; ground truth reads only HepG2 cells, the baselines one cell line at
  a time. On the STATE files (643k cells) these steps ran on the 8 GB PC, `learned_baselines.py` in ~3 min
  (labels) + seconds (training); the `all` version has 813k cells. MLP training uses CUDA when available
  (`learned.device`, or `DEVICE=cpu|cuda`).
- **Tier B DEGs.** The Tier B comparison panel (`verify_merge.py compare`) counts DEGs without the target
  gene, as `evaluate.py --exclude-target` does. Per-perturbation Jaccard and panel sizes reported in
  `verify/filtered_vs_all/` therefore exclude the CRISPRi target.

## 6. Dashboard (optional)

```bash
uv pip install -r experiments/perturbation_direction/dashboard_requirements.txt
streamlit run experiments/perturbation_direction/dashboard.py --server.port 8501 --server.headless true \
  -- --root outputs/perturbation_direction        # or the eval dir: --root <prepared_dir>/eval
# on your PC:  ssh -L 8501:localhost:8501 <server>   then open http://localhost:8501
```

## Legacy: the STATE-file route

How the frozen `results/replogle_test312_w2ema_3seeds_*` tables were made: train/test rebuilt from STATE's
combined, log-normalised matrix instead of the raw screens. The `filtered` raw version reproduces these
cells, counts and split exactly (gates A1-A4, A5). Kept for reproducing the frozen tables.

```bash
D=_artifacts/datasets/replogle_state; mkdir -p $D
URL=https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/resolve/main/replogle.h5ad
until curl -L -C - -sS --speed-limit 100000 --speed-time 60 -o $D/replogle.h5ad $URL; do sleep 10; done
ls -l $D/replogle.h5ad                                   # must be 22297813040 bytes
python $E/prepare_replogle_state.py --src $D/replogle.h5ad --train-json metadata/replogle_train.json \
  --test-json metadata/replogle_test.json --out-dir $D   # chunked, a few GB RAM
cat $D/prepare_info.json   # recovered/UMI ratio quantiles: min should not sit far below the 1% quantile

TRAIN=$D/replogle_train.h5ad; TEST=$D/replogle_test.h5ad; O=outputs/perturbation_direction/replogle
A=_artifacts/resubmission; S=experiments/perturbation_direction/splits/replogle_val60.json
python $E/audit.py --train-h5ad $TRAIN --test-h5ad $TEST \
  --train-json metadata/replogle_train.json --test-json metadata/replogle_test.json \
  --pert-key gene --control non-targeting --context-key cell_line --target-context hepg2 \
  --size-factor-pkl $A/replogle_log_size_factor_munew.pkl --ckpt $A/replogle_negative_binomial.ckpt
python $E/ground_truth.py --pert-h5ad $TEST --ctrl-h5ad $TRAIN --pert-key gene --control non-targeting \
  --context cell_line=hepg2 --metadata-json metadata/replogle_train.json --min-cells 15 --out-dir $O/truth
python $E/subset_truth.py --truth-dir $O/truth --split-json $S --which test --out-dir $O/truth_test312
python $E/baselines.py --train-h5ad $TRAIN --pert-key gene --control non-targeting \
  --context-key cell_line --target-context hepg2 --truth-dir $O/truth --out-dir $O/baselines
python $E/learned_baselines.py --train-h5ad $TRAIN --truth-dir $O/truth --pert-key gene --control non-targeting \
  --context-key cell_line --target-context hepg2 --out-dir $O/learned
python $E/evaluate.py --truth-dir $O/truth_test312 --exclude-target \
  --pred scLDM_s0=$O/pred/test312_w2_ema_seed0.parquet scLDM_s1=$O/pred/test312_w2_ema_seed1.parquet \
         scLDM_s2=$O/pred/test312_w2_ema_seed2.parquet \
  --baseline knn=$O/baselines/delta_knn.parquet mlp=$O/learned/delta_mlp.parquet \
             generic=$O/baselines/delta_generic.parquet transfer=$O/baselines/delta_transfer.parquet \
             prior=$O/learned/delta_prior.parquet \
  --specific-ref $O/baselines/delta_generic.parquet --out-dir $O/eval_test312
python $E/freeze_results.py --eval-dir $O/eval_test312 --tag replogle_test312_w2ema_3seeds --reference scLDM_s0
```

`--min-cells 15` is applied per half for the split-half ceiling. The HepG2 test perturbations have 30–433
cells, and 86 of them have fewer than 40, so the default of 20 would drop those 86. With 15, all 372 are
kept. The frozen tables also contain calibrated, bilinear and GBDT rows and an unseeded MLP: those
baselines were removed afterwards, so a rerun of this route reproduces every other row but not those.

`evaluate.py` writes two reports:
- `summary.csv` scores the direction of the full effect.
- `summary_specific.csv` scores the direction of the perturbation-specific part: the effect with the
  shared HepG2 response projected out, scaled per perturbation. It tells you whether a model gets more
  right than the shared response. Its `shared_strength_corr` column says how well a model predicts how
  strongly each perturbation triggers that shared response.
