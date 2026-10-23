# Running the Replogle evaluation on a GPU server

Generation is the expensive step. Everything else (data preparation, ground truth, baselines, evaluation)
runs on CPU in minutes to about an hour, on the server or on your PC.

Paths below are relative to the repo root. `E=experiments/perturbation_direction`.

## 1. Code and environment (once)

```bash
git clone <your-remote> scldm && cd scldm && git checkout perturbation-direction-eval
# or: rsync -a --exclude .venv --exclude _artifacts --exclude outputs <pc>:PycharmProjects/perturbation-direction-eval/ scldm/

# uv: curl -LsSf https://astral.sh/uv/install.sh | sh    (if not installed)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install --override experiments/perturbation_direction/overrides.txt -e ".[test]" scanpy pyarrow h5py \
  "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git@a3720244bf0cfc4950e89ad583ceb0f86ed07199"
python -m pytest -q tests            # expect: 30 passed, 1 skipped
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

`overrides.txt` pins anndata, scvi-tools, scanpy and cellarium-ml to versions that work together; the
repo's own pins conflict. torch 2.7 installs with CUDA 12.6 wheels by default. If the last command prints
`0` GPUs or fails, the server's NVIDIA driver is too old for those wheels. In that case, install the
torch 2.7.0 build for the CUDA version that `nvidia-smi` reports, for example
`uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118`.

## 2. Model checkpoint (~1 GB)

```bash
python $E/fetch_artifacts.py --match replogle --exclude gaussian     # -> _artifacts/resubmission/
```

## 3. Generation (multi-GPU)

The conditions come straight from the test metadata (HepG2 control + 372 test perturbations), so this
step can start right away, in parallel with steps 4 and 5.

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

## 4. Data (22 GB download, CPU)

```bash
D=_artifacts/datasets/replogle_state; mkdir -p $D
URL=https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/resolve/main/replogle.h5ad
until curl -L -C - -sS --speed-limit 100000 --speed-time 60 -o $D/replogle.h5ad $URL; do sleep 10; done
ls -l $D/replogle.h5ad                                   # must be 22297813040 bytes
python $E/prepare_replogle_state.py --src $D/replogle.h5ad --train-json metadata/replogle_train.json \
  --test-json metadata/replogle_test.json --out-dir $D   # chunked, a few GB RAM
cat $D/prepare_info.json   # recovered/UMI ratio quantiles: min should not sit far below the 1% quantile
```

## 5. Audit, ground truth and baselines (CPU)

```bash
TRAIN=$D/replogle_train.h5ad; TEST=$D/replogle_test.h5ad
python $E/audit.py --train-h5ad $TRAIN --test-h5ad $TEST \
  --train-json metadata/replogle_train.json --test-json metadata/replogle_test.json \
  --pert-key gene --control non-targeting --context-key cell_line --target-context hepg2 \
  --size-factor-pkl $A/replogle_log_size_factor_munew.pkl --ckpt $A/replogle_negative_binomial.ckpt
python $E/ground_truth.py --pert-h5ad $TEST --ctrl-h5ad $TRAIN --pert-key gene --control non-targeting \
  --context cell_line=hepg2 --metadata-json metadata/replogle_train.json --min-cells 15 --out-dir $O/truth
python $E/baselines.py --train-h5ad $TRAIN --pert-key gene --control non-targeting \
  --context-key cell_line --target-context hepg2 --truth-dir $O/truth --out-dir $O/baselines
```

`--min-cells 15` is applied per half for the split-half ceiling. The HepG2 test perturbations have 30–433
cells, and 86 of them have fewer than 40, so the default of 20 would drop those 86. With 15, all 372 are
kept.

Optional: trained classifiers (prior / bilinear / MLP / GBDT). This takes about 10 minutes on CPU for the labels, plus a
short GPU training run:

```bash
python $E/learned_baselines.py --train-h5ad $TRAIN --truth-dir $O/truth --pert-key gene --control non-targeting \
  --context-key cell_line --target-context hepg2 --out-dir $O/learned
```

Add them to step 6 with `--baseline mlp=$O/learned/delta_mlp.parquet gbdt=$O/learned/delta_gbdt.parquet ...`.

## 6. Evaluate (CPU, seconds)

```bash
python $E/evaluate.py --truth-dir $O/truth --exclude-target \
  --pred scLDM_w1=$O/pred/w1.parquet scLDM_w1_ema=$O/pred/w1_ema.parquet \
         scLDM_w2=$O/pred/w2.parquet scLDM_w4=$O/pred/w4.parquet scLDM_w1_counts=$O/pred/w1_sample.parquet \
  --baseline transfer=$O/baselines/delta_transfer.parquet generic=$O/baselines/delta_generic.parquet \
             calibrated=$O/baselines/delta_calibrated.parquet knn=$O/baselines/delta_knn.parquet \
             transfer_k562=$O/baselines/delta_transfer_k562.parquet transfer_rpe1=$O/baselines/delta_transfer_rpe1.parquet \
             transfer_jurkat=$O/baselines/delta_transfer_jurkat.parquet \
  --specific-ref $O/baselines/delta_generic.parquet --out-dir $O/eval
```

This writes two reports:
- `summary.csv` scores the direction of the full effect.
- `summary_specific.csv` scores the direction of the perturbation-specific part: the effect with the
  shared HepG2 response projected out, scaled per perturbation. It tells you whether a model gets more
  right than the shared response. Its `shared_strength_corr` column says how well a model predicts how
  strongly each perturbation triggers that shared response.

To evaluate on your PC instead, copy back only the small files:
`rsync -a server:scldm/outputs/perturbation_direction/replogle/pred/*.parquet outputs/perturbation_direction/replogle/pred/`.
Each one is about 6 MB.

## 7. Dashboard (optional)

```bash
uv pip install -r experiments/perturbation_direction/dashboard_requirements.txt
streamlit run experiments/perturbation_direction/dashboard.py --server.port 8501 --server.headless true \
  -- --root outputs/perturbation_direction
# on your PC:  ssh -L 8501:localhost:8501 <server>   then open http://localhost:8501
```
