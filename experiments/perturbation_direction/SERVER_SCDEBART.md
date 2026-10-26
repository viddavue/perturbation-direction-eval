# Running the scDEBART fine-tune on the GPU server

Goal: score **scDEBART itself** on the same Replogle HepG2 test split as scLDM, so the
"denoised logFC targets" comparison is like-for-like. The released scDEBART predictions are
scored on their own random split, and 241 of our 372 test perturbations are *training*
perturbations there, so they cannot be compared to our numbers. We therefore re-fine-tune
their model with their own script, on our split.

Everything below is prepared by `make_scdebart_job.py`; this file is the runbook.

## What the split looks like

Built from `metadata/replogle_test.json` (our 372 HepG2 test perturbations), restricted to
scDEBART's vocabulary and to genes their HepG2 DE file actually contains:

| | count |
|---|---|
| our genes (universe) | **1,784** of 2,000 (216 are aliases/lncRNAs absent from their 20k vocab) |
| test perturbations | **321** of 372 (22 too-new genes absent from their vocab, 16 absent from their DE file) |
| val perturbations | 30 |
| train perturbations | 1,893 (other dataset perturbations, disjoint from test) |
| test perturbations with HepG2 training cells | **0** (checked against the h5ad `obs`) |

The 321 test perturbations are the comparison set. Any scLDM number reported next to scDEBART
must be recomputed on exactly these 321.

## 1. Get the code and data on the server

```bash
git clone https://github.com/Jieun-Sung/scDEBART   # their repo
cd scDEBART
bash fetch_scdebart_data.sh                        # needs no huggingface_cli; see below
```

`fetch_scdebart_data.sh` (from this repo's `experiments/perturbation_direction/`) downloads the
five files into `./scDEBART_data`, skipping anything already present, and prints the exact
`ln -sfn`/`cp` commands to place them. It tries the `hf`/`huggingface-cli` binary, then
`python -m huggingface_hub`, then plain `curl`/`wget`/`urllib`, so it works on a server with
none of the HF tooling installed.

~11 GB of downloads; the DE h5 is the bulk of it. Their `data/` files must end up in `./data`
(the script hardcodes `datadir = "./data"`).

**Note:** `hvg_geneids_5000.pkl` is *not* published for Nadig_HepG2 (only `norman` ships one),
so their finetune script cannot run as-is. Our generated `hvg_geneids_1784.pkl` fills that role.

Environment: `conda env create -f scDEBART.yml`, or pip with `torch==2.5.1`, `deepspeed`,
`scvi-tools`, `scanpy`, `h5py`, `joblib`.

## 2. Copy the prepared job out of this repo

The generator was run locally and its output is:

```
hvg_geneids_1784.pkl        gene universe for --num_HVG 1784
train_val_test_30.pkl       train/val/test, Ensembl IDs, the format their script reads
split.json                  provenance, dropped genes/perturbations, leakage check
job.sh                      the deepspeed command
```

Copy them next to their repo and put the gene list where their script looks for it:

```bash
cp hvg_geneids_1784.pkl  perturb_seq/Nadig_HepG2/DE_results/
```

## 3. Run

```bash
NGPUS=4 bash job.sh          # DeepSpeed ZeRO-2, bf16, ~0.75 GB model
```

Runtime is dominated by the pretrain-sized BART forward over 1,784 genes per sample; the
training set is 1,893 perturbations, so expect a few hours on 4 GPUs, less if the GPUs are
recent. Early stopping is on val MSE (patience 5). Set `--num_epochs` in `job.sh` if you want
a hard cap.

### fp16 instead of bf16 (Volta / TITAN V)

`job.sh` follows their bf16 config, which is correct on Ampere+ but wasteful on Volta
(compute capability 7.0). Measured on one TITAN V, 2048x2048 matmul:

| dtype | time | throughput |
|---|---|---|
| fp16 | 0.27 ms | 64,297 GFLOP/s |
| fp32 | 1.48 ms | 11,583 GFLOP/s |
| bf16 | 2.05 ms | 8,361 GFLOP/s |

bf16 is emulated, so it is ~7.7x slower than fp16. Note `torch.cuda.is_bf16_supported()`
returns True here because it defaults to `including_emulation=True`; with
`including_emulation=False` it is False. On Volta use fp16.

The server variant therefore uses:
- `scDEBART_job/finetune_perturbseq_fp16.py` -- their script with `bfloat16` replaced by
  `float16` (12 sites), the original left untouched;
- `scDEBART_job/deepspeed_config_fp16.json` -- `fp16.enabled` with dynamic loss scaling,
  micro-batch 2 x 2 accumulation x 5 GPUs (same effective batch as their 4 x 5 x 1);
- `scDEBART_job/run_fp16.sh` -- activates the cloned env, sets `CUDA_VISIBLE_DEVICES`,
  points `--de_result_dir` at a directory holding a symlink to the 10 GB DE file.

The DE file is symlinked, not copied. Run it in tmux so it survives disconnects:
`tmux attach -t scdebart`.

## 4. What came back (our run)

```
scDEBART_job/out/predict_best/
  Nadig_HepG2_oursplit_test_gene_perturb_predictions.npz   321 perturbations x 1784 genes
scDEBART_job/out/train_val_test_30/
  model_weights/tuned_model_epoch_*.pt   best: epoch_7 (val MSE 0.146439)
  tuning_loss.txt                        13 epochs, then early stopping
```

The `.npz` holds `pred_logfc`, `true_logfc`, `true_ctrl_exp`, `gene_ids` keyed by integer gene
id. Convert it with `scdebart_predictions.py` (integer -> Ensembl -> symbol) and score it with
`--baseline scDEBART=<parquet>`. Score it against `--truth-dir truth_scvi`, the target it was
trained to predict; sign is comparable to the default target, magnitude is not.

## Patches required to run their code at all

Their released fine-tune cannot run as published. Nine problems, in the order they appeared.
Each is a real defect or a Volta/no-CUDA-toolkit limitation, not a nicety:

1. **Missing argparse entries.** `DEGTransformer` reads `args.embedding_dim`, `nhead`,
   `num_encoder_layer`, `num_decoder_layer`, `dropout`, and the loaders read `args.batch_size`,
   but their finetune parser defines none of them (they exist only in
   `Pretrain/pretrain_scDEBART.py`). Added with the pretrain defaults (768 / 12 / 8 / 4 / 0.2);
   `batch_size` must equal `train_micro_batch_size_per_gpu`.
2. **`hvg_geneids_{n}.pkl` is not published for Nadig_HepG2** (only `norman` ships one), and it
   must contain **integer** gene ids, not Ensembl strings -- their dataset compares it against
   the h5's integer ids. Ours are generated by `make_scdebart_job.py`.
3. **fp16 instead of bf16.** Volta (TITAN V, compute 7.0) has no bf16 hardware: measured
   8.4 TFLOP/s bf16 vs 64.3 TFLOP/s fp16. Note `torch.cuda.is_bf16_supported()` returns True
   here only because it defaults to `including_emulation=True`.
4. **`Tensor.float16()` does not exist** -- the method is `.half()`. A blind bf16->fp16 rename
   breaks the two dataset casts.
5. **DeepSpeed FusedAdam/CPUAdam need a CUDA toolkit** to JIT-compile, and this machine has no
   `nvcc`. Passing a plain `torch.optim.AdamW` to `deepspeed.initialize` avoids it; ZeRO-2 still
   shards the optimizer.
6. **The DE file stores ~50 seed replicates per perturbation** (101,850 rows / 2,036
   perturbations). Training on all of them makes an epoch ~50x longer, so the dataset keeps one
   row per perturbation -- consistent with the one-seed target choice.
7. **Profiles with no gene in the modelled set** produce an all-padding batch, and the top-k
   correlation loss then indexes an empty tensor. Such rows are dropped.
8. **Validation metrics were never all-reduced.** `avg_val_mse` is averaged over each rank's own
   shard, so ranks disagreed on `did_improve`, patience counters diverged, and one rank exited
   while the others blocked -> NCCL watchdog killed the run at the same point twice. Fixed with
   `all_reduce(AVG)`. Their logged val MSE was rank 0's slice, not the true score.
9. **Test predictions were only rank 0's shard.** The first successful run produced a file with
   65 of 321 perturbations. Fixed with `all_gather_object` + dedupe (sampler padding repeats a
   few perturbations).

Also: the DE file is read from **local disk** (`/tmp/fleischa_de`) with `num_workers=0`. Five
ranks x 4 forked workers reading HDF5/SWMR over NFS is a deadlock risk; it was not the cause of
the failures above, but it removes a variable.

## Results (our split, 1,784 genes, scVI target, paired n=215)

**Conditioning audit warning (supersedes the interpretation below):** the actual prediction
archive contains its target gene among the input tokens for only **51 of 321** perturbations.
For the other **270**, the code's -10 perturbation indicator is entirely zero. Restricting the
input sequence to the scored HVGs therefore removed the identity signal for most examples.
The table below describes this restricted-input run; it is NOT a valid basis for comparing
scDEBART's perturbation-prediction capability with scLDM. A corrected reproduction must keep
the target gene in the input sequence even when that gene is excluded from the scored output.
Run `scdebart_conditioning_audit.py --npz <archive> --out-dir <audit>` to reproduce the check.

| method | balanced acc | AUROC | Spearman on DEGs |
|---|---|---|---|
| kNN | 0.923 | 0.994 | 0.882 |
| scLDM w1 | 0.913 | 0.981 | 0.867 |
| MLP | 0.898 | 0.983 | 0.827 |
| bilinear | 0.891 | 0.975 | 0.811 |
| calibrated | 0.877 | 0.962 | 0.796 |
| generic | 0.846 | 0.942 | 0.810 |
| gbdt | 0.845 | 0.965 | 0.817 |
| **scDEBART** | **0.844** | 0.934 | 0.792 |
| transfer | 0.787 | 0.899 | 0.673 |

Paired Wilcoxon on balanced accuracy: scDEBART vs `generic` +0.001 (p=0.48, indistinguishable);
vs scLDM -0.068 (p=1e-13); vs kNN -0.089 (p=7e-23).

**Do not over-read this.** Deviations that all cut against scDEBART: one seed replicate per
perturbation instead of ~50, fp16, 13 epochs, 1,784 genes instead of 5,000, and -- importantly --
it is scored against *our* scVI target rather than the file it trained on, with no split-half
ceiling for that target. Its weakness on weak perturbations (0.646 at 5-19 DEGs) is consistent
with under-training. The supportable claim is "our reproduction does not beat scLDM", not
"scDEBART is weak".

## Pitfalls

- **`--pert_type INH`** is correct: Nadig HepG2 is CRISPRi. Their `OE` branch is for Norman.
- **`--num_HVG 1784` must match the `hvg_geneids_1784.pkl` you copied.** That file is ours; no
  `hvg_geneids_5000.pkl` exists for Nadig_HepG2, so their script fails on `pickle.load` unless
  our list is in `--de_result_dir`. A stale 5000-name list would silently train on other genes.
- **Their script reads only `de_masked_high_zero_logfc_to_zero.h5`**, not the small
  `DE_true_logFC...pkl`. We needed the latter only to learn the gene/perturbation universe.
- **`seed` here is a split seed, not a training seed.** Their three published seeds are three
  *splits*; our split is fixed, so a single run is enough unless we want training variance.
- **Scoring must drop the 51 test perturbations whose target gene is in the measured gene set**,
  or leave the target-gene exclusion on (`--exclude-target`), to match how scLDM is scored.

## Corrected run: the token bug did not explain the gap

The restricted vocabulary (1,784 scoring genes) excluded 270 of the 321 test targets, so those
profiles carried an all-zero perturbation indicator. All 321 targets are present in the DE file's
9,297 gene columns, so the fix is a vocabulary of scoring genes plus every perturbation target
that the DE file contains: **3,474 genes**, covering 321/321 test, 54/60 validation and
1,662/1,863 training targets.

Configuration: micro-batch 1 x accumulation 7 x 3 GPUs = 21 effective (their 20), fp16, 40 epochs
with patience 5 -> stopped at epoch 21, ~10 min/epoch.

Audit of the new archive: **321/321 target tokens present** (old run: 51/321).

| method | n | balanced accuracy | AUROC |
|---|---:|---:|---:|
| scLDM (w2 + EMA, seed 0) | 167 | **0.931** | 0.990 |
| scDEBART, corrected vocabulary | 167 | 0.768 | 0.885 |
| scDEBART, old vocabulary (bug) | 167 | 0.782 | 0.889 |

Paired on the common 164 perturbations: scLDM - corrected **+0.130 [+0.116, +0.159], p = 1.3e-18**;
corrected - old **+0.002 [0.000, +0.018], p = 0.13**.

Two conclusions:

1. **The conditioning bug was real but not causal.** Fixing it changed the score by +0.002, which is
   not distinguishable from zero. The earlier "compromised comparison" caveat can be retired: the
   deficit is a property of the model as reproduced, not of the missing indicators.
2. **The gap is uniform across strengths.** scLDM scores 0.858 / 0.894 / 0.924 / 0.972 over the
   5-19 / 20-99 / 100-499 / 500+ DEG bins; corrected scDEBART scores 0.661 / 0.778 / 0.806 / 0.765.
   It does not catch up on strong perturbations, and it gets slightly worse there.

Remaining caveats, unchanged and not in scDEBART's favour: a single training seed, 21 epochs, our
gene universe, and a sign-level target while the model predicts log2 fold change directly. A
definitive claim about scDEBART would need their 50-resample targets and full schedule.
