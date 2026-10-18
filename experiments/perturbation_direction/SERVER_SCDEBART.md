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

## 4. What comes back

```
perturbseq/Nadig_HepG2_oursplit/finetuned_model/seed_30/
  Nadig_HepG2_oursplit_test_gene_perturb_predictions.npz   pred_logfc per perturbation
  model_weights/tuned_model_epoch_*.pt
  tuning_loss.txt
```
The `.npz` holds `pred_logfc`, `true_logfc`, `true_ctrl_exp`, `gene_ids` keyed by integer gene
id. Convert it to the `pred/<name>.parquet` format `evaluate.py` expects (perturbation x gene,
symbols) with the same integer->Ensembl->symbol chain used here, then add it with
`--baseline scDEBART=<parquet>`. Phase 2 of the eval work (`scdebart_metrics.py`) is needed
before their EF/F1 numbers mean anything.

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
