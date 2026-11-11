# Parse1M: cytokine response on CD4 Naive cells

A separate report from Replogle. Different perturbation modality (cytokine stimulation, not
CRISPRi knockdown), different control semantics (`PBS`, not a non-targeting guide), and a
different gene overlap. Pooling the two would hide exactly the differences worth seeing.

## Data audit and its two limitations

Source: `pbmc_parse` (Klein 2025, figshare 28589774), one 13.1 GB h5ad, md5
`d89559875a6c75ab37e0adf84c6e177d`, 9,697,974 cells x 2,000 genes. This is **not** the
`CD4_Naive_holdout` train/test pair the configs reference; those are unpublished.

1. **Gene vocabulary mismatch.** The public file's 2,000 genes are its own HVGs. Only **1,313**
   of the model's 2,000-gene vocabulary are present; 687 are missing (`GCLC`, `LAP3`,
   `HS3ST1`, `AOC1`, ...). Ground truth is therefore built on 1,313 genes. Generation still
   covers the model's full 2,000, and evaluation intersects.
2. **Counts are downsampled.** `X` holds raw non-negative integers, but median row sum is
   **197** UMIs while the file's own `obs/gene_count` is 1,617 and `mread_count` 5,241. It is a
   reduced-depth cellflow export. After restricting to the 1,313 model genes the control median
   library size is **102.5**. Both arms are equally shallow, so the comparison is internally
   consistent, but the panel is much weaker than Replogle.

The model's label vocabularies match the data exactly: 18 cell types, 91 cytokines, and a
`treatment` column of `PBS` / `cytokine`.

## Protocol

- Cell type `CD4 Naive`; control `PBS`; perturbation key `cytokine`.
- Panel: the 27 held-out cytokines from `metadata/parse1m_test.json` (all 27 have >=60 cells;
  CD4 Naive has 1,725,983 cells in the source).
- `prepare_parse1m.py` keeps up to 2,000 cells per condition (seed 0), preserving the model's
  gene order, and writes a derived metadata json containing only the genes present in the data.
  **That derived json must never be passed to `generate.py`**, which needs the full vocabulary
  ordering.
- Ground truth: `ground_truth.py --control PBS --pert-key cytokine --min-cells 15`.
- **`--exclude-target` is deliberately not used.** The perturbation is a cytokine, so excluding
  a "target gene" would wrongly drop the gene named after the cytokine (e.g. `IL-2`).
- Generation: `czb_cd4_naive_holdout_negative_binomial` checkpoint, guidance 1, raw weights,
  128 cells per condition, library size 102.5 (the observed control median, so the generated
  depth matches the measured depth), seed 0, five GPUs.

## Results

24 of 27 cytokines have >=5 reference DEGs and enter the medians; median DEGs per cytokine is
20 (min 2, max 170).

| method | n | balanced acc | sign acc | AUROC | Spearman on DEGs |
|---|---:|---:|---:|---:|---:|
| scLDM w1 | 24 | **0.954** | 0.937 | 0.997 | 0.834 |
| scLDM w1 (shuffled) | 24 | 0.581 | 0.542 | 0.548 | 0.060 |
| noise ceiling (split-half) | 24 | 1.000 | 1.000 | 1.000 | 0.960 |

By reference strength (balanced accuracy): 0.942 at 5-19 DEGs (10 perturbations), 0.977 at
20-99 (10), 0.937 at 100-499 (4).

## How to read this

- **Specificity holds.** Shuffled predictions drop to ~0.58, so the direction calls carry
  cytokine-specific information rather than a generic stimulated/unstimulated shift.
- **The ceiling is saturated.** A split-half replicate scores 1.000 on direction, so on this
  panel direction is essentially noiseless and scLDM's 0.954 is a real gap but a narrow
  metric — it cannot separate methods that are all "mostly right". The Replogle panel has a
  ceiling of ~0.97, which is more discriminating.
- **Do not compare the medians across datasets.** Parse1M rests on 1,313 genes, 102.5 UMIs and
  20 median DEGs; Replogle rests on 2,000 genes, ~4,850 UMIs and hundreds of DEGs. The
  protocols differ in the data, not just the biology.
- **No baselines yet.** Cross-context transfer needs more than one cell type; a generic
  shared-response baseline for cytokines is the natural next comparator, alongside the
  kNN/MLP baselines used for Replogle.

## Reproduce

```bash
B=/system/user/studentwork/fleischa/scldm_generation_20261020   # server job dir
python experiments/perturbation_direction/prepare_parse1m.py \
  --src <figshare h5ad> --metadata-json metadata/parse1m_train.json \
  --test-json metadata/parse1m_test.json --cell-type "CD4 Naive" --control PBS \
  --max-cells 2000 --out-dir $B/outputs/perturbation_direction/parse1m/prepared
python experiments/perturbation_direction/ground_truth.py \
  --pert-h5ad .../prepared/parse1m_cd4naive.h5ad --ctrl-h5ad .../prepared/parse1m_cd4naive.h5ad \
  --layer X --pert-key cytokine --control PBS --context "cell_type=CD4 Naive" \
  --perturbations <27 cytokines> --metadata-json .../prepared/parse1m_genes_available.json \
  --min-cells 15 --out-dir .../parse1m/truth
python experiments/perturbation_direction/generate.py \
  --config _artifacts/resubmission/czb_cd4_naive_holdout_negative_binomial.yaml \
  --ckpt _artifacts/resubmission/czb_cd4_naive_holdout_negative_binomial.ckpt \
  --metadata-json metadata/parse1m_train.json --conditions .../parse1m/conditions.csv \
  --n-cells 128 --batch-size 32 --library-size 102.5 --guidance 1 --seed 0 --gpus 0,1,2,3,4 \
  --out .../parse1m/pred/w1.parquet
python experiments/perturbation_direction/evaluate.py \
  --truth-dir .../parse1m/truth --pred scLDM_w1=.../parse1m/pred/w1.parquet --out-dir .../parse1m/eval
```
