# VCC-oriented evaluation on Replogle

## Scope and status

No submission to the Virtual Cell Challenge is made. The official implementation is pinned to
`ArcInstitute/cell-eval2@a2630dcbda95fc0b819bf841e56d9561d9c1e1da` (0.19.0).
The scripts keep the existing training environment and default truth unchanged.

- `scdebart_conditioning_audit.py`: actual prediction-token coverage, including replicated profiles.
- `vcc_profiles.py`: official cosine/midrank retrieval kernel applied to supplied effect matrices.
  **Not** the official counts comparator or an anchored VCC score.
- `denoising_ablation.py`: fixed-prediction 2x2 estimator/filtering comparison with a paired cohort.
- `vcc_pilot.py`: fixed small panel, independent real-cell halves, and generation conditions.
- `vcc_cells.py`: all six raw metrics via the official `vcc2026` preset, explicit CPU Scanpy DE.
- `generate.py --save-cells`: opt-in genuine NB cell counts, in addition to the existing means.

The CPU backend is an explicit experimental choice, not a claim of parity with Challenge GPU DE.
All thresholds and the resolved preset are saved in the cell-run manifest. Reference-normalized
Challenge scores require their context-specific anchors; do not average these raw metrics into
an "overall" score. MSE uses the upstream panel-wide ratio-of-sums aggregation, not an average
of per-perturbation ratios.

## Optional environment

```bash
uv venv --python 3.11 .venv-vcc
uv pip install --python .venv-vcc/bin/python -r experiments/perturbation_direction/vcc_requirements.txt pyarrow pytest
```

Install the pinned requirement rather than an unpinned package. The manifests record the
expected source commit and installed package version. Runtime input hashes, exact genes,
panel and random seeds are recorded where applicable. The source commit is an expectation
from the installation recipe, not a runtime verification of editable source changes.

## 1. Conditioning audit (completed)

```bash
python experiments/perturbation_direction/scdebart_conditioning_audit.py \
  --npz outputs/perturbation_direction/scdebart_job/predictions/Nadig_HepG2_oursplit_test_gene_perturb_predictions.npz \
  --out-dir outputs/perturbation_direction/replogle/conditioning_audit
```

**51/321** predictions include their target token; **270/321 do not**. Those 270 get no explicit
perturbation indicator. Baseline inputs may differ between profiles, so this does not imply
all predictions are exactly identical. It does invalidate a clean capability comparison with
scLDM. Fixing the input sequence (include the target separately from the scoring gene universe)
is required before another scDEBART comparison. No corrected training job has been launched.

## 2. Fixed-panel effect retrieval (completed)

Inputs must be perturbation-by-gene **effects**, not expression profiles. All methods are
restricted to one common panel and gene universe; no DEG-count filter is used. PDS excludes
all panel target genes, uses midrank ties and denominator n-1. Every candidate is retained
when scoring a shuffled query. Zero-effect queries score 0.5, including numerical-zero
shared-response residuals. Top-1/top-5 diagnostics give fractional credit to tied matches.

```bash
O=outputs/perturbation_direction/replogle
.venv-vcc/bin/python experiments/perturbation_direction/vcc_profiles.py \
  --truth $O/truth/delta_true.parquet \
  --method scLDM=$O/restricted/scLDM_w1.parquet knn=$O/restricted/knn.parquet \
    generic=$O/restricted/generic.parquet scDEBART=$O/restricted/scDEBART.parquet \
  --shared-ref $O/baselines/delta_generic.parquet \
  --out-dir $O/pds_effect_diagnostic
```

For this panel: 321 perturbations, 1,784 common genes minus 51 panel targets = 1,733 features.
The reference is the existing mean-log difference target. Results:

| method | full effect mean retrieval | shared-response residual |
|---|---:|---:|
| scLDM | 0.675 | 0.803 |
| kNN | 0.592 | 0.811 |
| generic | 0.500 | 0.500 |

These are **effect-space diagnostic ranks**, not official VCC scores. Do not conclude that
scLDM beats kNN generally: the ranking depends on the metric and representation. scDEBART
results are conditioning-compromised (above). `paired_differences.csv` contains paired mean
differences with bootstrap intervals over query perturbations on the fixed candidate panel.
This quantifies variability across panel queries, not uncertainty about arbitrary new panels.

## 3. Denoising versus filtering (completed)

The previous scVI targets only saved detection-masked effects, so that run could not isolate
filtering. `scvi_targets.py` now additionally saves unmasked logFC and both detection matrices.
The one-seed rerun is stored separately in `truth_scvi_ablation/`.

```bash
O=outputs/perturbation_direction/replogle
python experiments/perturbation_direction/denoising_ablation.py \
  --train-h5ad _artifacts/datasets/replogle_state/replogle_train.h5ad \
  --test-h5ad _artifacts/datasets/replogle_state/replogle_test.h5ad \
  --truth-dir $O/truth --scvi-dir $O/truth_scvi_ablation \
  --method scLDM=$O/restricted/scLDM_w1.parquet knn=$O/restricted/knn.parquet \
    generic=$O/restricted/generic.parquet --out-dir $O/denoising_ablation
```

The observed estimator is log2 ratio of arithmetic mean per-cell CP10k, pseudocount 0.001
CP10k units. scVI uses posterior median logFC, not the same estimator with noise removed.
The contrast therefore still includes estimator differences. Filtering is an explicit mask
(detection >=0.3 in both arms), not replacement with zero. Original reference DEGs and fixed
predictions are used throughout; the target gene is excluded. The paired summary retains
246 perturbations with >=5 scored DEGs and defined balanced accuracy in every arm.

| estimator | detection filter | scLDM balanced accuracy | kNN |
|---|---|---:|---:|
| observed | off | 0.913 | 0.934 |
| observed | on | 0.922 | 0.933 |
| scVI | off | 0.906 | 0.929 |
| scVI | on | 0.918 | 0.928 |

On this cohort filtering improves scLDM's median direction score more than changing estimator.
This does not establish that denoising alone helps or hurts. Magnitude metrics are not applied
to our existing mean-log effects or classifier probabilities as though they were log2-FC.

## 4. Cell-level DE pilot

`vcc_pilot.py` selects three perturbations per strength bin with >=64 real cells, then splits
32 real cells per arm and per perturbation. Controls are also disjoint. Selection uses observed
strength solely for descriptive stratification, not test-driven tuning. `manifest.json` records
the exact cell IDs. `conditions.csv` includes the control and all nine perturbations.

```bash
O=outputs/perturbation_direction/replogle
python experiments/perturbation_direction/vcc_pilot.py \
  --train-h5ad _artifacts/datasets/replogle_state/replogle_train.h5ad \
  --test-h5ad _artifacts/datasets/replogle_state/replogle_test.h5ad \
  --truth-dir $O/truth --out-dir $O/vcc_pilot
.venv-vcc/bin/python experiments/perturbation_direction/vcc_cells.py \
  --real $O/vcc_pilot/real.h5ad --pred $O/vcc_pilot/replicate.h5ad \
  --n-cells 32 --out-dir $O/vcc_pilot/replicate_metrics
```

This real-half diagnostic validates the metric pipeline. Under the VCC preset the real control
is used as the predicted-expression comparator too; although the input halves are disjoint,
it is **not** an independently controlled, reference-normalized noise ceiling.

Initial 32-cell pilot (one split and generation seed; descriptive, not a ranking claim):

| raw metric | scLDM w1 | real-half comparator | scLDM finite coverage |
|---|---:|---:|---:|
| PDS cosine | 0.792 | 0.847 | 9/9 |
| expression MSE ratio (lower better) | 1.164 | 0.131 | panel-wide |
| DE logFC NMAE (lower better) | 0.655 | 1.613 | **1/9** |
| fidelity yield | 0.800 | 0.972 | **3/9** |
| reach | 1.000 | 1.000 | **3/9** |
| Jaccard | 0.723 | 0.621 | 9/9 |

These averages have different scoreable cohorts. In particular, high Jaccard partly reflects
empty significant sets on both sides (upstream assigns those 1), not successful discovery.
NMAE is defined only on NIFK in this small pilot. Its apparently better model score than the
real-half comparator is not evidence of superiority. `coverage.csv` counts emitted, finite
and unscored rows so such metrics cannot silently masquerade as full-panel measurements.

Generate genuine NB count samples for the same panel (local GPU or a separately approved
server job):

```bash
python experiments/perturbation_direction/generate.py \
  --config _artifacts/resubmission/replogle_negative_binomial.yaml \
  --ckpt _artifacts/resubmission/replogle_negative_binomial.ckpt \
  --metadata-json metadata/replogle_train.json --conditions $O/vcc_pilot/conditions.csv \
  --n-cells 32 --batch-size 16 --library-size 4664.5 --guidance 1 --seed 0 \
  --save-cells --out $O/vcc_pilot/scLDM_w1.parquet
.venv-vcc/bin/python experiments/perturbation_direction/vcc_cells.py \
  --real $O/vcc_pilot/real.h5ad --pred $O/vcc_pilot/scLDM_w1.h5ad \
  --n-cells 32 --out-dir $O/vcc_pilot/scLDM_metrics
```

`--save-cells` is opt-in, requires an NB head and a positive library size, writes one atomic
cell file per condition, and merges to `.h5ad`. Old profile-only runs remain resumable in
their original mode; use a new `--out` for cell-saving runs. The main profile-only workflow is
unchanged. The merged file is in-memory: use this option for the small pilot, not an unbounded
full-panel run. Do not repeat mean rows as cells or invent DE p-values for profile-only models.

## Next queued experiments (not launched on the server)

1. Add control-vs-control null groups and repeated real-cell splits; interpret undefined DE
   metrics explicitly, including the number of scoreable perturbations for each metric.
2. Freeze the panel across cell-count arms (16/32/64) based on the largest requested count.
   Repeat generation/sampling seeds; stratify DE yield, Jaccard, fidelity and reach by strength.
3. Select guidance/EMA on validation perturbations only; test the selected settings once.
4. Correct the scDEBART input-token coverage and restrict supervised training to the actual
   scLDM training contexts, not just the complement of mapped test labels, before retraining.

## 5. Cell-level DE on the full test panel (test-312)

Generated with `--save-cells` at 128 cells per condition from the validation-selected config
(w2 + EMA), then scored against real cells with cells equalised **per perturbation** (both arms
take the same n; n varies 30-64 because real perturbations hold 30-433 cells). Three independent
subsamples.

| raw metric | mean | min-max (3 draws) | perturbations scored |
|---|---:|---|---:|
| PDS cosine | 0.793 | 0.786-0.800 | 312/312 |
| expression MSE ratio (lower better) | 0.864 | 0.771-1.015 | panel-wide |
| DE logFC NMAE (lower better) | 0.681 | 0.649-0.711 | **134/312** |
| DE direction fidelity (yield) | 0.686 | 0.668-0.715 | 251/312 |
| DE direction reach | 0.841 | 0.788-0.876 | 195/312 |
| DE significance Jaccard | 0.303 | 0.284-0.314 | 312/312 |

Coverage is much better than the 32-cell pilot (NMAE 1/9 -> 134/312, fidelity 3/9 -> 251/312),
but NMAE still scores fewer than half the panel: at these cell counts the reference DE test does
not flag enough genes per perturbation to gate the metric. Treat NMAE as a partial-cohort number.

**Jaccard (0.303) is the weak point.** Direction agrees far better than the significant-gene set:
the model gets signs right on genes the reference calls, but recovers only about a third of the
overlap of responding genes. That is a discovery failure, not a direction failure, and it is the
clearest remaining gap.

### Anchored against a real-vs-real split of the same panel

`make_real_halves.py` splits every group (controls included) into disjoint halves — 13,075 /
13,231 cells — and `vcc_cells.py` scores one half against the other with the same 64-cell
equalisation. That is the best a 64-cell sample of the same experiment can do, so it is the
reference point, not a competitor.

| raw metric | scLDM | real-half | fraction of replicate |
|---|---:|---:|---:|
| PDS cosine | 0.793 | 0.811 | 0.98 |
| DE direction fidelity | 0.686 | 0.728 | 0.94 |
| DE direction reach | 0.841 | 0.987 | 0.85 |
| DE significance Jaccard | 0.303 | 0.526 | **0.58** |
| DE logFC NMAE (lower better) | 0.681 | 0.499 | 0.73 |
| expression MSE ratio (lower better) | 0.864 | -0.078 | far worse |

Reading:

- **Direction is essentially at replicate quality.** PDS 0.98 and fidelity 0.94 of what a
  disjoint half of the real experiment achieves. Whatever else fails, the model orders and
  signs genes close to the noise floor of the measurement.
- **Discovery is the failure.** Jaccard reaches 58% of the replicate: the model finds a third of
  the significant-gene overlap, and the replicate shows 0.526 is attainable at this cell count.
- **Magnitudes and absolute expression lag.** NMAE 0.681 against 0.499, and expression MSE at
  0.86 of the real effect size against roughly zero for a replicate.
- The real-half numbers are **not** a ceiling in the formal sense: two disjoint 64-cell samples
  are a pessimistic stand-in for full-depth measurement, so "0.58 of replicate" is a lower bound
  on the model's shortfall relative to a perfect prediction.
