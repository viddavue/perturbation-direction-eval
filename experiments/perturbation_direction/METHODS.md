# Methods: Replogle perturbation direction evaluation

How scLDM predicts the direction of CRISPRi perturbation effects on the Replogle-Nadig screens.

## (a) Data

**Sources.** The four raw single-cell screens of the Replogle-Nadig CRISPRi dataset are downloaded from
their original repositories:

| screen | source | file | md5 |
|---|---|---|---|
| HepG2 | GEO GSE264667 | `GSE264667_hepg2_raw_singlecell_01.h5ad` | `dfc676e4...` |
| Jurkat | GEO GSE264667 | `GSE264667_jurkat_raw_singlecell_01.h5ad` | `dae9fab2...` |
| K562 essential | figshare 20029387 (file 35773219) | `K562_essential_raw_singlecell_01.h5ad` | `4f1122ce...` |
| RPE1 | figshare 20029387 (file 35775606) | `rpe1_raw_singlecell_01.h5ad` | `6a2a9d0d...` |

The K562 screen is the **essential** screen, not the genome-wide (GWPS) screen. GWPS lacks 40 of the
checkpoint's 2,000 genes; all 188,590 K562 cells of the STATE file match the essential file.

**Why raw.** The scLDM checkpoint was trained on STATE's processed Replogle-Nadig data. The published
STATE combined file (`arcinstitute/Replogle-Nadig-Preprint`, 643,413 cells × 6,642 genes; the genes are
the Ensembl-ID intersection of the four screens) stores log1p-normalised values, from which the earlier
pipeline (`prepare_replogle_state.py`) recovered integer counts. Rebuilding from the original screens
turns that provenance into a check: the rebuild must reproduce STATE's cells, counts and split exactly
(the identity gates below), and it also gives an unfiltered version that STATE does not contain. Both
routes give the same counts on the same cells, so the normalisation does not depend on the route.

**Audit.** `audit_raw_screen.py` checks each raw screen: integer counts, non-negative, ≥ 300
non-targeting controls, no zero-library cell, every model gene present and not duplicated, and the
label vocabulary consistent with the training metadata. It passes on all four screens.

**Knockdown filter.** STATE's cell selection is reproduced exactly by `knockdown_filter.py`, a port of
cell-load's `filter_on_target_knockdown` with the semantics of cell-load commit
`d84018d113daa6fa64fc0b3b78fb29b53b235954` (2025-09-17; cell-load's current HEAD crashes on duplicated
gene names, so it cannot have built STATE). It runs on **raw counts**, separately per screen file, with
three stages:

1. Keep a target gene if `mean(own-target value in its cells) / mean(that gene in controls) < 0.30`.
2. Keep a cell of a kept target if `own-target value / control mean < 0.50`.
3. Drop targets with fewer than 30 kept cells.

Controls are always kept. Targets whose gene name is not in `var` are dropped. Duplicated gene names
use the first occurrence. All comparisons are strict `<`; a ~zero control mean drops the target. The
result agrees with STATE's cell set exactly (0 extra, 0 missing):

| line | raw cells | kept | kept targets | controls |
|---|---:|---:|---:|---:|
| HepG2 | 145,473 | 96,616 | 1,340 / 2,393 | 4,976 |
| Jurkat | 262,956 | 184,470 | 1,537 / 2,393 | 12,013 |
| K562 essential | 310,385 | 188,590 | 1,383 / 2,057 | 10,691 |
| RPE1 | 247,914 | 173,737 | 1,499 / 2,393 | 11,485 |

STATE renamed 777 barcodes that repeat across lines to `<barcode>-1` (Jurkat 92, K562 263, RPE1 422);
the matching rule strips the suffix.

**Harmonisation to the 2,000 model genes.** Every cell keeps the integer counts of the 2,000 genes the
checkpoint uses, in the order of `metadata/replogle_train.json`; all 2,000 are present exactly once in
every screen. The raw files contain duplicated gene names (`HSPA14` in every screen, `TBCE` in K562),
but none of them is a model gene. Cells whose target label is not in the model's 2,024-label vocabulary
(2,023 targets + `non-targeting`) are dropped. The full-file library (sum over all genes of the screen)
is kept in `obs` for the identity gates only.

**Split.** Test = HepG2 cells of the 372 test targets (`metadata/replogle_test.json`). Train = every
other kept cell: the donor lines (Jurkat, K562, RPE1) with their controls and their cells of the test
targets, plus HepG2's 4,976 controls and its non-test knockdowns (968 targets in the filtered version,
1,650 in the all version). No test target is a HepG2 training perturbation. Controls are each line's
own non-targeting cells, pooled over gem groups; they are never pooled across lines.

**Two versions.**

| version | description | train cells | test cells | HepG2 training targets |
|---|---|---|---|---|
| **filtered** (primary) | STATE's knockdown filter | 616,865 | 26,548 | 968 |
| all | no knockdown filter | 785,303 | 28,177 | 1,650 |

The filtered version reproduces STATE's cells, counts and split exactly. The all version keeps every
cell regardless of knockdown efficacy.

**Identity gates (A1–A5).** `verify_merge.py` gates the filtered version against STATE before any
model output is evaluated:

| gate | what | result |
|---|---|---|
| A1 | cell set + labels (643,413 matched) | 0 only-ours, 0 only-STATE |
| A2 | per-cell 2,000-gene count hash (sha1) | 0 mismatches |
| A3 | full-file library (the sum over all genes of the raw screen) | exact integer match per cell |
| A4 | train/test split identity | train 616,865, test 26,548 |
| A5 | ground truth vs frozen STATE truth (Δ, padj to 1e-9, DEG masks, cell counts) | max \|diff\| 0.0, identical DEG masks |

**Tier B (report).** The filtered-vs-all comparison (`verify_merge.py compare`) reports per-perturbation
statistics for HepG2: effect (Δ) Pearson correlation (median 0.994, 5% 0.963, min 0.875), OLS slope
(median 0.961), and DEG Jaccard (median 0.869). This is a descriptive report, not a gate.

## (b) Normalisation

Ground truth is **CP10k over the 2,000 model genes, then log1p**:

```
y = log1p(10,000 × c / L),  L = sum of counts over the 2,000 model genes
```

This matches the checkpoint: it was trained on integer counts over the same 2,000 genes (encoder input
`log1p(counts)`), and its NB decoder's mean is `softmax(logits) × L` with `L` the 2,000-gene library, so
the readout `softmax(logits) × 1e4` is exactly CP10k over these genes. The full-screen library (the normalisation the STATE file uses) differs from
the 2,000-gene library by a median factor of 3.45 per cell; using it would scale Δ by ~1.66 and add a
per-perturbation shift the model cannot predict (e.g. −0.13 for RPL knockdowns). The ground-truth
script is `ground_truth.py` (unchanged).

Ground truth is computed with `--min-cells 15` (the default is 20; 86 HepG2 test perturbations have
fewer than 40 cells, and 15 keeps all 372).

## (c) Model readout and generation

**Readout.** The NB decoder's expected expression per gene is `softmax(logits) × 1e4`, which needs no
library-size model and introduces no sampling noise. Every generated profile uses this deterministic
mean. The `--save-cells` flag additionally writes NB-sampled cell counts, used only in the cell-level
VCC diagnostics (appendix).

**Effect = generated perturbed − generated control.** Both arms come from the model under the same
guidance, from the same starting noise (`generate.py --guidance W [--ema]`). Comparing model-generated
cells against real controls would mix model bias into the predicted effect.

**Paired noise.** Every condition (control + perturbations, one seed) shares the same latent noise z,
so differences are purely the model's response to the conditioning label.

**Guidance.** Classifier-free guidance is applied to every label key. The configuration is chosen once
on the validation set and frozen.

**Reused generations.** The predictions are the original `generate.py` outputs, not regenerated for
this rebuild. They are in `outputs/perturbation_direction/replogle/` and at the server path on
student03, sha256-identical. The provenance:

- **Validation sweep:** w1, w1_ema, w2, w2_ema, w4, w4_ema (control + 60 validation perturbations,
  64 cells, seed 0). Selected: `w2_ema` (rule: max median balanced accuracy on validation; ties within
  0.005 prefer lower guidance, then raw).
- **Test:** w2_ema, seeds 0, 1, 2 (64 cells per condition), plus a 128-cell run, w1 and w0 controls.
  All batch size 32, library size 4850 (= exp of the checkpoint's mean log library for HepG2 controls).
- **Checkpoint:** sha256 `edb6bb7d...`; `generate.py` sha256 `b840ebfa...`, unchanged since generation.
  The 12 local prediction files are md5-identical to the server copies.

**Smoke test.** On student03 (Titan X Pascal), current code vs stored seed-0 output (made on a Titan V):
effect Pearson ≥ 0.999998, sign agreement 1.0 on all 2,000 genes, max |Δ effect| 0.010 (GPU-architecture
float differences, not bitwise).

## (d) Baselines

Five baselines plus two references:

| baseline | definition | fitted on |
|---|---|---|
| **kNN** | For each test perturbation, the HepG2 effects of the *k* training perturbations whose effects in the donor lines are most similar. *k* is chosen by 5-fold CV on that version's HepG2 training perturbations only. | `baselines.py`, train file |
| **MLP** | A seeded multi-layer perceptron that predicts P(up) for a (perturbation, gene, cell line) triple; scored as P(up) − 0.5. Gene embeddings from SVD of training effects; perturbation embeddings from donor-line effects (no test label in training). | `learned_baselines.py`, train file |
| **generic** | The average effect of all HepG2 training perturbations (the shared HepG2 response). This is the reference for the perturbation-specific report. | `baselines.py`, train file |
| **transfer** | For each test perturbation, the mean measured effect of the same perturbation in the donor cell lines (Jurkat, K562, RPE1). | `baselines.py`, train file |
| **prior** | A label prior: P(up \| gene) from the training labels of all perturbations and lines, Laplace-smoothed; it ignores the perturbation. | `learned_baselines.py`, train file |
| **shuffled** | Each perturbation is scored with another perturbation's scLDM prediction (a derangement). Tests specificity. One row per scLDM seed. | `evaluate.py` (post-hoc shuffle) |
| **noise ceiling (split-half)** | DEGs and Δ from one random half of the real cells, scored on the other half. Slightly pessimistic: each half has only half the cells. | `ground_truth.py` `--seed` |

Each version (filtered, all) refits its baselines on its own train file. This is the final baseline
set; calibrated (ridge), bilinear, GBDT, and per-donor-line transfer baselines have been removed.

MLP and prior output P(up) − 0.5, a direction confidence, not an effect size. They are scored in the
two direction reports and left out of the changed-vs-unchanged report (section g), which ranks genes by
the predicted effect size.

## (e) Validation protocol

The 372 HepG2 test perturbations are split into **60 validation** and **312 test**
(`splits/replogle_val60.json`). Every test target appears in at least one donor line in training.

**What is chosen on validation.** The guidance configuration (weight + EMA on/off) is chosen once by
the maximum median balanced sign accuracy on the 60 validation perturbations, using the filtered
version's truth. Ties within 0.005 prefer lower guidance, then raw weights over EMA. The frozen choice
is `w2_ema`.

**What is frozen.** The guidance config is never re-chosen on the test perturbations or on the all
version. The `select_check` gate (`side_by_side.py check-selection`) only confirms that the rebuilt
filtered truth reproduces the same choice.

**kNN's k.** Chosen by 5-fold cross-validation on the version's HepG2 training perturbations only,
never on validation or test perturbations. Test perturbations are never used for fitting any baseline.

## (f) Metrics and statistics

**DEGs.** A gene is a differentially expressed gene (DEG) for a perturbation if the BH-adjusted
Wilcoxon p < 0.05 **and** |Δ| > 0.1 (Δ = mean perturbed − mean control, log1p CP10k). On unchanged
genes the sign is a coin flip, so the two direction reports (full effect, perturbation-specific) score
DEGs only. A method is therefore never penalised there for predicting a change in a gene that does not
change; the third report (section g) asks that question.

**Balanced sign accuracy.** The mean of recall on up-DEGs and recall on down-DEGs. It stays at 0.5
for a constant "everything up" prediction regardless of class imbalance. Reported as the median over
perturbations.

**Target exclusion.** The CRISPRi target gene is excluded from every perturbation's DEG set
(`evaluate.py --exclude-target`). The fraction of perturbations whose target is predicted down is
reported separately as a sanity check.

**Perturbation-specific report.** The shared HepG2 response `g` (the generic baseline) is projected
out: the scored quantity is the sign of `r = Δ − β·g`, where β is fitted per perturbation by OLS,
separately for the observed and for each predicted Δ. A gene is a specific DEG if it is a DEG and
|r| > 0.1. β is fitted per perturbation because knockdowns differ in how strongly they trigger the
shared response, not only whether they do. The reference for the projection is `--specific-ref`, set
to the generic baseline for each version. The correlation between predicted and observed β across
perturbations is reported as `shared_strength_corr`.

**Perturbation threshold.** Perturbations with fewer than 5 scored DEGs (target excluded) are excluded
from the medians.

**Shared panel.** The main table's medians, CIs and paired tests are computed on the perturbations
with ≥ 5 DEGs (target excluded) in **both** the filtered and the all version, per report. This
keeps the columns comparable on the same perturbations. Each version's own-panel n and median are
listed beside it. The changed-vs-unchanged report additionally requires ≥ 20 unchanged genes
(section g).

**Bootstrap CI.** 95% percentile bootstrap over perturbations (2,000 resamples, seed 0). The unit of
uncertainty is the perturbation, not the gene or the cell.

**Paired tests.** Two-sided Wilcoxon signed-rank test of scLDM seed 0 vs every other row, on the
shared panel. The bootstrap CI of the paired median difference `reference − method` (> 0: reference
is better) is reported. Raw and Holm-corrected p-values are both shown; Holm correction is applied
within each (report, version) column.

## (g) Changed vs unchanged genes (third report)

The direction reports score DEGs only. The third report asks the complementary question: **does the
predicted effect size separate the genes that change from the genes that clearly do not?** A model
that predicts sizeable changes everywhere can have a perfect direction score; here it cannot.

**Gene classes.** Per knockdown, over the scored genes (the CRISPRi target excluded, as in the
direction reports):

| class | definition |
|---|---|
| changed | the DEGs of the direction reports: padj < 0.05 and \|Δ\| > 0.1 (and Δ ≠ 0) |
| unchanged | not a DEG, \|Δ\| < 0.05 (`--unchanged-max-effect`) and padj ≥ 0.5 (`--unchanged-min-padj`); a gene with a NaN padj is never unchanged |
| grey zone | everything else: not significant, but not clearly unaffected |

**Why a grey zone.** "Not significant" is not "unaffected". With a few dozen to a few hundred cells
per knockdown the test misses real but small effects, so the complement of the DEG set mixes
unaffected genes with under-powered ones. Calling all of them negatives would penalise a method for
predicting an effect that is real but was not detected. The unchanged class therefore needs
positive evidence of no change on both axes (a small effect and a large padj), and the genes in
between are left out of the headline. The thresholds are settings, not estimates: they are recorded
in each eval dir's `methods.json` and must agree across the versions of a table.

**Headline: `detect_auroc`.** Per knockdown, the AUROC of the score |Δ_pred| for changed (positive)
against unchanged (negative) genes: the probability that a random changed gene gets a larger
predicted effect than a random unchanged gene, ties counting one half. 0.5 is no separation; a
constant prediction scores exactly 0.5; it is undefined (NaN) when a knockdown has no changed or no
unchanged gene. It is rank-based on purpose: it needs no threshold on the prediction and does not
depend on the prediction's scale, so a method that systematically under- or over-estimates effect
sizes (an average over neighbours or over cell lines shrinks them, for example) is compared on what
it ranks, not on how it is calibrated. The summary is the median (and IQR) over knockdowns.

**Secondary numbers** (all per knockdown, medians in the tables):

| column | meaning |
|---|---|
| `detect_auroc_all` | the same AUROC with every other scored gene (unchanged + grey zone) as negatives; a sensitivity check that uses no threshold beyond the DEG call |
| `abs_pred_deg_median`, `abs_pred_unchanged_median` | median \|Δ_pred\| on the changed / unchanged genes |
| `deg_call_rate`, `unchanged_call_rate` | share of changed / unchanged genes called changed, \|Δ_pred\| > τ |
| `deg_hit_rate` | share of changed genes called changed with the correct sign |
| `three_class_balanced_acc` | mean of three recalls: up-DEGs predicted > τ, down-DEGs predicted < −τ, unchanged genes predicted \|·\| ≤ τ; NaN unless all three classes are non-empty |
| `n_deg`, `n_unchanged`, `n_grey` | genes per class (properties of the truth) |

τ is the truth's own effect threshold (`min_lfc`, 0.1); |Δ_pred| = τ exactly is not a call. These
numbers are secondary because they apply a threshold calibrated on the observed effects to
predictions that live on a different scale: a method that shrinks every effect by a constant factor
keeps its ranking (and its `detect_auroc`) but loses calls. They describe that calibration; they do
not rank methods. A gene without a prediction (NaN) has no score and is left out of the metrics, not
of the counts.

**Panel.** A knockdown is scored with ≥ 5 DEGs (the direction reports' rule, same count) **and** ≥ 20
unchanged genes (`--min-unchanged`), both with the target excluded. `side_by_side.py` applies the
same rule to build the shared panel of this report (both counts in both versions).

**Methods.** Every prediction with its shuffled control (the same derangement as in the full report)
and every baseline, except the methods declared direction-only (`evaluate.py --direction-only mlp
prior`): `learned_baselines.py` writes P(up) − 0.5, whose absolute value is a confidence about the
sign, not a predicted effect size, so ranking genes by it answers a different question. They stay in
the table as "not scored", like `generic` in the perturbation-specific report.

**Split-half ceiling.** On independent cells: the changed / unchanged labels and the signs come
from half A (its DEG mask, its Δ and its padj, `deg_half_a`, `delta_half_a`, `padj_half_a`), the score
is |Δ| of half B. Its summary row uses the panel of the full data, like the other ceilings; in
`side_by_side.py` it additionally needs both gene counts in its own half. `ground_truth.py` saves
`padj_half_a.parquet` for this; a truth dir written before that has no such file, and the report is
then written without a ceiling (with a message).

**Statistics.** As for the other reports: median over the shared panel, 95% percentile bootstrap CI,
paired Wilcoxon signed-rank of scLDM seed 0 against every other row with the CI of the paired median
difference, Holm within each (report, version) column, own-panel numbers beside it. The headline
metric of this report is `--unchanged-metric` (default `detect_auroc`); `--metric` applies to the
two direction reports.

**Caveats.**

- *Not significant is not unaffected.* This is why the unchanged class is defined positively and
  the grey zone is excluded from the headline; `detect_auroc_all` shows what happens without that
  choice.
- *Part of the score is not knockdown-specific.* Low-expressed genes are over-represented among the
  unchanged genes (a small log-scale effect and a large padj are easier to obtain there), and some
  genes respond to almost any knockdown. Knowing which genes are responsive at all therefore
  separates changed from unchanged genes without knowing the knockdown. The `generic` row (the same
  shared response for every knockdown) and the shuffled rows (another knockdown's prediction)
  measure exactly that share; what a method adds is its margin over them.
- *The split-half ceiling is a noise reference, not an upper bound.* Each half has half the cells,
  so |Δ| of half B carries more sampling noise than the full-data effect. The thresholded ceiling
  numbers are pessimistic for that reason: unchanged genes cross τ by chance more often, and half A
  labels fewer DEGs. The rank-based headline is affected too, because a noisy half is ranked
  against labels from another noisy half: a method whose predictions are smoother than a
  half-size sample can reach or exceed the ceiling without being "better than the data". Read it as
  what the real effect of half the cells achieves, not as the best achievable score.
- *The thresholds are choices.* Other values of the effect and padj bounds move genes between the
  unchanged class and the grey zone; the DEG set does not move.

## Appendix: Cell-level VCC diagnostics

The cell-level diagnostics use the pinned `ArcInstitute/cell-eval2` implementation at commit
`a2630dcbda95fc0b819bf841e56d9561d9c1e1da` (0.19.0). These are **raw metrics**, not official
Challenge leaderboard scores; reference-normalised scores require their context-specific anchors.
They were computed before the raw rebuild, on the STATE-route truth; that truth equals the filtered
version's exactly (gate A5), so they apply to the filtered version unchanged.

**Scripts and what they do:**

- `vcc_profiles.py` — cosine/midrank retrieval on effect matrices (a fixed panel of 321 perturbations ×
  1,733 features, from an earlier common gene universe). Not the official counts comparator.
- `denoising_ablation.py` — 2×2 estimator/filtering comparison (observed vs scVI logFC, detection
  filter on/off) with a paired cohort of 246 perturbations.
- `vcc_pilot.py` — 32-cell pilot with three perturbations per strength bin, disjoint real-cell halves.
- `vcc_cells.py` — all six raw metrics via the official `vcc2026` preset, explicit CPU Scanpy DE.
- `generate.py --save-cells` — opt-in genuine NB cell counts (NB head, positive library size required),
  one atomic cell file per condition, merged to `.h5ad` in memory (suitable for small pilots).

**Full-panel cell-level results (test-312).** Generated at 128 cells per condition (w2 + EMA),
equalised per perturbation (both arms same n, n varies 30–64 from real cell counts), three independent
subsamples:

| raw metric | mean | min–max (3 draws) | perturbations scored |
|---|---|---|---:|
| PDS cosine | 0.793 | 0.786–0.800 | 312/312 |
| expression MSE ratio (↓) | 0.864 | 0.771–1.015 | panel-wide |
| DE logFC NMAE (↓) | 0.681 | 0.649–0.711 | 134/312 |
| DE direction fidelity | 0.686 | 0.668–0.715 | 251/312 |
| DE direction reach | 0.841 | 0.788–0.876 | 195/312 |
| DE significance Jaccard | 0.303 | 0.284–0.314 | 312/312 |

Direction is near replicate quality (PDS 0.98 of a disjoint real-half split, fidelity 0.94). Discovery
is the weak point: Jaccard reaches 58% of the real-half value (0.303 vs 0.526). NMAE scores fewer than
half the panel — at these cell counts the reference DE test does not flag enough genes per perturbation
to gate the metric. The real-half comparator is not a formal ceiling (two disjoint 64-cell samples are
a pessimistic stand-in for full-depth measurement).

**Caveats.** All metrics are raw; do not average them into an "overall" score. The CPU DE backend is
an explicit experimental choice, not a claim of parity with Challenge GPU DE.