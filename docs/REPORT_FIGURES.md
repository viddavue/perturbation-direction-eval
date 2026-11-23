# Figures for the practical-work report

Figures are generated from existing outputs, without retraining, regenerating cells,
or choosing new thresholds on the test results. Generated files live under
`outputs/report_figures/` and are not committed.

## Implemented: three-class confusion matrices

From the repository root, in the prepared environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python experiments/perturbation_direction/visualization/confusion.py \
  --config experiments/configs/perturbation_direction/report_figures.yaml
```

The configuration selects the filtered evaluation, the committed shared-panel CSV,
and scLDM seed 0, kNN and generic. Paths may be changed with OmegaConf overrides,
for example `out_dir=outputs/report_figures/confusion_alternative`.
The stored `methods.json` must resolve to the original truth and predictions on
this machine. Copied server outputs may require correcting those paths locally.

Outputs:

- `confusion_macro.pdf` / `.png`: each observed-class row is first normalized
  within each perturbation, then averaged over perturbations with that class.
  `n` is the number of contributing perturbations, separately per observed class.
- `confusion_pooled.pdf` / `.png`: counts pooled across gene–perturbation pairs,
  then row-normalized. Strong perturbations with many DEGs have greater influence.
  Here `n` counts gene–perturbation pairs, not perturbations.
- `per_perturbation_counts.csv`: the nine integer counts for each method and perturbation.
- `confusion_summary.csv`: plotted proportions and denominators.
- `provenance.json`: resolved configuration, thresholds, panel identifiers and SHA256
  hashes of the manifest, panel, truth files and prediction files.

### Reading and captioning the figure

Rows are **observed**, columns **predicted**: down, unchanged, up. Observed changed
genes are DEGs; unchanged genes satisfy the evaluation's small-effect and
large-adjusted-p-value criteria. Grey-zone genes and the CRISPRi target are excluded.
Predictions below −0.1 are down, above +0.1 are up, and values on or between those
bounds are unchanged, using the manifest's recorded threshold.

This is a **threshold-based calibration diagnostic**. It is not Report 1's
sign accuracy (a small correctly signed effect can still be predicted unchanged),
nor Report 3's rank-based AUROC. The macro view uses mean row proportions, not the
report's medians; the diagonal must not be presented as a reproduced headline metric.
MLP and prior cannot be included because their magnitudes are direction confidence,
not expression-effect size.

Suggested caption:

> Three-class calibration of scLDM (seed 0), kNN and generic on the filtered
> changed-versus-unchanged shared panel (191 knockdowns). Rows denote observed
> classes and columns predicted classes at an absolute effect threshold of 0.1.
> Percentages are mean within-knockdown row proportions; each knockdown with
> that observed class has equal weight. Row labels give the number of contributing
> knockdowns. Grey-zone genes and the knocked-down target are excluded.

## Recommended next figures (not implemented yet)

1. **Paired scLDM–kNN comparison, three panels.** One point per perturbation,
   scLDM seed 0 against kNN for full-effect direction, specific direction and
   detection AUROC, with a diagonal and consistent axes. Use each report's
   committed shared panel and finite paired scores. This shows where the methods
   differ instead of repeating the main table; annotate the paired sample size.

2. **Performance by perturbation strength.** Median score by the number of
   full-data DEGs, with fixed bins and the number of perturbations per bin.
   Include scLDM, kNN, generic and shuffled scLDM where applicable. Do not
   assign the split-half reference to bins using its own half-data DEG counts.
   Keep the three reports' different eligible panels explicit.

3. **Generation controls as a compact dot plot.** Display the three report scores
   for w0, w1, selected w2+EMA and w2+EMA with 128 cells, from the committed
   controls tables. Label the raw/EMA weight difference: w1 versus w2 is not
   a pure guidance ablation. Show w0's restricted direction-score coverage.

4. **Full versus specific direction.** Paired per-perturbation scores or a
   compact distribution plot for scLDM, kNN and shuffled scLDM on the intersection
   of the two eligible panels. This illustrates the shared response. Do not
   interpret the score drop as a percentage of biological signal explained:
   projection also changes the scored gene sets.

5. **One illustrative perturbation.** Observed versus predicted effects for
   scLDM and kNN, with observed gene classes highlighted. Select the example by
   a documented rule (e.g. a median-performing perturbation), rather than
   selecting an unusually good or bad case after viewing many examples.

For a short report, prioritize the paired comparison and the confusion matrix;
the controls dot plot is a useful replacement for a second numerical table.
Threshold sensitivity and expression-quartile claims still need their own
reproducible analysis; these figures do not fill that gap.
