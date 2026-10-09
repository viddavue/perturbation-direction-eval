# Perturbation effect direction evaluation

How well does scLDM predict whether a perturbation pushes a gene **up or down**?

```
prepare_replogle_state.py  STATE Replogle-Nadig file -> scLDM-style train/test count h5ads
audit.py         fail-fast checks: counts layer, gene/label vocab, split structure, size factors, EMA
ground_truth.py  real Δ (perturbed − control, mean log1p CP10k), Wilcoxon DEGs, split-half noise ceiling
generate.py      scLDM profiles per condition with paired noise (same z for every condition)
baselines.py     transfer (mean / per line), generic shift, calibrated (ridge) and kNN transfer
transfer_models.py  calibrated + kNN cross-context predictors (fit on target-context TRAIN perturbations only)
evaluate.py      per-perturbation metrics, summary table, specificity (shuffled) control, plots
metrics.py       pure metric functions (tested in tests/test_direction_metrics.py)
```

## Design decisions

- **Effect = generated perturbed − generated control.** Both arms come from the model, under the same
  guidance, from the same starting noise. Comparing generated cells against *real* controls would mix
  model bias into the predicted effect.
- **The decoder's expected expression, not NB samples, is the main readout.** For the NB head that is
  softmax(logits) × 1e4, which needs no library-size model and no sampling noise. `--library-size`
  additionally stores an NB-sampled variant (`*_sample.parquet`), normalized the same way as real data.
- **Only DEGs are scored.** A DEG has BH-adjusted Wilcoxon p < 0.05 and |Δ| > 0.1. On unchanged genes
  the sign is a coin flip. The CRISPRi target gene is excluded (`--exclude-target`), and the fraction of
  perturbations whose target is predicted down is reported separately as a sanity check.
- **Headline metric: balanced sign accuracy**, the mean of the recall on up-DEGs and on down-DEGs. It
  stays at 0.5 for a constant "everything up" prediction even when up-DEGs are the majority. Also
  reported: plain sign accuracy, top-20 accuracy, AUROC, and Spearman correlation on DEGs.
- **Perturbations with < 5 DEGs** are excluded from the medians but kept in the per-perturbation
  tables. `by_strength.csv` stratifies the results by effect size.
- **References for the numbers:**
  - noise ceiling: DEGs and directions from one random half of the cells, scored on the other half. It
    is slightly pessimistic, since each half has only half the cells.
  - shuffled: each perturbation is scored with another perturbation's prediction, to test specificity.
  - transfer: the same perturbation's measured effect in other cell lines, averaged over them
    (`transfer`) or taken from one line (`transfer_<line>`).
  - generic: the average effect of the HepG2 training perturbations, i.e. the shared response.
  - calibrated: for each gene, a ridge regression from the other lines' effects to the HepG2 effect.
  - knn: the HepG2 effects of the training perturbations that behave most similarly in the other lines.
  - The calibrated and kNN baselines are fitted only on the 968 HepG2 training perturbations, with
    hyperparameters chosen by 5-fold cross-validation. Test perturbations are never used for fitting.
- **Perturbation-specific report** (`--specific-ref`): scores the sign of (Δ − shared response). Many
  CRISPRi knockdowns share a stress or proliferation signature, so plain sign accuracy can be high for a
  model that only learned that signature.

## Replogle (GPU machine, batch size 32)

**For a step-by-step setup on a (multi-)GPU server, see [SERVER.md](SERVER.md).** That includes
multi-GPU generation (`--gpus 0,1,2,3`, resumable), SLURM arrays, and time estimates. The commands below
are the single-machine version.

STATE-style split: test = HepG2 × 372 CRISPRi targets, all of which are seen in other cell lines in
training. The test file has **no HepG2 non-targeting cells**, so controls come from the train file.

**Data.** The scLDM `*_asSTATE_hvg.h5ad` files are not public. They are rebuilt from STATE's combined file
[`arcinstitute/Replogle-Nadig-Preprint/replogle.h5ad`](https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint).
That file has 643k cells × 6642 genes, the 4 cell lines, and label categories in the same order as
`metadata/replogle_train.json`. Its `X` is dense log1p-normalized expression. `prepare_replogle_state.py`
recovers integer counts per cell, keeps the model's 2000 genes, and splits test (HepG2 × the 372
test genes) from train (all other cells). It works in chunks, so it needs only a few GB of RAM.

```bash
A=_artifacts/resubmission; O=outputs/perturbation_direction/replogle; E=experiments/perturbation_direction
D=_artifacts/datasets/replogle_state
curl -L -C - -o $D/replogle.h5ad \
  https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/resolve/main/replogle.h5ad
python $E/prepare_replogle_state.py --src $D/replogle.h5ad --train-json metadata/replogle_train.json \
  --test-json metadata/replogle_test.json --out-dir $D
TRAIN=$D/replogle_train.h5ad; TEST=$D/replogle_test.h5ad

python $E/audit.py --train-h5ad $TRAIN --test-h5ad $TEST \
  --train-json metadata/replogle_train.json --test-json metadata/replogle_test.json \
  --pert-key gene --control non-targeting --context-key cell_line --target-context hepg2 \
  --size-factor-pkl $A/replogle_log_size_factor_munew.pkl --ckpt $A/replogle_negative_binomial.ckpt

python $E/ground_truth.py --pert-h5ad $TEST --ctrl-h5ad $TRAIN --pert-key gene --control non-targeting \
  --context cell_line=hepg2 --metadata-json metadata/replogle_train.json --out-dir $O/truth

python $E/baselines.py --train-h5ad $TRAIN --pert-key gene --control non-targeting \
  --context-key cell_line --target-context hepg2 --truth-dir $O/truth --out-dir $O/baselines

# pilot first: head -6 $O/truth/conditions.csv > $O/pilot.csv and run with --n-cells 64
LIB=$(python -c "import json;print(json.load(open('$O/truth/info.json'))['control_median_library_size'])")
for W in 1 2 4; do for EMA in "" "--ema"; do
  python $E/generate.py --config $A/replogle_negative_binomial.yaml --ckpt $A/replogle_negative_binomial.ckpt \
    --metadata-json metadata/replogle_train.json --conditions $O/truth/conditions.csv \
    --n-cells 256 --batch-size 32 --guidance $W $EMA --library-size $LIB \
    --out $O/pred/w${W}${EMA:+_ema}.parquet
done; done

python $E/evaluate.py --truth-dir $O/truth --exclude-target \
  --pred scLDM_w1=$O/pred/w1.parquet scLDM_w2=$O/pred/w2.parquet scLDM_w1_ema=$O/pred/w1_ema.parquet \
  --baseline transfer=$O/baselines/delta_transfer.parquet generic=$O/baselines/delta_generic.parquet \
  --out-dir $O/eval
```

## Dentate gyrus smoke test (laptop CPU, batch size 16)

This dataset has no perturbations. "Control" is the ImmGranule1 cluster and the "perturbations" are
other clusters. The run only checks that the pipeline works; the numbers say nothing about perturbation
modelling.

```bash
A=_artifacts/resubmission; O=outputs/perturbation_direction/dentate_gyrus; E=experiments/perturbation_direction
python $E/ground_truth.py --pert-h5ad _artifacts/datasets/dentategyrus_test.h5ad \
  --ctrl-h5ad _artifacts/datasets/dentategyrus_test.h5ad --layer X_counts --pert-key clusters \
  --control ImmGranule1 --perturbations CA OPC Nbl2 ImmAstro RadialGlia \
  --metadata-json metadata/dentategyrus_train.json --out-dir $O
python $E/generate.py --config $A/dentate_gyrus_negative_binomial.yaml --ckpt $A/dentate_gyrus_negative_binomial.ckpt \
  --metadata-json metadata/dentategyrus_train.json --conditions $O/conditions.csv \
  --n-cells 32 --batch-size 16 --device cpu --library-size 2514 --out $O/pred_w1.parquet
python $E/evaluate.py --truth-dir $O --pred scLDM=$O/pred_w1.parquet --out-dir $O/eval
```

Note: in the dentate gyrus h5ad, `X` is log-normalized and raw counts are in `layers['X_counts']`.
