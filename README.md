# Evaluating perturbation effects with scLDM

This practical work evaluates whether the single-cell latent diffusion model
[scLDM](https://github.com/czi-ai/scldm) predicts how gene expression changes after a
perturbation. It builds on the upstream scLDM codebase and its pretrained checkpoint;
the contribution of this repository is the evaluation pipeline, baselines, controls,
and analysis.

The main experiment uses the Replogle–Nadig CRISPRi screens. Perturbations held out in
HepG2 are observed in other cell lines during training, so the task is **cross-context
transfer**, not prediction of entirely unseen perturbations. We compare scLDM with
generic, transfer, kNN, MLP, and prior baselines.

## What is evaluated?

1. **Full-effect direction:** whether affected genes go up or down.
2. **Knockdown-specific direction:** direction after projecting out a shared response.
3. **Changed versus unchanged genes:** whether predicted effect magnitudes separate the
   two groups.

The raw-screen rebuild produces a knockdown-filtered dataset (primary) and an unfiltered
version for sensitivity analysis. On the filtered shared panel, scLDM seed 0 scores
0.934, 0.770 and 0.845 respectively; kNN scores 0.938, 0.820 and 0.879. The first
two numbers are balanced sign accuracy and the third is AUROC. These are descriptive
medians; see the [full tables](results/replogle_raw_hvg2000/replogle_raw_test312_w2ema_3seeds_side_by_side.md)
for confidence intervals, paired tests, panel sizes and caveats.

## Repository guide

| Location | Contents |
| --- | --- |
| [`experiments/perturbation_direction/`](experiments/perturbation_direction/) | Evaluation pipeline: data preparation, generation, baselines, metrics, runners and dashboard |
| [`experiments/configs/`](experiments/configs/) | Training and evaluation configurations |
| [`results/`](results/) | Frozen small tables and provenance, including the report's main results |
| [`docs/METHODS.md`](docs/METHODS.md) | Methods, definitions and evaluation choices |
| [`docs/SERVER.md`](docs/SERVER.md) | Data requirements and full server reproduction instructions |
| [`docs/PARSE1M.md`](docs/PARSE1M.md) | Separate exploratory Parse1M experiment |
| [`src/scldm/`](src/scldm/) | Upstream scLDM library |
| [`tests/perturbation_direction/`](tests/perturbation_direction/) | Evaluation tests grouped by pipeline stage; upstream library tests remain in `tests/` |

See the [pipeline guide](experiments/perturbation_direction/README.md) for commands and
the [results guide](results/README.md) for table provenance. Upstream installation,
training and inference documentation is retained in [docs/UPSTREAM_SCLDM.md](docs/UPSTREAM_SCLDM.md).
The original authors' work is credited there and in [LICENSE.md](LICENSE.md).

## Environment and reproduction

Python 3.11 or newer is required. On the development machine, follow the pinned
environment instructions in [docs/SERVER.md](docs/SERVER.md#1-code-and-environment-once);
the project dependency pins alone are not compatible with this evaluation pipeline.
The full experiment needs external raw data, scLDM checkpoints, and stored generations:
they are not included in Git. Check the server guide before running:

```bash
bash experiments/perturbation_direction/runners/run_replogle_raw_data.sh
bash experiments/perturbation_direction/runners/run_replogle_raw_eval.sh
```

The runners accept configuration overrides; their default paths are server-specific.
The evaluation reuses generated predictions and does not retrain scLDM.

Run the tests from the repository root using the prepared environment:

```bash
.venv/bin/python -m pytest -q tests
```

The synthetic runner and Streamlit tests can be resource-intensive; they do not require
the full raw screens. A checkpoint-dependent upstream test skips when its local files
are unavailable, and some optional VCC tests skip without `cell_eval2`.
To limit numerical-library threads on a workstation, use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests
```
