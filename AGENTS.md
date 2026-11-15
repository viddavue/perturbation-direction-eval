# Project instructions

## What this repo is

`scldm` is a single-cell latent diffusion model. Two stages:

1. `src/scldm/vae.py` (`TransformerVAE`): encodes a cell (counts over genes) into a fixed set of
   latent variables via cross-attention onto inducing points, and decodes latents back to an
   NB (or Gaussian) distribution over genes.
2. `src/scldm/models.py` (`LatentDiffusion`): flow matching on those latents with a DiT denoiser
   (`src/scldm/nnets.py`), conditional generation via classifier-free guidance.

`src/scldm/transport/` is ported from SiT and holds the flow-matching objective and ODE/SDE samplers.

## Layout

- `src/scldm/` — the installable library.
- `experiments/configs/` — Hydra configs (`vae_training.yaml`, `ldm_training.yaml`, `inference.yaml`,
  `model/*.yaml`, `datamodule/default.yaml` with per-dataset params, `paths/datasets.yaml`).
- `experiments/scripts/` — `train.py` (VAE), `train_ldm.py` (DiT, loads a frozen VAE), `inference.py`.
- `experiments/perturbation_direction/` — current research focus: scoring whether a perturbation
  pushes a gene up or down. Read its `README.md` and `docs/SERVER.md` before changing anything there.
- `metadata/*.json` — gene and label vocabularies; required for `parse1m` and `replogle` splits.
- `tests/perturbation_direction/` — tests grouped by pipeline stage; `tests/test_load.py` and
  `test_tokenize.py` cover the core library.

## Environment

- Python >= 3.11, `uv`, and an existing `.venv` (set it up before `uv run`).
- `cellarium-ml` is not installable from PyPI; install from the pinned git commit:
  `uv pip install -e ".[test]" "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git@a3720244bf0cfc4950e89ad583ceb0f86ed07199"`
- On this machine use `experiments/perturbation_direction/runners/overrides.txt`; the repo's own pins
  (`anndata<=0.10.9`, `scvi-tools`) conflict with the versions the perturbation pipeline needs.
- Large data and checkpoints are not in git: `_artifacts/resubmission/` (checkpoints, configs,
  size factors) and `_artifacts/datasets/` (AnnData). Fetch with
  `python experiments/perturbation_direction/fetch_artifacts.py --match <dataset>`.

## Verification

- Run `python -m pytest -q tests` after changing library or pipeline code (expect ~255 passed, 5 skipped).
- `ruff check` / `ruff format` via pre-commit; line length is 120 and the docstring convention is numpy.

## Conventions

- Configs are Hydra/OmegaConf; add defaults to `experiments/configs/`, never hardcode paths in Python.
- Keep `src/scldm/` free of experiment-specific logic. Evaluation code belongs in
  `experiments/perturbation_direction/`.
- `outputs/` is generated; do not commit it. Checkpoints and `.parquet` files can be large.
- Mu/sd size-factor pickles and metadata JSONs are release artifacts, not source.
