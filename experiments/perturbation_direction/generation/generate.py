"""Generate per-condition pseudobulk expression profiles from a pretrained scLDM with paired noise.

For every condition (a row of ``--conditions``) the same ``--n-cells`` starting noises are integrated
through the probability-flow ODE, so the difference between two conditions is not confounded by
different noise draws. Only the conditional (CFG-guided) prediction is computed.

Two readouts are stored per condition (mean over generated cells, log1p(CP10k) space):
  * ``mean``:   log1p of the decoder's expected expression, normalised to 10k (independent of library size)
  * ``sample``: NB count samples drawn at ``--library-size``, then log1p(CP10k) normalised like real data

Example:
    python experiments/perturbation_direction/generation/generate.py \
        --config _artifacts/resubmission/dentate_gyrus_negative_binomial.yaml \
        --ckpt _artifacts/resubmission/dentate_gyrus_negative_binomial.ckpt \
        --metadata-json metadata/dentategyrus_train.json \
        --conditions outputs/dg/conditions.csv --n-cells 32 --batch-size 16 --out outputs/dg/pred_w1.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scvi.distributions import NegativeBinomial

from scldm._utils import remap_config, remap_pickle
from scldm.logger import logger


def load_module(config_file: str, ckpt_file: str, device: str):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    module_config = OmegaConf.load(config_file)
    remap_config(module_config)
    module = hydra.utils.instantiate(module_config.model.module)

    checkpoint = torch.load(ckpt_file, map_location="cpu", pickle_module=remap_pickle, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    module_keys = set(module.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in module_keys}
    missing = module_keys - set(filtered)
    if missing:
        raise RuntimeError(f"{len(missing)} module keys missing from checkpoint, e.g. {sorted(missing)[:5]}")
    module.load_state_dict(filtered, strict=True)
    logger.info(f"Loaded {len(filtered)} tensors ({len(state_dict) - len(filtered)} checkpoint keys unused)")
    return module.to(device).eval()


class GuidedVelocity(torch.nn.Module):
    """CFG velocity for the conditional branch, matching ``DiT.forward_with_cfg`` semantics.

    joint:              v = u + mean(w) * (c_all - u)
    mutually_exclusive: v = u + sum_k w_k * (c_k - u)
    where u is the all-null (unconditional) prediction. With joint and mean(w) == 1 the
    unconditional pass is skipped since v == c_all.
    """

    def __init__(self, dit, guidance: dict[str, float]):
        super().__init__()
        self.dit = dit
        self.guidance = guidance

    def _null(self, batch_size: int, device) -> dict[str, torch.Tensor]:
        return {
            k: torch.full((batch_size,), n, device=device, dtype=torch.long)
            for k, n in self.dit.class_vocab_sizes.items()
        }

    def forward(self, x, t, condition):
        b = x.shape[0]
        if self.dit.condition_strategy == "joint":
            w = sum(self.guidance.values()) / len(self.guidance)
            cond = self.dit(x, t, condition, force_drop_ids=False)
            if w == 1.0:
                return cond
            uncond = self.dit(x, t, self._null(b, x.device), force_drop_ids=False)
            return uncond + w * (cond - uncond)
        uncond = self.dit(x, t, self._null(b, x.device), force_drop_ids=False)
        out = uncond.clone()
        for key, w in self.guidance.items():
            out += w * (self.dit(x, t, {key: condition[key]}, force_drop_ids=False) - uncond)
        return out


@torch.no_grad()
def decode_expected(vae, z: torch.Tensor, genes: torch.Tensor, gene_chunk: int):
    """Decode latents for all ``genes``, chunking over genes to bound memory.

    Gene queries only attend to the latent tokens, so per-gene outputs are independent of which other
    genes are decoded; the NB softmax is applied over the full gene set afterwards.

    Returns (expected CP10k [B, G], theta [B, G] or None for the Gaussian head).
    """
    head = vae.decoder_head
    head_name = head.__class__.__name__
    b = z.shape[0]
    outs, thetas = [], []
    for start in range(0, genes.numel(), gene_chunk):
        g = genes[start : start + gene_chunk].unsqueeze(0).expand(b, -1)
        g_dec = g if isinstance(vae.decoder.gene_embedding, torch.nn.Embedding) else vae.input_layer.gene_embedding(g)
        h = vae.decoder(z, g_dec, None)
        if head_name == "GaussianTransformerLayer":
            outs.append(head(h, g, None))
        elif head_name == "NegativeBinomialTransformerLayer":
            if isinstance(head.theta, torch.nn.Embedding):
                logits, theta = head.params(h).squeeze(-1), torch.exp(head.theta(g.long())).squeeze(-1)
            else:
                logits, log_theta = torch.chunk(head.params(h), 2, dim=-1)
                logits, theta = logits.squeeze(-1), torch.exp(log_theta).squeeze(-1)
            outs.append(logits / head.t)
            thetas.append(theta)
        else:
            raise NotImplementedError(head_name)
    out = torch.cat(outs, dim=1)
    if head_name == "GaussianTransformerLayer":
        # Gaussian head predicts log1p(CP10k) directly
        return torch.expm1(out.clamp(min=0)), None
    return torch.softmax(out, dim=1) * 1e4, torch.cat(thetas, dim=1)


def build_condition_indices(conditions: pd.DataFrame, metadata_json: str, keys: list[str]) -> dict[str, np.ndarray]:
    """Map label strings to indices using the TRAIN metadata ordering (as used during training)."""
    labels = json.loads(pathlib.Path(metadata_json).read_text())["labels"]
    out = {}
    for k in keys:
        lut = {str(v): i for i, v in enumerate(labels[k])}
        unknown = sorted(set(conditions[k].astype(str)) - set(lut))
        if unknown:
            raise KeyError(f"labels not in training vocabulary for '{k}': {unknown[:10]}")
        out[k] = conditions[k].astype(str).map(lut).to_numpy()
    return out


RUN_KEYS = (
    "config",
    "ckpt",
    "metadata_json",
    "conditions",
    "n_cells",
    "batch_size",
    "guidance",
    "ema",
    "library_size",
    "seed",
    "fp32",
    "save_cells",
)


def parts_dir(out: pathlib.Path) -> pathlib.Path:
    return out.with_name(out.stem + ".parts")


def part_paths(out: pathlib.Path, shard: int, num_shards: int) -> tuple[pathlib.Path, pathlib.Path]:
    d = parts_dir(out)
    return d / f"shard{shard:03d}of{num_shards:03d}.parquet", d / f"shard{shard:03d}of{num_shards:03d}_sample.parquet"


def check_run_settings(args) -> None:
    """Record the run settings next to the parts; refuse to resume/merge parts made with other settings."""
    d = parts_dir(pathlib.Path(args.out))
    d.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in RUN_KEYS}
    path = d / "settings.json"
    if path.exists():
        old = json.loads(path.read_text())
        old.setdefault("save_cells", False)  # old profile-only runs remain resumable
        if old != json.loads(json.dumps(settings)):
            raise SystemExit(
                f"{path} was written with different settings:\n  old {old}\n  new {settings}\n"
                "use a different --out or delete the .parts directory"
            )
    else:
        path.write_text(json.dumps(settings, indent=1))


def save_rows(rows: dict, gene_names: np.ndarray, path: pathlib.Path) -> None:
    df = pd.DataFrame(np.stack(list(rows.values())), index=pd.Index(list(rows), name="id"), columns=gene_names)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)  # atomic: an interrupted write never leaves a corrupt part


def cell_path(out: pathlib.Path, cid: str) -> pathlib.Path:
    """Stable, filesystem-safe path independent of shard count."""
    return parts_dir(out) / "cells" / (hashlib.sha256(cid.encode()).hexdigest() + ".h5ad")


def save_count_cells(counts, gene_names, condition: dict, path: pathlib.Path):
    """Persist actual NB samples, never repeated means. Used for the DE pilot only."""
    import anndata as ad
    from scipy import sparse

    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != len(gene_names):
        raise ValueError("count matrix/gene shape mismatch")
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("NB readout must be finite nonnegative integer counts")
    obs = pd.DataFrame(
        {k: [v] * len(counts) for k, v in condition.items()},
        index=[f"{condition['id']}_{i}" for i in range(len(counts))],
    )
    data = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=gene_names))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5ad")
    data.write_h5ad(tmp, compression="gzip")
    tmp.replace(path)


def run_shard(args, shard: int, num_shards: int) -> None:
    """Generate profiles for conditions[shard::num_shards]; saves after every condition and resumes."""
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest" if args.fp32 else "high")

    conditions = pd.read_csv(args.conditions, dtype=str)
    mine = conditions.iloc[shard::num_shards].reset_index(drop=True)
    out_mean, out_sample = part_paths(pathlib.Path(args.out), shard, num_shards)
    meta = json.loads(pathlib.Path(args.metadata_json).read_text())
    gene_names = np.asarray(meta["genes"])

    # resume: conditions finished by ANY earlier part (also with a different shard count) are skipped
    finished = set()
    for part in parts_dir(pathlib.Path(args.out)).glob("shard*of*[0-9].parquet"):
        finished |= set(pd.read_parquet(part).index)
    if args.save_cells:
        finished = {cid for cid in finished if cell_path(pathlib.Path(args.out), cid).exists()}
    rows_mean = pd.read_parquet(out_mean).to_dict("index") if out_mean.exists() else {}
    rows_sample = pd.read_parquet(out_sample).to_dict("index") if out_sample.exists() else {}
    rows_mean = {k: np.array([v[g] for g in gene_names]) for k, v in rows_mean.items()}
    rows_sample = {k: np.array([v[g] for g in gene_names]) for k, v in rows_sample.items()}
    todo = [i for i, cid in enumerate(mine["id"]) if cid not in finished]
    tag = f"[shard {shard}/{num_shards}]"
    logger.info(f"{tag} {len(mine)} conditions, {len(mine) - len(todo)} already done, device {args.device}")
    if not todo:
        return

    module = load_module(args.config, args.ckpt, args.device)
    dit = module.ema_model.ema_model if args.ema else module.diffusion_model
    dit.eval()
    keys = list(dit.class_vocab_sizes.keys())
    genes = torch.arange(1, len(gene_names) + 1, device=args.device)  # index 0 is <MASK>
    cond_idx = build_condition_indices(mine, args.metadata_json, keys)

    velocity = GuidedVelocity(dit, dict.fromkeys(keys, args.guidance))
    sample_fn = module.transport_sampler.sample_ode()
    seq_len, latent_dim = dit.seq_len, module.vae_model.encoder.latent_embedding
    n_batches = -(-args.n_cells // args.batch_size)

    t0 = time.time()
    for n_done, row_i in enumerate(todo, start=1):
        cid = mine["id"][row_i]
        sum_mean = torch.zeros(len(gene_names), dtype=torch.float64)
        sum_sample = torch.zeros(len(gene_names), dtype=torch.float64)
        done = 0
        count_batches = []
        for b in range(n_batches):
            bs = min(args.batch_size, args.n_cells - done)
            # identical noise for batch b in every condition (and on every shard) -> paired comparison
            gen = torch.Generator().manual_seed(args.seed * 100_003 + b)
            z = torch.randn((bs, seq_len, latent_dim), generator=gen).to(args.device)
            condition = {k: torch.full((bs,), int(cond_idx[k][row_i]), device=args.device) for k in keys}
            z1 = sample_fn(z, velocity, condition=condition)[-1]
            cp10k, theta = decode_expected(module.vae_model, z1, genes, args.gene_chunk)
            if args.save_cells and theta is None:
                raise ValueError("--save-cells requires an NB head; Gaussian readouts are not raw counts")
            sum_mean += torch.log1p(cp10k).sum(0).double().cpu()
            if args.library_size is not None and theta is not None:
                torch.manual_seed(args.seed * 100_003 + b + 7)
                counts = NegativeBinomial(mu=cp10k * (args.library_size / 1e4), theta=theta).sample()
                if args.save_cells:
                    count_batches.append(counts.cpu().numpy())
                lib = counts.sum(1, keepdim=True).clamp(min=1)
                sum_sample += torch.log1p(counts / lib * 1e4).sum(0).double().cpu()
            done += bs
        if args.save_cells:
            save_count_cells(
                np.concatenate(count_batches),
                gene_names,
                mine.iloc[row_i].to_dict(),
                cell_path(pathlib.Path(args.out), cid),
            )
        rows_mean[cid] = (sum_mean / done).numpy()
        save_rows(rows_mean, gene_names, out_mean)
        if args.library_size is not None:
            rows_sample[cid] = (sum_sample / done).numpy()
            save_rows(rows_sample, gene_names, out_sample)
        elapsed = time.time() - t0
        eta = elapsed / n_done * (len(todo) - n_done)
        logger.info(f"{tag} [{n_done}/{len(todo)}] {cid} done ({elapsed / n_done:.1f}s/cond, ETA {eta / 60:.0f} min)")


def merge(args) -> None:
    out = pathlib.Path(args.out)
    conditions = pd.read_csv(args.conditions, dtype=str)
    for suffix, pattern in (("", "shard*of*[0-9].parquet"), ("_sample", "shard*_sample.parquet")):
        parts = sorted(parts_dir(out).glob(pattern))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts])
        missing = sorted(set(conditions["id"]) - set(df.index))
        if missing:
            raise SystemExit(f"cannot merge: {len(missing)} conditions not generated yet, e.g. {missing[:5]}")
        df = df[~df.index.duplicated()].loc[conditions["id"]]
        df.index.name = "id"
        dst = out.with_name(out.stem + suffix + ".parquet")
        df.to_parquet(dst)
        logger.info(f"wrote {dst} ({len(df)} conditions)")
    if args.save_cells:
        import anndata as ad

        paths = [cell_path(out, cid) for cid in conditions["id"]]
        if not all(p.exists() for p in paths):
            raise SystemExit("missing cell files: rerun generation with --save-cells")
        cells = ad.concat([ad.read_h5ad(p) for p in paths], join="inner")
        cells.write_h5ad(out.with_suffix(".h5ad"), compression="gzip")


def launch_multi_gpu(args, gpus: list[str], argv: list[str]) -> None:
    """One worker process per GPU (each sees only its GPU), then merge."""
    import subprocess

    d = parts_dir(pathlib.Path(args.out))
    # drop launcher-only flags; workers get --shard/--num-shards and a pinned device
    child, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("--gpus", "--device"):
            skip = True
            continue
        if a.startswith(("--gpus=", "--device=")):
            continue
        child.append(a)
    procs = []
    for k, g in enumerate(gpus):
        log = open(d / f"shard{k:03d}of{len(gpus):03d}.log", "a")
        cmd = [
            sys.executable,
            "-u",
            __file__,
            *child,
            "--shard",
            str(k),
            "--num-shards",
            str(len(gpus)),
            "--device",
            "cuda",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g, "PYTHONUNBUFFERED": "1"}
        procs.append((k, g, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        logger.info(f"started shard {k} on GPU {g} (log: {log.name})")
    failed = []
    for k, g, p, log in procs:
        if p.wait() != 0:
            failed.append(f"shard {k} (GPU {g}): exit {p.returncode}, see {log.name}")
        log.close()
    if failed:
        raise SystemExit("some shards failed; rerun the same command to resume:\n  " + "\n  ".join(failed))
    merge(args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metadata-json", required=True, help="TRAIN metadata json (defines gene + label ordering)")
    ap.add_argument("--conditions", required=True, help="CSV: 'id' column + one column per condition key")
    ap.add_argument("--n-cells", type=int, default=256, help="generated cells per condition")
    ap.add_argument("--batch-size", type=int, default=32, help="cells per ODE solve (16 on CPU, 32 on GPU)")
    ap.add_argument("--guidance", type=float, default=1.0, help="CFG weight (applied to every key)")
    ap.add_argument("--ema", action="store_true", help="use EMA diffusion weights")
    ap.add_argument("--library-size", type=float, default=None, help="library size for NB count samples")
    ap.add_argument("--save-cells", action="store_true", help="save actual NB cell counts alongside profiles (pilot)")
    ap.add_argument("--gene-chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="full fp32 matmuls (default: TF32, as experiments/scripts/inference.py uses)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="output parquet (readout 'mean'); '_sample' variant alongside")
    ap.add_argument("--gpus", help="comma-separated GPU ids, one worker per GPU, e.g. 0,1,2,3 (then merges)")
    ap.add_argument("--shard", type=int, help="run only shard k of --num-shards (e.g. one SLURM array task)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true", help="merge finished shards into --out")
    args = ap.parse_args(argv)
    if args.save_cells and (args.library_size is None or args.library_size <= 0):
        ap.error("--save-cells requires a positive --library-size")

    if args.merge_only:
        merge(args)
        return
    check_run_settings(args)
    if args.gpus and len(args.gpus.split(",")) > 1:
        launch_multi_gpu(args, args.gpus.split(","), argv)
    elif args.shard is not None:
        run_shard(args, args.shard, args.num_shards)  # merge separately once all shards are done
    else:
        if args.gpus:
            args.device = f"cuda:{args.gpus}"
        run_shard(args, 0, 1)
        merge(args)


if __name__ == "__main__":
    main()
