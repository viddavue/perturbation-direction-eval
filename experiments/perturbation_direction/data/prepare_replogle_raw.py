"""Build the Replogle train/test h5ads from the raw screens, with and without STATE's knockdown filter.

Inputs and parameters come from ``experiments/configs/perturbation_direction/replogle_raw.yaml``
(OmegaConf; any key can be overridden on the command line as ``key=value``). For every screen:

1. ``knockdown_filter.read_target_expression`` + ``filter_on_target_knockdown`` give the per-cell
   knockdown keep mask (per screen or on all screens merged, as ``knockdown_filter.SCOPE`` says; the
   matrix the filter sees is ``knockdown_filter.TRANSFORM``). Controls are always kept.
2. One blocked pass over the raw matrix reads the 2,000 model-gene counts (``var[gene_col]``, metadata
   order), the cell's library over ALL genes of the file, and obs.
3. Labels are restricted to the model vocabulary (``train_json`` labels, incl. the control).
4. Split: ``test`` = ``cell_line == target_context`` and target in ``test_json``; ``train`` = every
   other kept cell, including the target line's controls and its other perturbations.

Written to ``out_dir``, per version ``v`` in ``versions``:

  replogle_train_raw_<v>.h5ad, replogle_test_raw_<v>.h5ad
      X    CSR int32 counts, var index = the metadata genes in order (no layers)
      obs  cell_id (raw obs index), gene / cell_line (categoricals in metadata order), screen,
           gem_group, UMI_count, library_full (count total over all genes of the raw file),
           kd_keep (the knockdown filter's decision; all True in ``filtered``)
      obs index ``<screen>:<cell_id>`` (raw barcodes can repeat across screens)

  ``filtered``: knockdown-kept cells (STATE's selection; the primary version). ``all``: every cell.

  manifest.json         inputs (size, md5 from the .md5 files, and md5_computed when out_dir holds the
                        checksums.json of verify_merge.py checksums), git commit, resolved config,
                        knockdown-filter module/settings, per version/split/line counts, filter
                        summaries, dropped labels, split checks
  knockdown_report.json full per-screen filter reports (per-target knockdown ratios)
  test_target_cells.csv cells per test target in each version

Split rules (asserted, the run fails otherwise): no test target is a training perturbation of the
target line; every test target is perturbed in at least one other line in train; every test target
has at least ``split.min_test_cells`` cells in the primary version (reported for the others).
Memory stays bounded: parts are written per row block and concatenated on disk.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.experimental import concat_on_disk
from omegaconf import DictConfig, OmegaConf
from scipy import sparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.data import knockdown_filter as kd
from experiments.perturbation_direction.data.audit_raw_screen import (
    iter_row_blocks,
    model_gene_positions,
    read_column,
    read_index,
    row_sums,
    take_columns,
    to_jsonable,
)
from experiments.perturbation_direction.data.make_state_reference import git_commit

VERSIONS = ("filtered", "all")
SPLITS = ("train", "test")
CHECKSUMS = "checksums.json"  # written into out_dir by verify_merge.py checksums


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """YAML config with ``key=value`` overrides (OmegaConf dotlist; list items as ``screens.1.md5=...``)."""
    cfg = OmegaConf.load(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, value = item.split("=", 1)
        OmegaConf.update(cfg, key, OmegaConf.from_dotlist([f"v={value}"]).v, merge=True)
    unknown = set(cfg.versions) - set(VERSIONS)
    if unknown:
        raise ValueError(f"unknown versions {sorted(unknown)}; known: {VERSIONS}")
    if cfg.primary_version not in cfg.versions:
        raise ValueError(f"primary_version {cfg.primary_version!r} is not in versions {list(cfg.versions)}")
    return cfg


def output_path(out_dir, split: str, version: str) -> pathlib.Path:
    return pathlib.Path(out_dir) / f"replogle_{split}_raw_{version}.h5ad"


def read_md5_file(path: pathlib.Path) -> str | None:
    """First token of ``<file>.md5`` (md5sum format), or None if absent/empty."""
    md5 = path.with_name(path.name + ".md5")
    if not md5.exists():
        return None
    text = md5.read_text().split()
    return text[0].lower() if text else None


def read_checksums(out_dir) -> dict[str, dict] | None:
    """Per-screen entries of ``<out_dir>/checksums.json`` (verify_merge.py checksums), None if absent."""
    p = pathlib.Path(out_dir) / CHECKSUMS
    if not p.exists():
        return None
    return {r["name"]: r for r in json.loads(p.read_text())["files"]}


def check_inputs(cfg: DictConfig) -> list[dict]:
    """Existence, size and md5 (from the .md5 files written after download) of every raw screen.

    When ``<out_dir>/checksums.json`` exists, each record also gets ``md5_computed``: the md5 that
    ``verify_merge.py checksums`` recomputed, used only if its entry still describes the file (same
    path, size and mtime; otherwise None plus ``md5_computed_note``). A current entry that disagrees
    with the .md5 file or the config md5 stops the run.
    """
    out, problems = [], []
    computed = read_checksums(cfg.out_dir)
    for s in cfg.screens:
        p = pathlib.Path(cfg.raw_dir) / s.file
        rec = {"name": s.name, "cell_line": s.cell_line, "path": str(p), "url": s.get("url")}
        if not p.exists():
            problems.append(f"{p}: missing")
            out.append(rec)
            continue
        rec["size"] = p.stat().st_size
        rec["expected_size"] = s.get("size")
        rec["md5"] = read_md5_file(p)
        rec["expected_md5"] = s.get("md5")
        rec["size_ok"] = rec["expected_size"] is None or rec["size"] == rec["expected_size"]
        rec["md5_ok"] = rec["md5"] is not None and (rec["expected_md5"] is None or rec["md5"] == rec["expected_md5"])
        if not rec["size_ok"]:
            problems.append(f"{p}: size {rec['size']} != expected {rec['expected_size']}")
        if cfg.require_md5 and not rec["md5_ok"]:
            problems.append(f"{p}: md5 file missing/empty or {rec['md5']} != expected {rec['expected_md5']}")
        if computed is not None:
            c, st = computed.get(s.name), p.stat()
            if c and (c.get("path"), c.get("size"), c.get("mtime_ns")) == (str(p), st.st_size, st.st_mtime_ns):
                rec["md5_computed"] = c.get("md5_computed")
                refs = [x for x in (rec["md5"], rec["expected_md5"]) if x is not None]
                if rec["md5_computed"] is None or any(rec["md5_computed"] != str(x).lower() for x in refs):
                    problems.append(f"{p}: recomputed md5 {rec['md5_computed']} ({CHECKSUMS}) != {refs}")
            else:
                rec["md5_computed"] = None
                rec["md5_computed_note"] = f"no current {CHECKSUMS} entry (missing, or size/mtime changed since)"
        out.append(rec)
    if problems:
        raise RuntimeError("raw inputs not ready:\n  " + "\n  ".join(problems))
    return out


def knockdown_masks(cfg: DictConfig, paths: list[pathlib.Path]) -> tuple[list[np.ndarray], list, dict]:
    """Per-screen keep masks from knockdown_filter, honouring its SCOPE."""
    params = {
        "control": cfg.control,
        "residual_expression": float(cfg.knockdown.residual_expression),
        "cell_residual_expression": float(cfg.knockdown.cell_residual_expression),
        "min_cells": int(cfg.knockdown.min_cells),
    }
    tes = []
    for p in paths:
        t0 = time.time()
        tes.append(
            kd.read_target_expression(
                p,
                pert_key=cfg.pert_key,
                control=cfg.control,
                gene_col=cfg.gene_col,
                transform=kd.TRANSFORM,
                block=cfg.block,
            )
        )
        print(f"[kd] read {p.name}: {len(tes[-1].perts)} cells in {time.time() - t0:.0f}s", flush=True)
    if kd.SCOPE == "per_screen":
        results = [kd.filter_on_target_knockdown(te, **params) for te in tes]
        masks, reports = [r[0] for r in results], [r[1] for r in results]
    elif kd.SCOPE == "merged":
        keep, report = kd.filter_on_target_knockdown(kd.concat_target_expression(tes), **params)
        bounds = np.cumsum([0] + [len(te.perts) for te in tes])
        masks = [keep[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        reports = [report]
    else:
        raise ValueError(f"unknown knockdown_filter.SCOPE {kd.SCOPE!r}")
    for te, m in zip(tes, masks, strict=True):
        m = np.asarray(m)
        if m.dtype != bool or m.shape != (len(te.perts),):
            raise ValueError(f"knockdown mask has dtype {m.dtype} / shape {m.shape}, expected bool ({len(te.perts)},)")
        if not m[np.asarray(te.perts).astype(str) == cfg.control].all():
            raise ValueError("knockdown filter dropped control cells")
    return [np.asarray(m) for m in masks], tes, {"params": params, "reports": reports}


def report_summary(report: dict) -> dict:
    """The small, scalar-ish part of a filter report (the full one goes to knockdown_report.json)."""
    out = {}
    for k, v in report.items():
        if k == "per_target":
            continue
        if k == "kept_targets":
            out["n_kept_targets"] = len(v)
            continue
        out[k] = v
    return out


def _blank_obs() -> dict[str, list]:
    return {"cell_id": [], "cell_line": [], "gene": [], "screen": [], "split": []}


def prepare(cfg: DictConfig) -> dict:
    """Run the whole data stage; returns the manifest (also written to ``out_dir/manifest.json``)."""
    t_start = time.time()
    out = pathlib.Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = out / "_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()

    meta = json.loads(pathlib.Path(cfg.train_json).read_text())
    genes = list(meta["genes"])
    vocab = list(meta["labels"][cfg.pert_key])
    lines = list(meta["labels"][cfg.context_key])
    test_targets = list(json.loads(pathlib.Path(cfg.test_json).read_text())["labels"][cfg.pert_key])
    if cfg.control not in vocab:
        raise ValueError(f"control {cfg.control!r} is not in the label vocabulary")
    if missing := sorted(set(test_targets) - set(vocab)):
        raise ValueError(f"test targets outside the vocabulary: {missing[:5]}")
    screens = list(cfg.screens)
    names = [s.name for s in screens]
    if len(set(names)) != len(names):
        raise ValueError("screen names must be unique")
    if bad := sorted({s.cell_line for s in screens} - set(lines)):
        raise ValueError(f"screen cell lines outside the vocabulary: {bad}")
    if cfg.target_context not in {s.cell_line for s in screens}:
        raise ValueError(f"no screen for target_context {cfg.target_context!r}")

    inputs = check_inputs(cfg)
    paths = [pathlib.Path(r["path"]) for r in inputs]
    masks, tes, kd_info = knockdown_masks(cfg, paths)

    versions = list(cfg.versions)
    vocab_set, test_set = set(vocab), set(test_targets)
    parts: dict[tuple[str, str], list[str]] = {(v, s): [] for v in versions for s in SPLITS}
    obs_log: dict[str, dict[str, list]] = {v: _blank_obs() for v in versions}
    raw_counts, dropped, lib_check = {}, {v: {} for v in versions}, {}
    screen_cat = pd.CategoricalDtype(names)
    gene_cat, line_cat = pd.CategoricalDtype(vocab), pd.CategoricalDtype(lines)
    var = pd.DataFrame(index=pd.Index(genes))

    for si, (s, path, keep_kd, te) in enumerate(zip(screens, paths, masks, tes, strict=True)):
        t0 = time.time()
        with h5py.File(path, "r") as f:
            var_names = read_column(f["var"], cfg.gene_col).astype(str)
            pos, missing, dup = model_gene_positions(var_names, genes)
            if missing or dup:
                raise ValueError(f"{s.name}: model genes missing {missing[:5]} / duplicated {dup[:5]}")
            cell_ids = read_index(f["obs"])
            labels = np.array([str(x) for x in read_column(f["obs"], cfg.pert_key)], dtype=object)
            n = len(cell_ids)
            if not np.array_equal(np.asarray(te.perts).astype(str), labels.astype(str)):
                raise ValueError(f"{s.name}: knockdown_filter's perts differ from obs[{cfg.pert_key!r}]")
            extra = {}
            for col, dtype in (("gem_group", np.int64), ("UMI_count", np.float32)):
                extra[col] = (
                    read_column(f["obs"], col).astype(dtype) if col in f["obs"] else np.full(n, -1).astype(dtype)
                )
            is_ctrl = labels == cfg.control
            in_vocab = pd.Index(labels).isin(vocab_set)
            is_test = (s.cell_line == cfg.target_context) & pd.Index(labels).isin(test_set)
            keep = {"filtered": keep_kd & in_vocab, "all": in_vocab.copy()}
            raw_counts[s.name] = {
                "cells": n,
                "controls": int(is_ctrl.sum()),
                "targets": int(len(set(labels[~is_ctrl]))),
                "kd_kept_cells": int(keep_kd.sum()),
                "kd_kept_targets": int(len(set(labels[keep_kd & ~is_ctrl]))),
            }
            for v in versions:
                gone = pd.Series(labels[~in_vocab & (keep_kd if v == "filtered" else True)]).value_counts()
                dropped[v][s.name] = {
                    "cells": int(gone.sum()),
                    "labels": int(len(gone)),
                    "top": {str(k): int(c) for k, c in gone.head(20).items()},
                }

            library = np.zeros(n)
            for b, (start, stop, rows) in enumerate(iter_row_blocks(f["X"], cfg.block)):
                library[start:stop] = row_sums(rows)
                sub = take_columns(rows, pos)
                if (sub < 0).any() or not np.array_equal(sub, np.rint(sub)):
                    raise ValueError(f"{s.name}: non-integer or negative counts in rows {start}-{stop}")
                csr = sparse.csr_matrix(sub.astype(np.int32))
                sl = slice(start, stop)
                obs_block = pd.DataFrame(
                    {
                        "cell_id": cell_ids[sl],
                        "gene": pd.Categorical(labels[sl], dtype=gene_cat),
                        "cell_line": pd.Categorical([s.cell_line] * (stop - start), dtype=line_cat),
                        "screen": pd.Categorical([s.name] * (stop - start), dtype=screen_cat),
                        "gem_group": extra["gem_group"][sl],
                        "UMI_count": extra["UMI_count"][sl],
                        "library_full": library[sl].astype(np.int64),
                        "kd_keep": keep_kd[sl],
                    },
                    index=pd.Index([f"{s.name}:{c}" for c in cell_ids[sl]]),
                )
                for v in versions:
                    for split, in_split in (("test", is_test[sl]), ("train", ~is_test[sl])):
                        m = keep[v][sl] & in_split
                        if not m.any():
                            continue
                        p = parts_dir / f"{v}_{split}_{si:02d}_{b:05d}.h5ad"
                        ad.AnnData(X=csr[m], obs=obs_block[m], var=var).write_h5ad(p)
                        parts[(v, split)].append(str(p))
                        log = obs_log[v]
                        log["cell_id"].append(cell_ids[sl][m])
                        log["cell_line"].append(np.repeat(s.cell_line, m.sum()))
                        log["gene"].append(labels[sl][m])
                        log["screen"].append(np.repeat(s.name, m.sum()))
                        log["split"].append(np.repeat(split, m.sum()))
        same = bool(np.array_equal(library, np.asarray(te.library, dtype=np.float64)))
        lib_check[s.name] = {"equal_to_knockdown_filter_library": same, "zero_library_cells": int((library <= 0).sum())}
        if not same:
            raise ValueError(f"{s.name}: library over all genes differs from knockdown_filter's")
        print(f"[prepare] {s.name}: {n} cells, {time.time() - t0:.0f}s", flush=True)

    outputs = {}
    for v in versions:
        outputs[v] = {}
        for split in SPLITS:
            dst = output_path(out, split, v)
            if not parts[(v, split)]:
                raise RuntimeError(f"no cells for version {v!r}, split {split!r}")
            concat_on_disk(parts[(v, split)], dst, axis=0, join="inner", merge="same")
            outputs[v][split] = check_output(dst, genes, vocab, lines)
            print(f"[prepare] wrote {dst} ({outputs[v][split]['cells']} cells)", flush=True)
    shutil.rmtree(parts_dir)

    obs_tables = {v: pd.DataFrame({k: np.concatenate(x) for k, x in obs_log[v].items()}) for v in versions}
    counts = {v: split_counts(obs_tables[v], cfg.control) for v in versions}
    checks = {v: split_checks(obs_tables[v], test_targets, cfg, required=(v == cfg.primary_version)) for v in versions}
    controls_kept = controls_check(obs_tables, cfg.control) if {"filtered", "all"} <= set(versions) else None
    for v in versions:
        for split in SPLITS:
            if outputs[v][split]["cells"] != int((obs_tables[v].split == split).sum()):
                raise RuntimeError(f"{v}/{split}: written cell count differs from the selection")
    tt = pd.DataFrame(
        {v: obs_tables[v][obs_tables[v].split == "test"].gene.value_counts() for v in versions}, index=test_targets
    ).fillna(0)
    tt.astype(int).rename_axis("target").to_csv(out / "test_target_cells.csv")

    (out / "knockdown_report.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "scope": kd.SCOPE,
                    "screens": names if kd.SCOPE == "per_screen" else ["merged"],
                    "reports": kd_info["reports"],
                }
            ),
            indent=1,
        )
    )
    kd_file = pathlib.Path(kd.__file__)
    passed = all(c["pass"] for c in checks.values()) and (controls_kept is None or controls_kept["pass"])
    manifest = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t_start, 1),
        "git": git_commit(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "knockdown_filter": {
            "module": str(kd_file),
            "module_sha256": hashlib.sha256(kd_file.read_bytes()).hexdigest(),
            "TRANSFORM": kd.TRANSFORM,
            "SCOPE": kd.SCOPE,
            "CELL_LOAD_COMMIT": kd.CELL_LOAD_COMMIT,
            "params": kd_info["params"],
            "summary": dict(
                zip(
                    names if kd.SCOPE == "per_screen" else ["merged"],
                    [report_summary(r) for r in kd_info["reports"]],
                    strict=True,
                )
            ),
            "full_report": "knockdown_report.json",
        },
        "inputs": inputs,
        "metadata": {
            "train_json": str(cfg.train_json),
            "test_json": str(cfg.test_json),
            "n_genes": len(genes),
            "n_vocab": len(vocab),
            "n_test_targets": len(test_targets),
        },
        "raw_counts": raw_counts,
        "library_check": lib_check,
        "dropped_labels": dropped,
        "outputs": outputs,
        "counts": counts,
        "split_checks": checks,
        "controls_kept": controls_kept,
        "test_target_cells": "test_target_cells.csv",
        "pass": passed,
    }
    (out / "manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=1))
    if not passed:
        failed = [
            f"{v}:{k}" for v, c in checks.items() for k, r in c["rules"].items() if r["required"] and not r["pass"]
        ]
        if controls_kept is not None and not controls_kept["pass"]:
            failed.append("controls_kept")
        raise RuntimeError(f"split checks failed: {failed}; see {out / 'manifest.json'}")
    return manifest


def check_output(path: pathlib.Path, genes: list[str], vocab: list[str], lines: list[str]) -> dict:
    """Re-open a written h5ad and check the promised layout."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        enc = X.attrs.get("encoding-type")
        if enc != "csr_matrix" or X["data"].dtype != np.int32:
            raise RuntimeError(f"{path}: X is {enc} {X['data'].dtype}, expected csr_matrix int32")
        if list(read_index(f["var"])) != genes:
            raise RuntimeError(f"{path}: var index is not the metadata genes in order")
        if "layers" in f and len(f["layers"]):
            raise RuntimeError(f"{path}: unexpected layers {list(f['layers'])}")
        for col, cats in (("gene", vocab), ("cell_line", lines)):
            node = f["obs"][col]
            if not (isinstance(node, h5py.Group) and list(read_column(node, "categories")) == cats):
                raise RuntimeError(f"{path}: obs[{col!r}] is not categorical with the metadata categories")
        ids = read_index(f["obs"])
        if not pd.Index(ids).is_unique:
            raise RuntimeError(f"{path}: obs index not unique")
        shape = [int(x) for x in X.attrs["shape"]]
        return {
            "path": str(path),
            "size": path.stat().st_size,
            "cells": shape[0],
            "genes": shape[1],
            "nnz": int(X["data"].shape[0]),
        }


def split_counts(obs: pd.DataFrame, control: str) -> dict:
    """Cells, targets and controls per split and line (plus totals)."""
    out = {}
    for split in SPLITS:
        d = obs[obs.split == split]
        per_line = {}
        for line, g in d.groupby("cell_line"):
            per_line[str(line)] = {
                "cells": int(len(g)),
                "targets": int(g.gene[g.gene != control].nunique()),
                "controls": int((g.gene == control).sum()),
            }
        per_line["total"] = {
            "cells": int(len(d)),
            "targets": int(d.gene[d.gene != control].nunique()),
            "controls": int((d.gene == control).sum()),
        }
        out[split] = per_line
    return out


def split_checks(obs: pd.DataFrame, test_targets: list[str], cfg: DictConfig, required: bool) -> dict:
    """The split rules for one version; ``required`` marks which rules gate the run."""
    ctx, ctrl = cfg.target_context, cfg.control
    train, test = obs[obs.split == "train"], obs[obs.split == "test"]
    ctx_train = set(train.gene[(train.cell_line == ctx) & (train.gene != ctrl)])
    leaked = sorted(ctx_train & set(test_targets))
    donor = train[(train.cell_line != ctx) & (train.gene != ctrl)]
    donor_targets = set(donor.gene)
    no_donor = sorted(set(test_targets) - donor_targets)
    n_test = test.gene.value_counts().reindex(test_targets, fill_value=0)
    small = {str(k): int(v) for k, v in n_test[n_test < int(cfg.split.min_test_cells)].items()}
    wrong_test = int(((test.cell_line != ctx) | ~test.gene.isin(test_targets)).sum())
    rules = {
        "no_test_target_in_target_line_train": {"pass": not leaked, "required": True, "targets": leaked},
        "test_target_in_a_donor_line": {"pass": not no_donor, "required": True, "targets": no_donor},
        "test_only_target_line_test_targets": {"pass": wrong_test == 0, "required": True, "cells": wrong_test},
        "test_target_min_cells": {
            "pass": not small,
            "required": required,
            "min_cells": int(cfg.split.min_test_cells),
            "targets_below": small,
            "n_test_targets_with_cells": int((n_test > 0).sum()),
            "cells_per_test_target": {
                "min": int(n_test.min()),
                "median": float(n_test.median()),
                "max": int(n_test.max()),
            },
        },
    }
    donors_per_target = donor.groupby("gene", observed=True).cell_line.nunique().reindex(test_targets, fill_value=0)
    return {
        "rules": rules,
        "pass": all(r["pass"] for r in rules.values() if r["required"]),
        "donor_lines_per_test_target": {
            str(k): int(v) for k, v in donors_per_target.value_counts().sort_index().items()
        },
    }


def controls_check(obs_tables: dict[str, pd.DataFrame], control: str) -> dict:
    """Controls are never filtered: per line, the filtered version holds every control of the all version."""
    per = {}
    for v in ("filtered", "all"):
        o = obs_tables[v]
        per[v] = o[o.gene == control].groupby("cell_line").size().to_dict()
    ok = per["filtered"] == per["all"]
    return {"pass": ok, "controls": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. out_dir=/tmp/x block=5000")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    manifest = prepare(cfg)
    for v, c in manifest["counts"].items():
        print(
            f"{v}: train {c['train']['total']['cells']} cells / {c['train']['total']['targets']} targets, "
            f"test {c['test']['total']['cells']} cells / {c['test']['total']['targets']} targets"
        )
    print(f"manifest: {pathlib.Path(cfg.out_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
