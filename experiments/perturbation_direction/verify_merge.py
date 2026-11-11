"""Check the raw rebuild against STATE (identity gates) and compare the filtered and unfiltered versions.

Subcommands
-----------
``checksums`` (raw inputs; exit code 1 if any file fails)
    Recomputes the md5 of every raw screen in the config (``--jobs`` files at a time, 64 MB reads) and
    compares it with the config's ``md5`` AND the ``<file>.md5`` sidecar written at download time (a
    missing one fails when ``require_md5``). Writes ``<out_dir>/checksums.json`` with size and mtime per
    file, which prepare_replogle_raw.py copies into its manifest as ``md5_computed``.

``identity`` (Tier A gates A1-A4; exit code 1 if any fails)
    Our primary (``filtered``) train/test h5ads from prepare_replogle_raw.py against
    ``state_reference.parquet`` from make_state_reference.py.

    A1  per line, our filtered cell set equals STATE's, and every matched cell has the same target
        label (gem_group and UMI_count are compared too). Reports only-ours / only-STATE counts.
        Also fails on a cell id that occurs more than once in a line on either side (within a split or
        across train/test: ``A1_duplicated_ours`` / ``A1_duplicated_state``; the first copy takes part
        in the mapping, the others count as unmatched), on a line mapped by ``key`` unless
        ``--allow-key-mapping`` (``A1_key_mapping``), and on a barcode-mapped pair whose cell id is not
        the raw id its STATE id resolved to (``A1_cell_id``).
    A2  ``counts_hash`` (sha1 of the int32 2,000-gene count vector) is equal for every matched cell.
    A3  the full-file library is the same integer: ``|library_full - lib_full_state| < 0.5`` and
        ``rint(lib_full_state) == library_full`` for every matched cell (STATE's implied library is
        within 0.008 of an integer for all 643,413 STATE cells).
    A4  our filtered test cells == STATE's test cells, our filtered train cells == STATE's train
        cells; every difference is listed by line, target and reason.

    Cell mapping, decided per line against the raw cell ids of the line (the unfiltered ``all`` version
    when it exists, so a filter difference cannot change the mapping):
      ``obs_index``              every STATE cell id is a raw cell id;
      ``obs_index_make_unique``  as above after removing the ``-<k>`` suffix anndata's
                                 ``obs_names_make_unique`` gave barcodes that repeat across screens when
                                 STATE merged them (hepg2, jurkat, k562, rpe1 order; observed on the real
                                 files: 92 jurkat, 263 k562, 422 rpe1 cells, all ``-1``);
      ``key``                    otherwise (gem_group, UMI_count, gene), with the counts hash appended
                                 only where that key repeats. Under ``key`` mapping A2 cannot fail for
                                 the tie-broken cells; their number is reported. A1 fails on a ``key``
                                 line unless ``--allow-key-mapping`` is given.

``truth`` (A5; exit code 1 if it fails)
    Two ground_truth.py output dirs: delta_true / padj max |diff| <= tol (default 1e-9, NaN == NaN),
    identical deg masks, identical ``info.json`` n_cells, same perturbations and genes.

``compare`` (Tier B; a report, never a gate)
    Two ground_truth.py dirs, ``filtered`` vs ``all``. Per perturbation: n_cells and n_DEG in both,
    Pearson and OLS slope of delta_all on delta_filtered, DEG Jaccard; computed without the
    perturbation's own target gene (as evaluate.py --exclude-target scores), whose delta is reported
    separately. Panel sizes with >= --min-degs DEGs, the target gene not counted (as evaluate.py
    --exclude-target selects its panel), and summary quantiles. Writes JSON + CSV.

Usage::

    python verify_merge.py checksums --config experiments/configs/perturbation_direction/replogle_raw.yaml
    python verify_merge.py identity --config experiments/configs/perturbation_direction/replogle_raw.yaml
    python verify_merge.py truth --ours <truth_filtered> --reference <truth_state> --out a5.json
    python verify_merge.py compare --filtered <truth_filtered> --all <truth_all> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from audit_raw_screen import quantiles, read_column, to_jsonable
from make_state_reference import counts_hash, csr_rows, file_digest

KEY_COLS = ("gem_group", "UMI_count", "gene")


# ---------------------------------------------------------------------------
# raw inputs: recomputed md5
# ---------------------------------------------------------------------------
def _md5_timed(path: pathlib.Path) -> tuple[str, float]:
    t0 = time.time()
    return file_digest(path, "md5"), round(time.time() - t0, 1)


def checksums(cfg, jobs: int = 4) -> dict:
    """md5 of every raw screen, recomputed (``jobs`` files in parallel), vs the config md5 and the sidecar."""
    from prepare_replogle_raw import read_md5_file

    recs, todo = [], {}
    for s in cfg.screens:
        p = pathlib.Path(cfg.raw_dir) / s.file
        rec = {
            "name": s.name,
            "cell_line": s.cell_line,
            "path": str(p),
            "md5_expected": s.get("md5"),
            "md5_source": s.get("md5_source"),
            "expected_size": s.get("size"),
        }
        if p.exists():
            st = p.stat()
            rec.update(size=st.st_size, mtime_ns=st.st_mtime_ns, md5_sidecar=read_md5_file(p))
            todo[s.name] = p
        recs.append(rec)
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(todo) or 1))) as ex:  # hashlib releases the GIL
        futures = {name: ex.submit(_md5_timed, p) for name, p in todo.items()}
        done = {name: f.result() for name, f in futures.items()}
    for rec in recs:
        problems = []
        if rec["name"] not in done:
            problems.append("file missing")
        else:
            rec["md5_computed"], rec["seconds"] = done[rec["name"]]
            for k, ref in (("match_expected", "md5_expected"), ("match_sidecar", "md5_sidecar")):
                rec[k] = None if rec[ref] is None else rec["md5_computed"] == str(rec[ref]).lower()
                if rec[k] is False:
                    problems.append(f"computed {rec['md5_computed']} != {ref} {rec[ref]}")
                elif rec[k] is None and cfg.require_md5:
                    problems.append(f"no {ref} (require_md5)")
            if rec["expected_size"] is not None and rec["size"] != rec["expected_size"]:
                problems.append(f"size {rec['size']} != expected {rec['expected_size']}")
            st = pathlib.Path(rec["path"]).stat()  # the file must not change while it is hashed
            if (st.st_size, st.st_mtime_ns) != (rec["size"], rec["mtime_ns"]):
                problems.append("file changed while it was hashed")
        rec["problems"] = problems
        rec["pass"] = not problems
    return {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs": jobs,
        "require_md5": bool(cfg.require_md5),
        "files": recs,
        "pass": all(r["pass"] for r in recs),
    }


def run_checksums(args) -> int:
    from prepare_replogle_raw import CHECKSUMS, load_config

    cfg = load_config(args.config, args.overrides)
    report = checksums(cfg, jobs=args.jobs)
    out = pathlib.Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (CHECKSUMS + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(report), indent=1))
    os.replace(tmp, out / CHECKSUMS)
    for r in report["files"]:
        state = "PASS" if r["pass"] else "FAIL " + "; ".join(r["problems"])
        took = f" ({r['seconds']}s)" if "seconds" in r else ""
        print(f"  {r['name']}: md5 {r.get('md5_computed', '-')}{took} {state}")
    print(f"checksums: {'PASS' if report['pass'] else 'FAIL'} -> {out / CHECKSUMS}")
    return 0 if report["pass"] else 1


# ---------------------------------------------------------------------------
# Tier A: identity
# ---------------------------------------------------------------------------
def load_ours(paths: dict[str, str | pathlib.Path], *, with_hash: bool = True, block: int = 20_000) -> pd.DataFrame:
    """Obs of our prepared files (split -> path) plus, optionally, each cell's counts hash."""
    frames = []
    for split, path in paths.items():
        with h5py.File(path, "r") as f:
            cols = {c: read_column(f["obs"], c) for c in ("cell_id", "cell_line", "gene", "gem_group", "UMI_count")}
            if "library_full" in f["obs"]:
                cols["library_full"] = read_column(f["obs"], "library_full")
            df = pd.DataFrame(cols)
            df["split"] = split
            if with_hash:
                X = f["X"]
                n, m = (int(s) for s in X.attrs["shape"])
                hashes: list[str] = []
                for start in range(0, n, block):
                    hashes.extend(counts_hash(csr_rows(X, start, min(start + block, n), m).astype("<i4")))
                df["counts_hash"] = hashes
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in ("cell_id", "cell_line", "gene"):
        out[c] = out[c].astype(str)
    return out


def _key(df: pd.DataFrame) -> pd.Series:
    return (
        df["gem_group"].astype(np.int64).astype(str)
        + "|"
        + df["UMI_count"].astype(np.float64).map(repr)
        + "|"
        + df["gene"].astype(str)
    )


def resolve_state_ids(state_ids: pd.Series, universe: set[str]) -> tuple[pd.Series, pd.Series] | None:
    """Raw cell id behind each STATE cell id, or None when the ids cannot be resolved one to one.

    STATE merged the four screens and made the obs names unique the anndata way: the first occurrence of
    a barcode keeps it, later ones (in the line order hepg2, jurkat, k562, rpe1) get ``-1``, ``-2``, ...
    So a STATE id is either a raw id of its line, or a raw id plus one ``-<k>`` suffix. A suffixed id is
    accepted only if its base is a raw id of the line that no unsuffixed STATE id of the line claims.
    Returns (raw id per STATE row, the suffixed STATE ids).
    """
    direct = state_ids.isin(universe)
    rest = state_ids[~direct]
    base = rest.str.replace(r"-\d+$", "", regex=True)
    ok = (base != rest) & base.isin(universe) & ~base.isin(set(state_ids[direct]))
    if not ok.all():
        return None
    raw = state_ids.copy()
    raw[~direct] = base
    if raw.duplicated().any():
        return None
    return raw, rest


def map_line(ours: pd.DataFrame, state: pd.DataFrame, universe: set[str] | None) -> tuple[pd.Series, dict]:
    """Map one line's STATE rows (index) to our rows (index); returns (Series state_row -> our_row, info).

    Cell ids must be unique on both sides (identity() reports duplicates and passes the first copy).
    For the barcode methods ``info["_raw_ids"]`` holds the raw id each STATE row resolved to.
    """
    for name, df in (("ours", ours), ("STATE", state)):
        if not df.cell_id.is_unique:
            raise ValueError(f"map_line: duplicated cell ids in {name}")
    universe = set(ours.cell_id) if universe is None else universe
    found = state.cell_id.isin(universe)
    resolved = resolve_state_ids(state.cell_id, universe)
    if resolved is not None:
        raw, suffixed = resolved
        pos = pd.Series(ours.index, index=ours.cell_id)
        hit = raw[raw.isin(pos.index)]
        mapping = pd.Series(pos.loc[hit.to_numpy()].to_numpy(), index=hit.index)
        return mapping, {
            "method": "obs_index" if suffixed.empty else "obs_index_make_unique",
            "state_ids_in_raw": int(found.sum()),
            "state_cells": int(len(state)),
            "make_unique_suffix_stripped": int(len(suffixed)),
            "examples_suffixed": suffixed.head(10).tolist(),
            "_suffix_bases": raw[suffixed.index],
            "_raw_ids": raw,
        }
    k_s, k_o = _key(state), _key(ours)
    dup = set(k_s[k_s.duplicated(keep=False)]) | set(k_o[k_o.duplicated(keep=False)])
    tie_s, tie_o = k_s.isin(dup), k_o.isin(dup)
    k_s = k_s.where(~tie_s, k_s + "|" + state.counts_hash)
    k_o = k_o.where(~tie_o, k_o + "|" + ours.counts_hash)
    amb_s, amb_o = set(k_s[k_s.duplicated(keep=False)]), set(k_o[k_o.duplicated(keep=False)])
    ok_o = k_o[~k_o.isin(amb_o)]
    pos = pd.Series(ok_o.index, index=ok_o.to_numpy())
    usable = k_s[~k_s.isin(amb_s) & k_s.isin(pos.index)]
    mapping = pd.Series(pos.loc[usable.to_numpy()].to_numpy(), index=usable.index)
    return mapping, {
        "method": "key",
        "key": list(KEY_COLS),
        "state_ids_in_raw": int(found.sum()),
        "state_cells": int(len(state)),
        "tie_broken_by_counts_hash": int(tie_s.sum()),
        "ambiguous_state_cells": int(k_s.isin(amb_s).sum()),
        "ambiguous_our_cells": int(k_o.isin(amb_o).sum()),
    }


def _duplicate_detail(df: pd.DataFrame, side: str) -> list[str]:
    """One note per row of ``df`` (rows sharing a cell id): how often and in which splits it occurs."""
    if df.empty:
        return []
    n = df.cell_id.map(df.cell_id.value_counts())
    splits = df.cell_id.map(df.groupby("cell_id").split.agg(lambda x: ",".join(map(str, x))))
    return [f"cell id {k}x in {side} (splits {sp})" for k, sp in zip(n, splits, strict=True)]


def identity(
    ours: pd.DataFrame,
    state: pd.DataFrame,
    *,
    universe: dict[str, set[str]] | None = None,
    allow_key_mapping: bool = False,
    max_examples: int = 20,
) -> tuple[dict, pd.DataFrame]:
    """Gates A1-A4; returns (report, per-cell problem table).

    A line mapped by the (gem_group, UMI_count, gene) key fails A1 unless ``allow_key_mapping``.
    """
    ours = ours.reset_index(drop=True)
    state = state.reset_index(drop=True)
    problems: list[pd.DataFrame] = []

    def problem(gate: str, df: pd.DataFrame, detail) -> None:
        if len(df):
            problems.append(
                pd.DataFrame(
                    {
                        "gate": gate,
                        "cell_line": df.cell_line.to_numpy(),
                        "cell_id": df.cell_id.to_numpy(),
                        "gene": df.gene.to_numpy(),
                        "detail": detail,
                    }
                )
            )

    lines = sorted(set(ours.cell_line) | set(state.cell_line))
    # STATE's line order (first appearance in the reference, which keeps STATE row order): a make-unique
    # suffix '-<k>' is only legitimate when the bare barcode is already a STATE cell of an EARLIER line
    line_order = list(dict.fromkeys(state.cell_line))
    a1, mapping_info, maps = {}, {}, []
    for line in lines:
        o_all, s_all = ours[ours.cell_line == line], state[state.cell_line == line]
        # a cell id twice in a line (within a split or across train/test) is a failure, not a crash: the
        # first copy takes part in the mapping, the others stay unmatched
        dup_o = o_all.cell_id.duplicated(keep=False).to_numpy()
        dup_s = s_all.cell_id.duplicated(keep=False).to_numpy()
        problem("A1_duplicated_ours", o_all[dup_o], _duplicate_detail(o_all[dup_o], "ours"))
        problem("A1_duplicated_state", s_all[dup_s], _duplicate_detail(s_all[dup_s], "STATE"))
        o, s = o_all[~o_all.cell_id.duplicated()], s_all[~s_all.cell_id.duplicated()]

        mp, info = map_line(o, s, None if universe is None else universe.get(line))
        raw_ids = info.pop("_raw_ids", None)
        bases = info.pop("_suffix_bases", None)
        if bases is not None and len(bases):
            info["suffix_base_is_state_id_of_another_line"] = int(
                state.cell_id[state.cell_line != line].isin(set(bases)).sum()
            )
            earlier = line_order[: line_order.index(line)] if line in line_order else []
            ok = bases.isin(set(state.cell_id[state.cell_line.isin(earlier)])).to_numpy()
            info["suffix_base_is_state_id_of_earlier_line"] = int(ok.sum())
            problem(
                "A1_suffix_base",
                s.loc[bases.index[~ok]],
                "make-unique suffix, but the bare barcode is not a STATE cell of an earlier line",
            )
        info["allowed"] = info["method"] != "key" or allow_key_mapping
        mapping_info[line] = info
        if not info["allowed"]:
            problems.append(
                pd.DataFrame(
                    {
                        "gate": ["A1_key_mapping"],
                        "cell_line": [line],
                        "cell_id": [""],
                        "gene": [""],
                        "detail": [
                            f"STATE ids do not resolve to raw ids ({info['state_ids_in_raw']}/{info['state_cells']} "
                            "found); mapped by (gem_group, UMI_count, gene), refused without --allow-key-mapping"
                        ],
                    }
                )
            )
        so, oo = s.loc[mp.index], o.loc[mp.to_numpy()]
        if raw_ids is None:  # key mapping: there is no resolved raw id to check against
            cell_id_mismatch = None
            info["pairs_with_equal_cell_id"] = int((so.cell_id.to_numpy() == oo.cell_id.to_numpy()).sum())
        else:
            want = raw_ids.loc[mp.index].to_numpy()
            bad_id = oo.cell_id.to_numpy() != want
            cell_id_mismatch = int(bad_id.sum())
            problem(
                "A1_cell_id",
                so[bad_id],
                [
                    f"state={a} resolved_raw={b} ours={c}"
                    for a, b, c in zip(so.cell_id[bad_id], want[bad_id], oo.cell_id[bad_id], strict=True)
                ],
            )
        only_state, only_ours = s_all.drop(index=mp.index), o_all.drop(index=mp.to_numpy())
        diff = {}
        for c in KEY_COLS:
            a, b = so[c].to_numpy(), oo[c].to_numpy()
            if c == "gene":
                a, b = a.astype(str), b.astype(str)
            else:
                a, b = a.astype(np.float64), b.astype(np.float64)
            bad = a != b
            diff[c] = int(bad.sum())
            problem(f"A1_{c}", so[bad], [f"state={x} ours={y}" for x, y in zip(a[bad], b[bad], strict=True)])
        problem("A1_only_state", only_state, "STATE cell not in our filtered set")
        problem("A1_only_ours", only_ours, "our filtered cell not in STATE")
        a1[line] = {
            "state_cells": int(len(s_all)),
            "our_cells": int(len(o_all)),
            "matched": int(len(mp)),
            "only_state": int(len(only_state)),
            "only_ours": int(len(only_ours)),
            "label_mismatch": diff["gene"],
            "gem_group_mismatch": diff["gem_group"],
            "UMI_count_mismatch": diff["UMI_count"],
            "duplicated_ours": int(dup_o.sum()),
            "duplicated_state": int(dup_s.sum()),
            "mapping_method": info["method"],
            "mapping_allowed": info["allowed"],
            "cell_id_mismatch": cell_id_mismatch,
            "examples_only_state": only_state.cell_id.head(max_examples).tolist(),
            "examples_only_ours": only_ours.cell_id.head(max_examples).tolist(),
            "examples_duplicated_ours": sorted(set(o_all.cell_id[dup_o]))[:max_examples],
            "examples_duplicated_state": sorted(set(s_all.cell_id[dup_s]))[:max_examples],
            "pass": (
                len(only_state) == 0
                and len(only_ours) == 0
                and not any(diff.values())
                and not dup_o.any()
                and not dup_s.any()
                and info["allowed"]
                and not cell_id_mismatch
                and info.get("suffix_base_is_state_id_of_earlier_line", 0) == info.get("make_unique_suffix_stripped", 0)
            ),
        }
        maps.append(pd.DataFrame({"s": mp.index.to_numpy(), "o": mp.to_numpy()}))
    m = pd.concat(maps, ignore_index=True) if maps else pd.DataFrame({"s": [], "o": []}, dtype=np.int64)
    S, O = state.loc[m.s].reset_index(drop=True), ours.loc[m.o].reset_index(drop=True)

    bad_hash = S.counts_hash.to_numpy() != O.counts_hash.to_numpy()
    problem("A2_counts_hash", S[bad_hash], "counts differ")
    a2 = {
        "matched": int(len(S)),
        "mismatch": int(bad_hash.sum()),
        "per_line": {line: int(bad_hash[S.cell_line == line].sum()) for line in lines},
        "examples": S.cell_id[bad_hash].head(max_examples).tolist(),
        "pass": not bad_hash.any(),
    }

    # A3: the same integer library. lib_full_state is implied by STATE's float32 X (within 0.008 of an
    # integer on every real STATE cell), library_full is our integer count total over the raw file.
    ls, lo = S.lib_full_state.to_numpy(np.float64), O.library_full.to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        absd = np.abs(lo - ls)
        rel = absd / ls
        ok_lib = np.isfinite(ls) & (absd < 0.5) & (np.rint(ls) == lo)
    bad_lib = ~ok_lib  # NaN (no model-gene count in STATE) fails too
    problem(
        "A3_library",
        S[bad_lib],
        [
            f"ours={x:.0f} state={y:.6f} abs={a:.3g}"
            for x, y, a in zip(lo[bad_lib], ls[bad_lib], absd[bad_lib], strict=True)
        ],
    )

    def _max(x: np.ndarray) -> float | None:
        return float(np.nanmax(x)) if np.isfinite(x).any() else None

    a3 = {
        "rule": "|library_full - lib_full_state| < 0.5 and rint(lib_full_state) == library_full",
        "matched": int(len(S)),
        "fail": int(bad_lib.sum()),
        "nan": int(np.isnan(ls).sum()),
        "max_abs_diff": _max(absd),
        "abs_diff": quantiles(absd),
        "max_rel_diff": _max(rel),
        "rel_diff": quantiles(rel),
        "per_line_max_abs": {line: _max(absd[(S.cell_line == line).to_numpy()]) for line in lines},
        "pass": not bad_lib.any(),
    }

    # A4: split identity over the union of both cell sets (matched pairs + cells on one side only)
    only_s = state.drop(index=m.s)
    only_o = ours.drop(index=m.o)
    joined = pd.concat(
        [
            pd.DataFrame(
                {"cell_line": S.cell_line, "cell_id": S.cell_id, "gene": S.gene, "ours": O.split, "state": S.split}
            ),
            pd.DataFrame(
                {
                    "cell_line": only_s.cell_line,
                    "cell_id": only_s.cell_id,
                    "gene": only_s.gene,
                    "ours": None,
                    "state": only_s.split,
                }
            ),
            pd.DataFrame(
                {
                    "cell_line": only_o.cell_line,
                    "cell_id": only_o.cell_id,
                    "gene": only_o.gene,
                    "ours": only_o.split,
                    "state": None,
                }
            ),
        ],
        ignore_index=True,
    )
    joined[["ours", "state"]] = joined[["ours", "state"]].fillna("-")
    diff = joined[joined.ours != joined.state].copy()
    diff["reason"] = np.select(
        [diff.ours == "-", diff.state == "-"],
        ["only in STATE (not in our filtered set)", "only in ours (not in STATE)"],
        default="split differs",
    )
    problem(
        "A4_split",
        diff,
        [f"{r}: ours={o} state={s}" for r, o, s in zip(diff.reason, diff.ours, diff.state, strict=True)],
    )
    groups = (
        diff.groupby(["cell_line", "gene", "reason", "ours", "state"])
        .size()
        .rename("cells")
        .reset_index()
        .sort_values("cells", ascending=False)
    )
    a4 = {}
    for split in ("train", "test"):
        o_in, s_in = (joined.ours == split).to_numpy(), (joined.state == split).to_numpy()
        a4[split] = {
            "ours": int(o_in.sum()),
            "state": int(s_in.sum()),
            "only_ours": int((o_in & ~s_in).sum()),
            "only_state": int((s_in & ~o_in).sum()),
            "pass": bool((o_in == s_in).all()),
        }
    a4["differences"] = groups.head(100).to_dict(orient="records")
    a4["pass"] = a4["train"]["pass"] and a4["test"]["pass"]

    report = {
        "mapping": mapping_info,
        "allow_key_mapping": bool(allow_key_mapping),
        "A1_cell_set_and_labels": {"per_line": a1, "pass": all(v["pass"] for v in a1.values())},
        "A2_counts_hash": a2,
        "A3_library": a3,
        "A4_split": a4,
    }
    report["pass"] = all(
        report[k]["pass"] for k in ("A1_cell_set_and_labels", "A2_counts_hash", "A3_library", "A4_split")
    )
    table = (
        pd.concat(problems, ignore_index=True)
        if problems
        else pd.DataFrame(columns=["gate", "cell_line", "cell_id", "gene", "detail"])
    )
    return report, table


def run_identity(args) -> int:
    cfg = None
    if args.config:
        from prepare_replogle_raw import load_config, output_path

        cfg = load_config(args.config, args.overrides)
        version = args.version or cfg.primary_version
        args.train = args.train or str(output_path(cfg.out_dir, "train", version))
        args.test = args.test or str(output_path(cfg.out_dir, "test", version))
        args.state_ref = args.state_ref or cfg.state_reference
        args.out_dir = args.out_dir or str(pathlib.Path(cfg.out_dir) / "verify")
        if args.universe_train is None and "all" in cfg.versions:
            args.universe_train = str(output_path(cfg.out_dir, "train", "all"))
            args.universe_test = str(output_path(cfg.out_dir, "test", "all"))
    missing = [k for k in ("train", "test", "state_ref", "out_dir") if not getattr(args, k)]
    if missing:
        raise SystemExit(f"identity: need --config or --{', --'.join(m.replace('_', '-') for m in missing)}")

    ours = load_ours({"train": args.train, "test": args.test}, block=args.block)
    state = pd.read_parquet(args.state_ref)
    universe = None
    if args.universe_train:
        u = load_ours(
            {k: p for k, p in (("train", args.universe_train), ("test", args.universe_test)) if p}, with_hash=False
        )
        universe = {line: set(g.cell_id) for line, g in u.groupby("cell_line")}
    report, table = identity(ours, state, universe=universe, allow_key_mapping=args.allow_key_mapping)
    report["inputs"] = {
        "train": args.train,
        "test": args.test,
        "state_reference": args.state_ref,
        "universe": [args.universe_train, args.universe_test] if args.universe_train else None,
    }
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity.json").write_text(json.dumps(to_jsonable(report), indent=1))
    table.to_csv(out / "identity_problems.csv", index=False)
    for gate in ("A1_cell_set_and_labels", "A2_counts_hash", "A3_library", "A4_split"):
        print(f"{gate}: {'PASS' if report[gate]['pass'] else 'FAIL'}")
    for line, r in report["A1_cell_set_and_labels"]["per_line"].items():
        refused = "" if r["mapping_allowed"] else " (REFUSED: pass --allow-key-mapping to accept it)"
        print(
            f"  {line}: mapping={r['mapping_method']}{refused} state={r['state_cells']} ours={r['our_cells']} "
            f"only_state={r['only_state']} only_ours={r['only_ours']} label_mismatch={r['label_mismatch']} "
            f"duplicated ours/state={r['duplicated_ours']}/{r['duplicated_state']} "
            f"cell_id_mismatch={r['cell_id_mismatch']}"
        )
    print(f"identity: {'PASS' if report['pass'] else 'FAIL'} -> {out / 'identity.json'}")
    return 0 if report["pass"] else 1


# ---------------------------------------------------------------------------
# A5: truth identity
# ---------------------------------------------------------------------------
def read_truth(d: str | pathlib.Path) -> dict:
    d = pathlib.Path(d)
    info = json.loads((d / "info.json").read_text())
    return {
        "delta": pd.read_parquet(d / "delta_true.parquet"),
        "padj": pd.read_parquet(d / "padj.parquet"),
        "deg": pd.read_parquet(d / "deg.parquet"),
        "n_cells": {str(k): int(v) for k, v in info.get("n_cells", {}).items()},
        "info": info,
    }


def truth_identity(ours: dict, ref: dict, tol: float = 1e-9) -> dict:
    """A5: two ground-truth dirs are the same numbers."""
    out = {"tol": tol}
    d0 = ref["delta"]
    same_rows = set(ours["delta"].index) == set(d0.index)
    same_cols = set(ours["delta"].columns) == set(d0.columns)
    out["perturbations"] = {
        "ours": int(len(ours["delta"])),
        "reference": int(len(d0)),
        "only_ours": sorted(set(ours["delta"].index) - set(d0.index))[:20],
        "only_reference": sorted(set(d0.index) - set(ours["delta"].index))[:20],
        "pass": same_rows,
    }
    out["genes"] = {"pass": same_cols, "ours": int(ours["delta"].shape[1]), "reference": int(d0.shape[1])}
    rows = d0.index.intersection(ours["delta"].index)
    cols = d0.columns.intersection(ours["delta"].columns)
    for name in ("delta", "padj"):
        a = ours[name].reindex(index=rows, columns=cols).to_numpy(np.float64)
        b = ref[name].reindex(index=rows, columns=cols).to_numpy(np.float64)
        nan_diff = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        mx = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        out[name] = {"max_abs_diff": mx, "nan_pattern_diff": nan_diff, "pass": mx <= tol and nan_diff == 0}
    a = ours["deg"].reindex(index=rows, columns=cols).fillna(False).to_numpy(bool)
    b = ref["deg"].reindex(index=rows, columns=cols).fillna(False).to_numpy(bool)
    out["deg"] = {"differing_entries": int((a != b).sum()), "pass": bool((a == b).all())}
    nc_o, nc_r = ours["n_cells"], ref["n_cells"]
    diff_n = {k: [nc_o.get(k), nc_r.get(k)] for k in sorted(set(nc_o) | set(nc_r)) if nc_o.get(k) != nc_r.get(k)}
    out["n_cells"] = {"differing": len(diff_n), "examples": dict(list(diff_n.items())[:20]), "pass": not diff_n}
    out["pass"] = all(out[k]["pass"] for k in ("perturbations", "genes", "delta", "padj", "deg", "n_cells"))
    return out


def run_truth(args) -> int:
    report = truth_identity(read_truth(args.ours), read_truth(args.reference), tol=args.tol)
    report["inputs"] = {"ours": args.ours, "reference": args.reference}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(report), indent=1))
    for k in ("perturbations", "genes", "delta", "padj", "deg", "n_cells"):
        print(f"A5 {k}: {'PASS' if report[k]['pass'] else 'FAIL'}")
    print(f"A5 truth identity: {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


# ---------------------------------------------------------------------------
# Tier B: filtered vs all
# ---------------------------------------------------------------------------
def _pearson_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy = float(xc @ xc), float(yc @ yc)
    if sxx == 0:
        return np.nan, np.nan
    sxy = float(xc @ yc)
    return (sxy / np.sqrt(sxx * syy) if syy > 0 else np.nan), sxy / sxx


def compare_versions(filtered: dict, all_: dict, *, min_degs: int = 5) -> tuple[pd.DataFrame, dict]:
    """Per-perturbation agreement of the unfiltered truth with the filtered one (target gene excluded)."""
    df_, da = filtered["delta"], all_["delta"]
    genes = df_.columns.intersection(da.columns)
    perts = sorted(set(df_.index) | set(da.index))
    rows = []
    for p in perts:
        in_f, in_a = p in df_.index, p in da.index
        r = {
            "perturbation": p,
            "in_filtered": in_f,
            "in_all": in_a,
            "n_cells_filtered": filtered["n_cells"].get(p),
            "n_cells_all": all_["n_cells"].get(p),
            "n_deg_filtered": int(filtered["deg"].loc[p].sum()) if in_f else None,
            "n_deg_all": int(all_["deg"].loc[p].sum()) if in_a else None,
        }
        if in_f and in_a:
            keep = genes[genes != p]
            x = df_.loc[p, keep].to_numpy(np.float64)
            y = da.loc[p, keep].to_numpy(np.float64)
            r["pearson"], r["slope_all_on_filtered"] = _pearson_slope(x, y)
            a = filtered["deg"].loc[p, keep].to_numpy(bool)
            b = all_["deg"].loc[p, keep].to_numpy(bool)
            union = int((a | b).sum())
            r["deg_jaccard"] = (a & b).sum() / union if union else np.nan
            r["n_deg_filtered_excl_target"], r["n_deg_all_excl_target"] = int(a.sum()), int(b.sum())
            if p in genes:
                r["delta_target_filtered"], r["delta_target_all"] = float(df_.loc[p, p]), float(da.loc[p, p])
        rows.append(r)
    table = pd.DataFrame(rows).set_index("perturbation")
    both = table[table.in_filtered & table.in_all]
    # the panel evaluate.py --exclude-target scores: DEGs counted without the perturbation's own target
    sel_f = set(both.index[both.n_deg_filtered_excl_target >= min_degs]) if len(both) else set()
    sel_a = set(both.index[both.n_deg_all_excl_target >= min_degs]) if len(both) else set()
    summary = {
        "perturbations": {
            "filtered": int(table.in_filtered.sum()),
            "all": int(table.in_all.sum()),
            "both": int(len(both)),
            "only_filtered": sorted(table.index[table.in_filtered & ~table.in_all]),
            "only_all": sorted(table.index[~table.in_filtered & table.in_all]),
        },
        "panel_min_degs": {
            "min_degs": min_degs,
            "deg_count": "target gene excluded (n_deg_*_excl_target), as evaluate.py --exclude-target",
            "filtered": len(sel_f),
            "all": len(sel_a),
            "both": len(sel_f & sel_a),
            "either": len(sel_f | sel_a),
        },
        "target_gene_excluded": True,
    }
    for c in (
        "pearson",
        "slope_all_on_filtered",
        "deg_jaccard",
        "n_deg_filtered",
        "n_deg_all",
        "n_deg_filtered_excl_target",
        "n_deg_all_excl_target",
        "n_cells_filtered",
        "n_cells_all",
    ):
        if c in both:
            summary[c] = quantiles(both[c].to_numpy(np.float64))
    panel = both.loc[sorted(sel_f & sel_a)]
    summary["on_panel_both"] = {
        c: quantiles(panel[c].to_numpy(np.float64))
        for c in ("pearson", "slope_all_on_filtered", "deg_jaccard")
        if c in panel
    }
    if "delta_target_filtered" in both:
        summary["delta_target"] = {
            "filtered": quantiles(both.delta_target_filtered.to_numpy(np.float64)),
            "all": quantiles(both.delta_target_all.to_numpy(np.float64)),
        }
    return table, summary


def run_compare(args) -> int:
    table, summary = compare_versions(read_truth(args.filtered), read_truth(args.all), min_degs=args.min_degs)
    summary["inputs"] = {"filtered": args.filtered, "all": args.all}
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "filtered_vs_all.csv")
    (out / "filtered_vs_all.json").write_text(json.dumps(to_jsonable(summary), indent=1))
    q = summary.get("pearson") or {}
    print(
        f"Tier B: {summary['perturbations']['both']} perturbations in both; median Pearson {q.get('0.5')}, "
        f"median slope {(summary.get('slope_all_on_filtered') or {}).get('0.5')}, "
        f"panel >= {args.min_degs} DEGs: filtered {summary['panel_min_degs']['filtered']}, "
        f"all {summary['panel_min_degs']['all']} -> {out}"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("checksums", help="recompute the raw files' md5; compare with config and .md5 sidecar")
    p.add_argument("--config", required=True, help="replogle_raw.yaml")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides for --config")
    p.add_argument("--jobs", type=int, default=4, help="files hashed in parallel (64 MB buffer each)")
    p.set_defaults(func=run_checksums)

    p = sub.add_parser("identity", help="Tier A gates A1-A4 (filtered version vs STATE)")
    p.add_argument("--config", help="replogle_raw.yaml; fills the paths below")
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides for --config")
    p.add_argument("--version", help="version to check (default: config primary_version)")
    p.add_argument("--train")
    p.add_argument("--test")
    p.add_argument("--state-ref")
    p.add_argument("--universe-train", help="unfiltered train h5ad: decides the cell mapping")
    p.add_argument("--universe-test")
    p.add_argument("--out-dir")
    p.add_argument(
        "--allow-key-mapping",
        action="store_true",
        help="accept a line whose STATE ids do not resolve to raw ids, mapped by (gem_group, UMI_count, gene)",
    )
    p.add_argument("--block", type=int, default=20_000)
    p.set_defaults(func=run_identity)

    p = sub.add_parser("truth", help="A5: two ground_truth.py dirs are identical")
    p.add_argument("--ours", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=float, default=1e-9)
    p.set_defaults(func=run_truth)

    p = sub.add_parser("compare", help="Tier B report: filtered vs all truth")
    p.add_argument("--filtered", required=True)
    p.add_argument("--all", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-degs", type=int, default=5)
    p.set_defaults(func=run_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
