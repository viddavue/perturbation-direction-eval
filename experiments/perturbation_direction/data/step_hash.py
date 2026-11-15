"""Step hashes for the raw runners (run_replogle_raw_data.sh, run_replogle_raw_eval.sh): is a step still done?

A step's hash is the sha256 of a canonical JSON of its inputs:

  config    sha256 of the resolved config (OmegaConf, overrides applied), keys in
            ``PROVENANCE_ONLY_KEYS`` (``md5_source``) removed: a note on where a value came from
            never reruns a step
  argv      the step's command line, except the values named with ``--mask-argv``: a volatile value such
            as the run timestamp (recorded in a table's provenance) is replaced by ``ARGV_MASK``
  code      sha256 of the step's script(s) and of every pipeline module they import, transitively
            (``import x`` / ``from x import ...`` anywhere in the file, function-local imports included,
            where ``x.py`` sits next to the importing file)
  files     sha256 of small input files the runner names (metadata JSONs, state_reference, .md5 files)
  stats     size and mtime_ns of large input files (the raw screens, for the checksums gate)
  upstream  the verbatim content of the upstream steps' ``.done`` markers

Installed package versions are not part of the hash: rerun with FORCE=1 after changing the environment.

``adopt_prepare`` decides whether a ``prepare.done`` written by the earlier runner (a timestamp, no
hash) may be kept: see its docstring for what it checks and what it cannot.

Usage (the runner calls these)::

    python step_hash.py hash --config-digest <sha256> [--code a.py]... [--file f]... [--stat f]... \
        [--upstream x.done]... [--mask-argv value]... [--info components.json] -- <argv...>  # prints the hash
    python step_hash.py diff old_components.json new_components.json        # names what changed
    python step_hash.py adopt-prepare --config <yaml> [overrides...] [--out report.json]   # exit 0 = adopt
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import sys
from collections.abc import Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))


HERE = pathlib.Path(__file__).resolve().parent
PROVENANCE_ONLY_KEYS = frozenset({"md5_source"})
ARGV_MASK = "<volatile>"  # stands in for a --mask-argv value: recorded provenance that changes every run
LEGACY_NOTE = (
    "A legacy marker records no code hash, so adoption cannot prove which prepare_replogle_raw.py / "
    "audit_raw_screen.py / make_state_reference.py produced the outputs; it checks what manifest.json "
    "recorded (config, knockdown-filter module, raw md5s, output sizes) and verify_identity reruns."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | pathlib.Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while buf := fh.read(chunk):
            h.update(buf)
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def strip_provenance(obj):
    """Drop provenance-only keys at any depth."""
    if isinstance(obj, dict):
        return {k: strip_provenance(v) for k, v in obj.items() if k not in PROVENANCE_ONLY_KEYS}
    if isinstance(obj, list):
        return [strip_provenance(v) for v in obj]
    return obj


def plain_config(cfg) -> dict:
    """The resolved config as plain JSON types, provenance-only keys removed."""
    if not isinstance(cfg, dict):
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(cfg, resolve=True)
    return strip_provenance(json.loads(json.dumps(cfg, default=str)))


def config_digest(cfg) -> str:
    return sha256_bytes(canonical(plain_config(cfg)).encode())


def _imported_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def code_closure(scripts: Iterable[str | pathlib.Path]) -> dict[str, str]:
    """sha256 of each script and of every module next to it that it imports, transitively.

    Keys are paths relative to the first script's directory (absolute for files elsewhere).
    """
    todo = [pathlib.Path(s).resolve() for s in scripts]
    if not todo:
        return {}
    base = todo[0].parent
    seen: dict[pathlib.Path, str] = {}
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        src = p.read_bytes()
        seen[p] = sha256_bytes(src)
        tree = ast.parse(src, filename=str(p))
        for name in _imported_names(tree):
            cand = p.parent / f"{name}.py"
            if cand.is_file():
                todo.append(cand.resolve())
        # Resolve qualified pipeline imports against the package containing this
        # script, rather than the current working directory or installed code.
        package = next((parent for parent in p.parents if parent.name == "perturbation_direction"), None)
        if package:
            prefix = "experiments.perturbation_direction."
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
                    module = node.module.removeprefix(prefix)
                    target = package.joinpath(*module.split("."))
                    candidates = [target.with_suffix(".py")]
                    candidates.extend(target / f"{alias.name}.py" for alias in node.names)
                    todo.extend(c.resolve() for c in candidates if c.is_file())
    return {(str(p.relative_to(base)) if p.is_relative_to(base) else str(p)): h for p, h in sorted(seen.items())}


def _file_entry(p: pathlib.Path) -> str:
    return sha256_file(p) if p.is_file() else "missing"


def _stat_entry(p: pathlib.Path) -> dict | str:
    if not p.exists():
        return "missing"
    st = p.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def step_hash(
    *,
    config_digest: str,
    argv: Iterable[str],
    code: Iterable[str | pathlib.Path] = (),
    files: Iterable[str | pathlib.Path] = (),
    stats: Iterable[str | pathlib.Path] = (),
    upstream: Iterable[str | pathlib.Path] = (),
    mask: Iterable[str] = (),
) -> tuple[str, dict]:
    """(hash, components) of one step; see the module docstring. ``mask``: argv values replaced by ``ARGV_MASK``."""
    masked = frozenset(mask)
    components = {
        "config": config_digest,
        "argv": [ARGV_MASK if a in masked else a for a in argv],
        "code": code_closure(code),
        "files": {str(p): _file_entry(pathlib.Path(p)) for p in files},
        "stats": {str(p): _stat_entry(pathlib.Path(p)) for p in stats},
        "upstream": {
            pathlib.Path(p).name: (pathlib.Path(p).read_text() if pathlib.Path(p).is_file() else "missing")
            for p in upstream
        },
    }
    return sha256_bytes(canonical(components).encode()), components


def diff_components(old: dict, new: dict) -> list[str]:
    """Which components (and which entries of them) differ, e.g. ``['code: knockdown_filter.py']``."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            keys = sorted(x for x in set(a) | set(b) if a.get(x) != b.get(x))
            out.append(f"{k}: {', '.join(keys)}")
        else:
            out.append(k)
    return out


# ---------------------------------------------------------------------------
# legacy prepare.done
# ---------------------------------------------------------------------------
def adopt_prepare(cfg) -> dict:
    """May a legacy ``prepare.done`` (no hash) be kept for this config? Returns a report with ``adoptable``.

    Every check must pass:

    - ``manifest.json`` exists and says ``pass``;
    - its recorded config equals the current resolved config (provenance-only keys ignored);
    - its knockdown-filter module sha256, TRANSFORM, SCOPE and CELL_LOAD_COMMIT equal the current ones;
    - per raw screen: the md5 the manifest recorded (from the .md5 file) equals the md5 that
      ``checksums.json`` recomputed for the current file (same path, size, mtime) and the config md5,
      the size is unchanged, and the file was not modified after the manifest was written;
    - every output h5ad exists with the recorded size and was not modified after the manifest;
      knockdown_report.json and test_target_cells.csv exist;
    - the metadata gene / vocabulary / test-target counts are unchanged.

    It cannot check which code produced the outputs (``LEGACY_NOTE``).
    """
    from experiments.perturbation_direction.data import knockdown_filter as kd
    from experiments.perturbation_direction.data.prepare_replogle_raw import SPLITS, output_path, read_checksums

    out = pathlib.Path(cfg.out_dir)
    mpath = out / "manifest.json"
    checks: dict[str, dict] = {}

    def check(name: str, problems: list[str]) -> None:
        checks[name] = {"pass": not problems, **({"problems": problems} if problems else {})}

    if not mpath.is_file():
        check("manifest", ["manifest.json missing"])
        return {"adoptable": False, "checks": checks, "manifest": str(mpath), "note": LEGACY_NOTE}
    m = json.loads(mpath.read_text())
    m_mtime = mpath.stat().st_mtime_ns  # same file system as the outputs: no clock skew
    check("manifest_pass", [] if m.get("pass") is True else ["manifest pass is not true"])

    cur, old = plain_config(cfg), strip_provenance(m.get("config") or {})
    check("config", [f"differs: {k}" for k in sorted(set(cur) | set(old)) if cur.get(k) != old.get(k)])

    kd_now = {
        "module_sha256": sha256_file(kd.__file__),
        "TRANSFORM": kd.TRANSFORM,
        "SCOPE": kd.SCOPE,
        "CELL_LOAD_COMMIT": kd.CELL_LOAD_COMMIT,
    }
    kd_old = m.get("knockdown_filter") or {}
    check(
        "knockdown_filter",
        [f"{k}: manifest {kd_old.get(k)} != current {v}" for k, v in kd_now.items() if kd_old.get(k) != v],
    )

    computed = read_checksums(out) or {}
    recorded = {r.get("name"): r for r in m.get("inputs") or []}
    for s in cfg.screens:
        p = pathlib.Path(cfg.raw_dir) / s.file
        rec, c, problems = recorded.get(s.name), computed.get(s.name), []
        if rec is None:
            problems.append("not in the manifest inputs")
        if c is None:
            problems.append("no checksums.json entry (the checksums step runs before prepare)")
        if not p.is_file():
            problems.append(f"{p} missing")
        if not problems:
            st = p.stat()
            if not c.get("pass"):
                problems.append("checksums failed for this file")
            if (c.get("path"), c.get("size"), c.get("mtime_ns")) != (str(p), st.st_size, st.st_mtime_ns):
                problems.append("checksums.json entry does not describe the current file")
            md5 = c.get("md5_computed")
            if md5 is None or rec.get("md5") != md5 or (s.get("md5") is not None and s.md5 != md5):
                problems.append(f"md5 manifest {rec.get('md5')} / recomputed {md5} / config {s.get('md5')}")
            if rec.get("path") != str(p) or rec.get("size") != st.st_size:
                problems.append(f"path/size differ from the manifest ({rec.get('path')}, {rec.get('size')})")
            if st.st_mtime_ns > m_mtime:
                problems.append("raw file modified after the manifest was written")
        check(f"input_{s.name}", problems)

    outputs = m.get("outputs") or {}
    for v in cfg.versions:
        for split in SPLITS:
            p, rec, problems = output_path(out, split, v), (outputs.get(v) or {}).get(split), []
            if rec is None:
                problems.append("not in the manifest outputs")
            elif not p.is_file():
                problems.append(f"{p} missing")
            else:
                if p.stat().st_size != rec.get("size") or rec.get("path") != str(p):
                    problems.append(f"size {p.stat().st_size} / path differ from the manifest ({rec.get('size')})")
                if p.stat().st_mtime_ns > m_mtime:
                    problems.append("modified after the manifest was written")
            check(f"output_{v}_{split}", problems)
    check(
        "reports",
        [f"{n} missing" for n in ("knockdown_report.json", "test_target_cells.csv") if not (out / n).is_file()],
    )

    meta = json.loads(pathlib.Path(cfg.train_json).read_text())
    now = {
        "n_genes": len(meta["genes"]),
        "n_vocab": len(meta["labels"][cfg.pert_key]),
        "n_test_targets": len(json.loads(pathlib.Path(cfg.test_json).read_text())["labels"][cfg.pert_key]),
    }
    rec = m.get("metadata") or {}
    check("metadata", [f"{k}: manifest {rec.get(k)} != current {v}" for k, v in now.items() if rec.get(k) != v])
    return {
        "adoptable": all(c["pass"] for c in checks.values()),
        "checks": checks,
        "manifest": str(mpath),
        "manifest_created": m.get("created"),
        "note": LEGACY_NOTE,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hash", help="print a step's hash")
    p.add_argument("--config-digest", required=True)
    p.add_argument("--code", action="append", default=[], help="step script (its pipeline imports follow)")
    p.add_argument("--file", action="append", default=[], help="small input file, hashed by content")
    p.add_argument("--stat", action="append", default=[], help="large input file, hashed by size and mtime")
    p.add_argument("--upstream", action="append", default=[], help="upstream .done marker")
    p.add_argument("--mask-argv", action="append", default=[], help="replace this argv value before hashing")
    p.add_argument("--info", help="write the components here (JSON)")
    p.add_argument("argv", nargs="*", help="the step's command line, after --")

    p = sub.add_parser("diff", help="name the components that differ between two --info files")
    p.add_argument("old")
    p.add_argument("new")

    p = sub.add_parser("adopt-prepare", help="exit 0 if a legacy prepare.done may be kept")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    p.add_argument("--out", help="write the report here (JSON)")

    args = ap.parse_args(argv)
    if args.cmd == "hash":
        h, comp = step_hash(
            config_digest=args.config_digest,
            argv=args.argv,
            code=args.code,
            files=args.file,
            stats=args.stat,
            upstream=args.upstream,
            mask=args.mask_argv,
        )
        if args.info:
            pathlib.Path(args.info).write_text(json.dumps({"hash": h, **comp}, indent=1, sort_keys=True))
        print(h)
        return 0
    if args.cmd == "diff":
        old = json.loads(pathlib.Path(args.old).read_text()) if pathlib.Path(args.old).is_file() else {}
        new = json.loads(pathlib.Path(args.new).read_text())
        old.pop("hash", None)
        new.pop("hash", None)
        print("; ".join(diff_components(old, new)) if old else "no earlier components recorded")
        return 0

    from experiments.perturbation_direction.data.prepare_replogle_raw import load_config

    report = adopt_prepare(load_config(args.config, args.overrides))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    for name, c in report["checks"].items():
        if not c["pass"]:
            print(f"  {name}: {'; '.join(c['problems'])}")
    print(f"adopt prepare: {'yes' if report['adoptable'] else 'no'} ({report['manifest']})")
    return 0 if report["adoptable"] else 1


if __name__ == "__main__":
    sys.exit(main())
