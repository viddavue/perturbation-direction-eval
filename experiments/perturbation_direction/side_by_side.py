"""Side-by-side main table of the raw Replogle rebuild: every method on the ``filtered`` and the ``all`` version.

run_replogle_raw_eval.sh evaluates each version end to end (its own ground truth, baselines refitted on its own
train file, evaluate.py on its own test truth). This script puts those evaluate.py output dirs next to each
other, for both of evaluate.py's reports (full effect, perturbation-specific component).

Conventions (those of the frozen ``results/replogle_test312_w2ema_3seeds_*`` tables; statistics from stats.py)
------------------------------------------------------------------------------------------------------------
* **Shared panel.** Per report, the perturbations with >= ``--min-degs`` scored DEGs (the target gene excluded,
  as evaluate.py --exclude-target counts them) in EVERY version. Medians, CIs and paired tests are computed on
  it, so the versions are compared on the same perturbations. Each version's own panel (what freeze_results.py
  would report for that eval dir alone) is listed beside it for transparency.
* **Median** balanced accuracy over perturbations, **95% percentile bootstrap CI** over perturbations.
* **Paired** two-sided Wilcoxon signed-rank test of the reference (scLDM seed 0) against every other row, with
  the bootstrap CI of the paired median difference ``reference - method`` (> 0: the reference is better), Holm
  within each (report, version) column. Raw and Holm p are both reported.
* The split-half ceiling counts its DEGs on half the cells. On the shared panel it is restricted to the panel
  and, as in freeze_results.py, also needs >= min-degs of its own DEGs, so its n is smaller; its own-panel row
  applies only the latter (freeze_results.py's number exactly).
* n can be below the panel size for any method: balanced accuracy is undefined without both up and down DEGs.
* A method missing from a version or a report (``generic`` never has a perturbation-specific part: evaluate.py
  drops pure shared-response predictions there) is reported as missing, never dropped silently.

Subcommands
-----------
``table``           the main table from one evaluate.py dir per version (``--eval filtered=<dir> --eval all=<dir>``)
``check-selection`` the val sweep rescored by select_sweep.py must choose the frozen config (exit 1 otherwise)
``config``          print replogle_raw_eval.yaml, resolved, as shell assignments (read by run_replogle_raw_eval.sh)

Outputs of ``table`` in --out-dir (``<tag>`` = --tag):
  <tag>_side_by_side.csv, <tag>_side_by_side_specific.csv  wide tables, one row per method, ``<version>_<stat>``
  <tag>_side_by_side_long.csv    the same numbers in long form (report, version, method)
  <tag>_side_by_side_panel.csv   per report and perturbation: n_deg per version, own-panel and shared membership
  <tag>_side_by_side_vs_frozen.csv  (with --frozen-summary) primary own panel vs a frozen summary table
  <tag>_side_by_side.md          the markdown table
  <tag>.provenance.json          git commit + dirty flag, sha256 of every input, settings, timestamp
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stats
from freeze_results import read_per_perturbation, sha256

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSIONS = ("filtered", "all")  # prepare_replogle_raw.py's versions
REPORTS = {"full": "", "specific": "_specific"}  # report -> evaluate.py file suffix
REPORT_TITLES = {"full": "Full effect", "specific": "Perturbation-specific component (shared response projected out)"}
BASELINE_ORDER = ("knn", "mlp", "generic", "transfer", "prior")
KIND_ORDER = ("pred", "baseline", "shuffled", "ceiling", "other")
STATS = ("n", "median", "ci_low", "ci_high", "n_paired", "diff", "diff_ci_low", "diff_ci_high", "p_raw", "p_holm")
OWN = ("n_own", "median_own", "ci_low_own", "ci_high_own")
CODE = (  # scripts of the evaluation stage, hashed into the provenance (matters when the tree is dirty)
    "ground_truth.py",
    "subset_truth.py",
    "verify_merge.py",
    "select_sweep.py",
    "baselines.py",
    "transfer_models.py",
    "learned_baselines.py",
    "direction_models.py",
    "metrics.py",
    "evaluate.py",
    "stats.py",
    "freeze_results.py",
    "side_by_side.py",
    "run_replogle_raw_eval.sh",
)
NAME = re.compile(r"^[A-Za-z0-9_.+-]+$")


def file_stem(name: str) -> str:
    """evaluate.py's per-perturbation file name for a method name (its stem is the method key here)."""
    return name.replace(" ", "_").replace("/", "_")


def label(stem: str) -> str:
    """Readable method name: 'scLDM_s0_(shuffled)' -> 'scLDM_s0 (shuffled)'."""
    return stem.replace("noise_ceiling", "noise ceiling").replace("_(", " (")


def is_ceiling(stem: str) -> bool:
    return stem.startswith("noise_ceiling")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read_eval(eval_dir: str | pathlib.Path) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict]]:
    """Per-perturbation tables of both reports ({report: {method: frame}}) and the methods.json entries."""
    eval_dir = pathlib.Path(eval_dir)
    frames = {report: read_per_perturbation(eval_dir, suffix) for report, suffix in REPORTS.items()}
    manifest = eval_dir / "methods.json"
    methods = json.loads(manifest.read_text())["methods"] if manifest.exists() else []
    return frames, methods


EVAL_SETTINGS = ("exclude_target", "eps", "min_degs", "control")


def check_eval_settings(eval_dirs: dict[str, str | pathlib.Path]) -> dict[str, dict]:
    """The evaluate.py settings that define n_deg (and so the shared panel) must agree across versions.

    Reads each eval dir's methods.json; exits if a setting differs between versions, if a version was not
    scored with ``--exclude-target``, or if a methods.json is missing. Returns {version: settings}.
    """
    settings = {}
    for v, d in eval_dirs.items():
        manifest = pathlib.Path(d) / "methods.json"
        if not manifest.exists():
            raise SystemExit(f"{manifest} missing: cannot confirm the evaluate.py settings of version {v!r}")
        m = json.loads(manifest.read_text())
        settings[v] = {k: m.get(k) for k in EVAL_SETTINGS}
    first = next(iter(settings.values()))
    differ = sorted(k for k in EVAL_SETTINGS if any(s[k] != first[k] for s in settings.values()))
    if differ:
        raise SystemExit(f"evaluate.py settings differ between versions for {differ}: {settings}")
    if first["exclude_target"] is not True:
        raise SystemExit(f"versions were not scored with --exclude-target: {settings}")
    return settings


def method_order(evals: dict[str, dict[str, dict[str, pd.DataFrame]]], manifests: dict[str, list[dict]]):
    """Every method seen in any version/report, ordered: preds, baselines, shuffled controls, ceiling, others.

    Returns (ordered method stems, {stem: kind}).
    """
    kinds, seen = {}, []
    for methods in manifests.values():
        for m in methods:
            stem = file_stem(m["name"])
            kinds.setdefault(stem, m.get("kind", "other"))
            seen.append(stem)
    for per_report in evals.values():
        for frames in per_report.values():
            for stem in sorted(frames):
                kinds.setdefault(stem, "ceiling" if is_ceiling(stem) else "other")
                seen.append(stem)
    first = {s: i for i, s in reversed(list(enumerate(seen)))}

    def key(stem):
        kind = kinds[stem] if kinds[stem] in KIND_ORDER else "other"
        sub = BASELINE_ORDER.index(stem) if kind == "baseline" and stem in BASELINE_ORDER else len(BASELINE_ORDER)
        return KIND_ORDER.index(kind), sub, first[stem]

    return sorted(kinds, key=key), kinds


# ---------------------------------------------------------------------------
# panels and statistics
# ---------------------------------------------------------------------------
def scored_degs(frames: dict[str, pd.DataFrame]) -> tuple[pd.Series, list[str]]:
    """Scored DEGs per perturbation, from the non-ceiling methods' ``n_deg`` (a property of the truth).

    Every non-ceiling method of one evaluate.py report scores the same DEGs, so their ``n_deg`` agree; the
    perturbations where they do not are returned (the maximum is used). The ceiling is left out: its DEGs come
    from half the cells.
    """
    cols = {m: f["n_deg"] for m, f in frames.items() if not is_ceiling(m)}
    if not cols:
        raise ValueError("no non-ceiling method to read the scored DEGs from")
    table = pd.DataFrame(cols)
    inconsistent = sorted(table.index[table.nunique(axis=1, dropna=True) > 1])
    return table.max(axis=1), inconsistent


def column_stats(
    frames: dict[str, pd.DataFrame],
    panel: set[str],
    *,
    reference: str,
    metric: str,
    min_degs: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """One (report, version) column: shared-panel median/CI/paired test and own-panel median/CI per method."""
    notes = []
    shared = {m: f.loc[f.index.intersection(sorted(panel))] for m, f in frames.items()}
    on_panel = stats.summarize(shared, metric=metric, min_degs=min_degs, n_boot=n_boot, seed=seed)
    own = stats.summarize(frames, metric=metric, min_degs=min_degs, n_boot=n_boot, seed=seed)
    out = pd.DataFrame(index=list(frames), columns=[*STATS, *OWN], dtype=float)
    out.loc[on_panel.index, ["n", "median", "ci_low", "ci_high"]] = on_panel[
        ["n_perturbations", "median", "ci_low", "ci_high"]
    ].to_numpy(float)
    out.loc[own.index, list(OWN)] = own[["n_perturbations", "median", "ci_low", "ci_high"]].to_numpy(float)

    scored = {m: f for m, f in shared.items() if len(stats.usable(f, metric, min_degs))}
    for m in sorted(set(frames) - set(scored)):
        notes.append(f"{m}: no perturbation of the shared panel scored")
    if reference not in frames:
        notes.append(f"reference {reference} missing: no paired tests")
    elif reference not in scored:
        notes.append(f"reference {reference} scores no perturbation of the shared panel: no paired tests")
    elif len(scored) > 1:
        tests = stats.pairwise_tests(
            scored, metric=metric, min_degs=min_degs, reference=reference, n_boot=n_boot, seed=seed
        ).set_index("right")
        ok = tests["p_raw"].notna()
        if not ok.all():  # a comparison without common perturbations has no p: keep it out of the Holm family
            tests.loc[ok, "p_holm"] = stats.holm(tests.loc[ok, "p_raw"])
            tests.loc[~ok, "p_holm"] = np.nan
        cols = {"n": "n_paired", "median_difference": "diff", "ci_low": "diff_ci_low", "ci_high": "diff_ci_high"}
        tests = tests.rename(columns=cols)
        cols = ["n_paired", "diff", "diff_ci_low", "diff_ci_high", "p_raw", "p_holm"]
        out.loc[tests.index, cols] = tests[cols].to_numpy(float)
    return out, notes


def side_by_side(
    evals: dict[str, dict[str, dict[str, pd.DataFrame]]],
    manifests: dict[str, list[dict]],
    *,
    reference: str,
    metric: str = "balanced_acc",
    min_degs: int = 5,
    n_boot: int = stats.DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict:
    """All numbers of the main table; ``evals`` is {version: {report: {method: per-perturbation frame}}}."""
    versions = list(evals)
    order, kinds = method_order(evals, manifests)
    long_rows, panel_rows, panels, notes = [], [], {}, {}
    for report in REPORTS:
        n_deg, own_panel, inconsistent = {}, {}, {}
        for v in versions:
            n_deg[v], inconsistent[v] = scored_degs(evals[v][report])
            own_panel[v] = set(n_deg[v].index[n_deg[v] >= min_degs])
        shared = set.intersection(*own_panel.values())
        allp = sorted(set().union(*(set(s.index) for s in n_deg.values())))
        for p in allp:
            row = {"report": report, "perturbation": p}
            for v in versions:
                row[f"n_deg_{v}"] = n_deg[v].get(p, np.nan)
                row[f"panel_{v}"] = p in own_panel[v]
            row["shared"] = p in shared
            panel_rows.append(row)
        panels[report] = {
            "shared": len(shared),
            "own": {v: len(own_panel[v]) for v in versions},
            "only_in": {v: sorted(own_panel[v] - shared) for v in versions},
            "absent_from": {v: sorted(set(allp) - set(n_deg[v].index)) for v in versions},
            "n_deg_inconsistent": {v: inconsistent[v] for v in versions if inconsistent[v]},
        }
        notes[report] = {}
        for v in versions:
            col, col_notes = column_stats(
                evals[v][report],
                shared,
                reference=reference,
                metric=metric,
                min_degs=min_degs,
                n_boot=n_boot,
                seed=seed,
            )
            missing = [m for m in order if m not in evals[v][report]]
            notes[report][v] = {"missing_methods": missing, "notes": col_notes}
            for m in order:
                row = {"report": report, "version": v, "method": m, "label": label(m), "kind": kinds[m]}
                row["status"] = "ok" if m in col.index else "missing"
                row["reference"] = m == reference
                for c in (*STATS, *OWN):
                    row[c] = col.loc[m, c] if m in col.index else np.nan
                long_rows.append(row)
    long = pd.DataFrame(long_rows)
    return {
        "long": long,
        "panel": pd.DataFrame(panel_rows),
        "panels": panels,
        "notes": notes,
        "order": order,
        "kinds": kinds,
        "versions": versions,
        "reference": reference,
    }


def wide(long: pd.DataFrame, report: str, versions: list[str]) -> pd.DataFrame:
    """One row per method (in table order), columns ``<version>_<stat>`` for every version."""
    sub = long[long.report == report]
    order = list(dict.fromkeys(sub.method))
    parts = []
    for v in versions:
        part = sub[sub.version == v].set_index("method").reindex(order)[["status", *STATS, *OWN]]
        parts.append(part.add_prefix(f"{v}_"))
    out = pd.concat(parts, axis=1)
    out.insert(0, "label", [label(m) for m in order])
    out.index.name = "method"
    return out


def compare_frozen(long: pd.DataFrame, frozen: dict[str, str], version: str) -> pd.DataFrame:
    """The version's own-panel numbers next to a frozen summary table (freeze_results.py output), per report."""
    rows = []
    for report, path in frozen.items():
        ref = pd.read_csv(path, index_col=0)
        sub = long[(long.report == report) & (long.version == version) & (long.status == "ok")].set_index("method")
        for m in [*sub.index, *sorted(set(ref.index) - set(sub.index))]:
            f_ok, o_ok = m in ref.index, m in sub.index
            row = {"report": report, "method": m, "in_frozen": f_ok, "in_ours": o_ok}
            row["frozen_n"] = ref.loc[m, "n_perturbations"] if f_ok else np.nan
            row["frozen_median"] = ref.loc[m, "median"] if f_ok else np.nan
            row["own_n"] = sub.loc[m, "n_own"] if o_ok else np.nan
            row["own_median"] = sub.loc[m, "median_own"] if o_ok else np.nan
            row["median_diff"] = row["own_median"] - row["frozen_median"]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _num(x, digits: int = 3, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if signed and round(x, digits) == 0:
        x = 0.0  # no "-0.000"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _int(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else str(int(x))


def _p(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2g}"


def _ci(lo, hi, signed: bool = False) -> str:
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lo, hi)):
        return ""
    return f" [{_num(lo, signed=signed)}, {_num(hi, signed=signed)}]"


def markdown(result: dict, *, primary: str, title: str, min_degs: int, n_boot: int, seed: int, frozen=None) -> str:
    versions, reference = result["versions"], result["reference"]
    head = {v: f"{v} (primary)" if v == primary else v for v in versions}
    lines = [
        f"# {title}",
        "",
        f"Median balanced sign accuracy on the **shared panel**: perturbations with >= {min_degs} scored DEGs (target "
        f"gene excluded) in every version ({', '.join(versions)}), per report. 95% percentile bootstrap CI over "
        f"perturbations ({n_boot} resamples, seed {seed}). Δ = paired median difference {label(reference)} − method "
        f"(> 0: {label(reference)} better) with its bootstrap CI; p = two-sided Wilcoxon signed-rank, raw / Holm "
        "within each column (report × version). n = perturbations entering the median: at most the panel size, "
        "fewer where the metric is undefined (balanced accuracy needs up and down DEGs) and for the split-half "
        f"ceiling, which also needs >= {min_degs} DEGs in its half. `—` = not scored there.",
        "",
    ]
    for report in REPORTS:
        tab = wide(result["long"], report, versions)
        pan = result["panels"][report]
        own = ", ".join(f"{v} own {pan['own'][v]}" for v in versions)
        lines += [f"## {REPORT_TITLES[report]}", "", f"Shared panel: **{pan['shared']}** perturbations ({own}).", ""]
        hdr = ["method"]
        for v in versions:
            hdr += [f"{head[v]} n", f"{head[v]} median [95% CI]", "Δ [95% CI]", "p raw / Holm"]
        lines += ["| " + " | ".join(hdr) + " |", "|---" + "|---:|---|---|---" * len(versions) + "|"]
        for m, r in tab.iterrows():
            name = f"**{r['label']}** (reference)" if m == reference else r["label"]
            cells = [name]
            for v in versions:
                if r[f"{v}_status"] != "ok":
                    cells += ["—", "—", "", ""]
                    continue
                med = _num(r[f"{v}_median"]) + _ci(r[f"{v}_ci_low"], r[f"{v}_ci_high"])
                if m == reference:
                    diff, p = "", ""
                else:
                    diff = _num(r[f"{v}_diff"], signed=True) + _ci(
                        r[f"{v}_diff_ci_low"], r[f"{v}_diff_ci_high"], signed=True
                    )
                    p = f"{_p(r[f'{v}_p_raw'])} / {_p(r[f'{v}_p_holm'])}"
                cells += [_int(r[f"{v}_n"]), med, diff, p]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            f"Own panels: each version's own >= {min_degs}-DEG perturbations, as freeze_results.py would report "
            "that eval dir alone.",
            "",
        ]
        hdr = ["method"] + [x for v in versions for x in (f"{head[v]} n", f"{head[v]} median [95% CI]")]
        lines += ["| " + " | ".join(hdr) + " |", "|---" + "|---:|---" * len(versions) + "|"]
        for _, r in tab.iterrows():
            cells = [r["label"]]
            for v in versions:
                ok = r[f"{v}_status"] == "ok"
                cells += [
                    _int(r[f"{v}_n_own"]) if ok else "—",
                    (_num(r[f"{v}_median_own"]) + _ci(r[f"{v}_ci_low_own"], r[f"{v}_ci_high_own"])) if ok else "—",
                ]
            lines.append("| " + " | ".join(cells) + " |")
        extra = []
        for v in versions:
            note = result["notes"][report][v]
            if note["missing_methods"]:
                extra.append(f"- not scored in {v}: {', '.join(label(m) for m in note['missing_methods'])}")
            extra += [f"- {v}: {n}" for n in note["notes"]]
            if pan["only_in"][v]:
                extra.append(f"- in the {v} panel only ({len(pan['only_in'][v])}): {', '.join(pan['only_in'][v][:20])}")
        if extra:
            lines += ["", *extra]
        lines.append("")
    if frozen is not None and len(frozen):
        both = frozen[frozen.in_frozen & frozen.in_ours]
        lines += [
            f"## {primary} own panel vs the frozen STATE-based tables",
            "",
            "Same perturbations and truth when A5 passed; a non-zero Δ is a method whose inputs changed "
            "(e.g. the MLP, now seeded).",
            "",
            "| report | method | frozen n | frozen median | own n | own median | Δ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in both.iterrows():
            lines.append(
                f"| {r.report} | {label(r.method)} | {_int(r.frozen_n)} | {_num(r.frozen_median)} | {_int(r.own_n)} | "
                f"{_num(r.own_median)} | {_num(r.median_diff, 4, signed=True)} |"
            )
        only = frozen[frozen.in_frozen != frozen.in_ours]
        if len(only):
            pairs = zip(only.method, only.report, only.in_frozen, strict=True)
            listed = ", ".join(f"{label(m)} ({r}, {'frozen' if f else 'ours'} only)" for m, r, f in pairs)
            lines += ["", f"In one of them only: {listed}"]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def load_eval_config(path: str | pathlib.Path, overrides: list[str] | None = None):
    """replogle_raw_eval.yaml with ``key=value`` overrides (OmegaConf dotlist, as prepare_replogle_raw.py's)."""
    from omegaconf import OmegaConf

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


def git_state(repo: pathlib.Path = REPO, exclude: pathlib.Path | None = None) -> dict:
    """HEAD commit and whether the working tree differs from it (tracked changes or untracked files).

    ``exclude``: a directory whose changes are ignored -- the tables' own output dir, which the run itself
    rewrites (a regenerated table must not make its own provenance dirty). Ignored paths are counted apart.
    """

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        top = pathlib.Path(git("rev-parse", "--show-toplevel").strip()).resolve()
        changed = [line for line in git("status", "--porcelain", "--untracked-files=all").splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "changed_paths": None}
    ignored = []
    if exclude is not None:
        try:
            prefix = pathlib.Path(exclude).resolve().relative_to(top).as_posix().rstrip("/") + "/"
        except ValueError:  # output dir outside the repository: nothing of it can show up in git status
            prefix = None
        if prefix:
            # porcelain v1: 'XY path' or 'XY old -> new'; the (last) path is what matters
            ignored = [ln for ln in changed if ln[3:].split(" -> ")[-1].strip('"').startswith(prefix)]
            changed = [ln for ln in changed if ln not in ignored]
    state = {"commit": commit, "dirty": bool(changed), "changed_paths": len(changed)}
    if exclude is not None:
        state["ignored_output_paths"] = len(ignored)
    return state


def _hash(path: str | pathlib.Path) -> str | None:
    path = pathlib.Path(path)
    return sha256(path) if path.is_file() else None


def eval_inputs(eval_dir: pathlib.Path, methods_json: dict | None) -> dict[str, str | None]:
    """sha256 of an eval dir's tables and of every file its methods.json names (truth dir files, preds, baselines)."""
    out = {}
    for path in sorted(eval_dir.glob("summary*.csv")) + sorted(eval_dir.glob("per_perturbation*/*.csv")):
        out[str(path)] = _hash(path)
    if (eval_dir / "methods.json").exists():
        out[str(eval_dir / "methods.json")] = _hash(eval_dir / "methods.json")
    if not methods_json:
        return out

    def resolve(entry):
        abs_path = pathlib.Path(entry.get("abs_path", entry["path"]))
        return entry["path"], abs_path if abs_path.exists() else pathlib.Path(entry["path"])

    truth = methods_json.get("truth_dir")
    if truth:
        shown, real = resolve(truth)
        for f in sorted(p for p in real.glob("*") if p.is_file()):
            out[str(pathlib.Path(shown) / f.name)] = _hash(f)
    entries = [methods_json["specific_ref"]] if methods_json.get("specific_ref") else []
    entries += [m for m in methods_json.get("methods", []) if "path" in m]
    for entry in entries:
        shown, real = resolve(entry)
        out[shown] = _hash(real)
    return out


def provenance(
    result: dict,
    *,
    tag: str,
    timestamp: str | None,
    eval_dirs: dict[str, str],
    primary: str,
    metric: str,
    min_degs: int,
    n_boot: int,
    seed: int,
    extra_inputs: dict[str, str],
    config: str | None,
    overrides: list[str],
    tables: list[str],
    frozen: dict[str, str] | None,
    git: dict | None = None,
) -> dict:
    """The provenance record. ``git``: a git_state() snapshot taken before any table was written."""
    inputs = {}
    for d in eval_dirs.values():
        d = pathlib.Path(d)
        manifest = json.loads((d / "methods.json").read_text()) if (d / "methods.json").exists() else None
        inputs.update(eval_inputs(d, manifest))
    for path in (frozen or {}).values():
        inputs[str(path)] = _hash(path)
    cfg = None
    if config:
        from omegaconf import OmegaConf

        cfg = {
            "path": config,
            "sha256": _hash(config),
            "overrides": overrides,
            "resolved": OmegaConf.to_container(load_eval_config(config, overrides), resolve=True),
        }
    missing = {r: {v: n["missing_methods"] for v, n in per.items()} for r, per in result["notes"].items()}
    return {
        "tag": tag,
        "timestamp": timestamp,
        "git": git if git is not None else git_state(),
        "code": {name: _hash(HERE / name) for name in CODE},
        "config": cfg,
        "eval_dirs": eval_dirs,
        "versions": result["versions"],
        "primary": primary,
        "reference": result["reference"],
        "metric": metric,
        "min_degs": min_degs,
        "panel": "per report: perturbations with >= min_degs scored DEGs (target excluded) in every version",
        "bootstrap": {"n_boot": n_boot, "seed": seed, "unit": "perturbations", "interval": "percentile, 95%"},
        "test": "two-sided Wilcoxon signed-rank on paired perturbations; Holm within each (report, version) column",
        "difference": "median(reference - method) over the paired perturbations",
        "panels": result["panels"],
        "missing_methods": missing,
        "notes": {r: {v: n["notes"] for v, n in per.items()} for r, per in result["notes"].items()},
        "inputs": inputs,
        "extra_inputs": {name: {"path": path, "sha256": _hash(path)} for name, path in extra_inputs.items()},
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def _named(items: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what} {item!r} is not NAME=PATH")
        name, path = item.split("=", 1)
        if name in out:
            raise SystemExit(f"{what} {name!r} given twice")
        out[name] = path
    return out


def run_table(args) -> int:
    eval_dirs = _named(args.eval, "--eval")
    if args.primary not in eval_dirs:
        raise SystemExit(f"--primary {args.primary!r} is not one of the --eval versions {list(eval_dirs)}")
    eval_dirs = {args.primary: eval_dirs[args.primary], **{v: d for v, d in eval_dirs.items() if v != args.primary}}
    check_eval_settings(eval_dirs)
    evals, manifests = {}, {}
    for v, d in eval_dirs.items():
        evals[v], manifests[v] = read_eval(d)
    result = side_by_side(
        evals,
        manifests,
        reference=args.reference,
        metric=args.metric,
        min_degs=args.min_degs,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    out = pathlib.Path(args.out_dir)
    # git state BEFORE anything is written, ignoring the output dir itself (rerunning a table rewrites it)
    git = git_state(exclude=out)
    out.mkdir(parents=True, exist_ok=True)
    tag, written = args.tag, []

    def write_csv(frame: pd.DataFrame, name: str, index: bool = True) -> None:
        frame.to_csv(out / name, index=index)
        written.append(name)

    for report, suffix in REPORTS.items():
        write_csv(wide(result["long"], report, result["versions"]), f"{tag}_side_by_side{suffix}.csv")
    write_csv(result["long"], f"{tag}_side_by_side_long.csv", index=False)
    write_csv(result["panel"], f"{tag}_side_by_side_panel.csv", index=False)
    frozen_paths = _named(args.frozen_summary, "--frozen-summary")
    unknown = set(frozen_paths) - set(REPORTS)
    if unknown:
        raise SystemExit(f"--frozen-summary reports must be among {list(REPORTS)}, got {sorted(unknown)}")
    frozen = compare_frozen(result["long"], frozen_paths, args.primary) if frozen_paths else None
    if frozen is not None:
        write_csv(frozen, f"{tag}_side_by_side_vs_frozen.csv", index=False)
    md = markdown(
        result,
        primary=args.primary,
        title=args.title or tag,
        min_degs=args.min_degs,
        n_boot=args.n_boot,
        seed=args.seed,
        frozen=frozen,
    )
    (out / f"{tag}_side_by_side.md").write_text(md)
    written.append(f"{tag}_side_by_side.md")
    prov = provenance(
        result,
        tag=tag,
        timestamp=args.timestamp,
        eval_dirs=eval_dirs,
        primary=args.primary,
        metric=args.metric,
        min_degs=args.min_degs,
        n_boot=args.n_boot,
        seed=args.seed,
        extra_inputs=_named(args.input, "--input"),
        config=args.config,
        overrides=args.override or [],
        tables=[*written, f"{tag}.provenance.json"],
        frozen=frozen_paths,
        git=git,
    )
    (out / f"{tag}.provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(md)
    print(f"wrote {len(written) + 1} files to {out}: " + ", ".join([*written, f"{tag}.provenance.json"]))
    return 0


def check_selection(rescored: dict, expected: str, frozen: dict | None = None, tol: float = 1e-9) -> dict:
    """The rescored sweep must choose ``expected`` (as must the frozen selection); the two tables are compared."""
    out = {"expected": expected, "chosen": rescored["chosen"], "chosen_ok": rescored["chosen"] == expected}
    out["table"] = rescored["table"]
    if frozen is not None:
        out["frozen_chosen"] = frozen["chosen"]
        out["frozen_chosen_ok"] = frozen["chosen"] == expected
        ours = {r["config"]: r for r in rescored["table"]}
        theirs = {r["config"]: r for r in frozen["table"]}
        diffs, worst, same = {}, 0.0, set(ours) == set(theirs)
        for c in sorted(set(ours) | set(theirs)):
            a = ours.get(c, {}).get("balanced_acc_median")
            b = theirs.get(c, {}).get("balanced_acc_median")
            d = abs(a - b) if a is not None and b is not None else None
            n_same = ours.get(c, {}).get("n_scored") == theirs.get(c, {}).get("n_scored")
            diffs[c] = {"rescored": a, "frozen": b, "abs_diff": d, "same_n_scored": n_same}
            if d is None or not np.isfinite(d) or not n_same:
                same = False
            else:
                worst = max(worst, d)
        out["vs_frozen"] = diffs
        out["max_abs_diff"] = worst
        out["table_matches_frozen"] = same and worst <= tol
    out["pass"] = out["chosen_ok"] and out.get("frozen_chosen_ok", True)
    return out


def run_check_selection(args) -> int:
    rescored = json.loads(pathlib.Path(args.rescored).read_text())
    frozen = json.loads(pathlib.Path(args.frozen).read_text()) if args.frozen else None
    report = check_selection(rescored, args.expected, frozen, tol=args.tol)
    report["inputs"] = {"rescored": args.rescored, "frozen": args.frozen}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    table = pd.DataFrame(rescored["table"]).set_index("config")
    print(table.sort_values("balanced_acc_median", ascending=False).to_string())
    print(f"rescored choice: {report['chosen']} (expected {args.expected})")
    if frozen is not None:
        print(
            f"frozen choice: {report['frozen_chosen']}; table matches the frozen one: "
            f"{report['table_matches_frozen']} (max |diff| {report['max_abs_diff']:.3g})"
        )
    print(f"selection check: {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


def shell_config(cfg) -> str:
    """replogle_raw_eval.yaml as ``CFG_<KEY>=<quoted>`` lines plus one tab-separated line per named file."""

    def q(v) -> str:
        if v is None:
            return "''"
        if isinstance(v, bool):
            return "1" if v else "0"
        return shlex.quote(str(v))

    def named(section) -> dict[str, str]:
        items = {} if section is None else dict(section)
        bad = [k for k in items if not NAME.match(str(k))]
        if bad:
            raise ValueError(f"names must match {NAME.pattern}: {bad}")
        return {str(k): str(v) for k, v in items.items()}

    preds, sweep, controls = named(cfg.preds), named(cfg.selection.sweep), named(cfg.get("controls"))
    if cfg.reference not in preds:
        raise ValueError(f"reference {cfg.reference!r} is not one of preds {list(preds)}")
    if cfg.selection.expected not in sweep:
        raise ValueError(f"selection.expected {cfg.selection.expected!r} is not one of the sweep {list(sweep)}")
    clash = set(controls) & set(preds)
    if clash:
        raise ValueError(f"controls reuse pred names {sorted(clash)}")
    versions = [cfg.primary_version, *[v for v in cfg.versions if v != cfg.primary_version]]
    gt, fr, lr, sel, ev, st = cfg.ground_truth, cfg.frozen, cfg.learned, cfg.selection, cfg.evaluate, cfg.stats
    values = {
        "PREPARED_DIR": cfg.prepared_dir,
        "REQUIRE_DATA_DONE": bool(cfg.require_data_done),
        "OUT_DIR": cfg.out_dir,
        "RESULTS_DIR": cfg.results_dir,
        "RESULTS_TAG": cfg.results_tag,
        "VERSIONS": " ".join(versions),
        "PRIMARY": cfg.primary_version,
        "TRAIN_JSON": cfg.train_json,
        "SPLIT_JSON": cfg.split_json,
        "PERT_KEY": gt.pert_key,
        "CONTROL": gt.control,
        "CONTEXT_KEY": gt.context_key,
        "TARGET_CONTEXT": gt.target_context,
        "GT_MIN_CELLS": gt.min_cells,
        "GT_ALPHA": gt.alpha,
        "GT_MIN_LFC": gt.min_lfc,
        "GT_SEED": gt.seed,
        "FROZEN_TRUTH": fr.truth,
        "FROZEN_TRUTH_TEST": fr.truth_test,
        "A5_TOL": fr.a5_tol,
        "FROZEN_SUMMARY": fr.get("summary"),
        "FROZEN_SUMMARY_SPECIFIC": fr.get("summary_specific"),
        "BASE_MIN_CELLS": cfg.baselines.min_cells,
        "L_MIN_CELLS": lr.min_cells,
        "L_ALPHA": lr.alpha,
        "L_MIN_LFC": lr.min_lfc,
        "L_EMB_DIM": lr.embedding_dim,
        "L_EPOCHS": lr.epochs,
        "L_BATCH": lr.batch_size,
        "L_SEED": lr.seed,
        "L_DEVICE": lr.device,
        "SEL_EXPECTED": sel.expected,
        "SEL_FROZEN_JSON": sel.get("frozen_json"),
        "SEL_TIE_TOL": sel.tie_tolerance,
        "REFERENCE": cfg.reference,
        "MIN_DEGS": ev.min_degs,
        "EPS": ev.eps,
        "EVAL_SEED": ev.seed,
        "METRIC": st.metric,
        "N_BOOT": st.n_boot,
        "STATS_SEED": st.seed,
    }
    lines = [f"CFG_{k}={q(v)}" for k, v in values.items()]
    for kind, items in (("SWEEP", sweep), ("PRED", preds), ("CONTROL", controls)):
        for name, path in items.items():
            if "\t" in path or "\n" in path:
                raise ValueError(f"{kind.lower()} {name}: path contains a tab or newline")
            lines.append(f"{kind}\t{name}\t{path}")
    return "\n".join(lines)


def run_config(args) -> int:
    print(shell_config(load_eval_config(args.config, args.overrides)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("table", help="the side-by-side main table from one evaluate.py dir per version")
    p.add_argument("--eval", action="append", required=True, help="VERSION=EVAL_DIR (evaluate.py --out-dir); repeat")
    p.add_argument("--primary", required=True, help="the primary version (first column group)")
    p.add_argument("--reference", default="scLDM_s0", help="method every other row is paired against")
    p.add_argument("--metric", default="balanced_acc")
    p.add_argument("--min-degs", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=stats.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True, help="file prefix, e.g. replogle_raw_test312_w2ema_3seeds")
    p.add_argument("--title", help="markdown title (default: the tag)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--timestamp", help="run time recorded in the provenance (passed in by the caller)")
    p.add_argument("--config", help="replogle_raw_eval.yaml the run used (hashed and stored resolved)")
    p.add_argument("--override", action="append", help="an OmegaConf override the run used; repeat")
    p.add_argument("--input", action="append", help="NAME=PATH of a further input to hash (manifest, gates); repeat")
    p.add_argument("--frozen-summary", action="append", help="REPORT=CSV (full|specific): a frozen summary to compare")
    p.set_defaults(func=run_table)

    p = sub.add_parser("check-selection", help="the rescored sweep must choose the frozen config")
    p.add_argument("--rescored", required=True, help="select_sweep.py --out on the primary version's val truth")
    p.add_argument("--expected", required=True, help="the frozen choice, e.g. w2_ema")
    p.add_argument("--frozen", help="the frozen selection.json (its choice and table are compared)")
    p.add_argument("--tol", type=float, default=1e-9, help="table agreement tolerance (reported, not a gate)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=run_check_selection)

    p = sub.add_parser("config", help="print replogle_raw_eval.yaml as shell assignments")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    p.set_defaults(func=run_config)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
