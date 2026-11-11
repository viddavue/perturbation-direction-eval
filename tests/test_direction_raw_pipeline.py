"""The raw Replogle data stage: per-screen audit, harmonisation into two versions, and the STATE gates.

Synthetic raw screens mimic the real files: X holds integer counts over more genes than the model
uses (dense float32 or CSR), var carries ``gene_name`` (with a duplicated non-model name and MT- genes),
obs carries ``gene`` / ``gem_group`` / ``UMI_count``. Knockdown outcomes are designed to be unambiguous
for any transform: a knocked-down cell has 0 counts of its target, any other cell has 10, as the
controls do. Per line (cells kept by the filter in brackets):

  hepg2  controls 50 [50]; T1 36 kd + 4 not [36]; T2 40 kd [40]; T3 40 not [0, stage 1];
         T4 20 kd [0, < 30 cells]; TX 35 kd [35 by the filter, but TX is outside the vocabulary]
  k562   controls 50 [50]; T1 40 kd [40]; T2 38 kd + 2 not [38]; T3 40 kd [40]; T4 40 kd [40]

Test targets: T1, T2 (hepg2).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "perturbation_direction"))

import audit_raw_screen
import make_state_reference
import prepare_replogle_raw
import verify_merge

REPO = Path(__file__).resolve().parents[1]
E = REPO / "experiments" / "perturbation_direction"

MODEL_GENES = [f"M{i}" for i in range(10)] + ["T1", "T2"]  # T1/T2 are targets and model genes
VOCAB = ["T1", "T2", "T3", "T4", "non-targeting"]
LINES = ["hepg2", "k562"]
TEST_TARGETS = ["T1", "T2"]
CTRL = "non-targeting"
DESIGN = {
    "hepg2": {CTRL: (50, 0), "T1": (36, 4), "T2": (40, 0), "T3": (0, 40), "T4": (20, 0), "TX": (35, 0)},
    "k562": {CTRL: (50, 0), "T1": (40, 0), "T2": (38, 2), "T3": (40, 0), "T4": (40, 0)},
}  # label -> (knocked-down cells, not knocked-down cells)
EXPECTED_KEPT = {
    "hepg2": {CTRL: 50, "T1": 36, "T2": 40, "TX": 35},
    "k562": {CTRL: 50, "T1": 40, "T2": 38, "T3": 40, "T4": 40},
}


# ---------------------------------------------------------------------------
# synthetic raw screens
# ---------------------------------------------------------------------------
def raw_genes() -> list[str]:
    """var gene names of a raw screen: model genes shuffled among others, a duplicate, MT- genes."""
    names = MODEL_GENES + ["T3", "T4", "TX", "FILL", "DUP", "DUP", "MT-CO1", "MT-ND1", "X1", "X2"]
    order = np.random.default_rng(42).permutation(len(names))
    return [names[i] for i in order]


def make_counts(line: str, seed: int, design: dict | None = None) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    design = design or DESIGN[line]
    names = raw_genes()
    col = {}
    for i, g in enumerate(names):
        col.setdefault(g, i)
    labels, kd = [], []
    for lab, (n_kd, n_not) in design.items():
        labels += [lab] * (n_kd + n_not)
        kd += [lab != CTRL] * n_kd + [False] * n_not
    labels, kd = np.array(labels, dtype=object), np.array(kd)
    perm = rng.permutation(len(labels))
    labels, kd = labels[perm], kd[perm]
    n = len(labels)
    c = rng.negative_binomial(2, 0.4, size=(n, len(names))).astype(np.float64)
    c[:, col["M0"]] = 1  # one count of 1 per cell (lets prepare_replogle_state recover counts)
    for t in ("T1", "T2", "T3", "T4", "TX"):
        c[:, col[t]] = 10
    for i in np.flatnonzero(kd):
        c[i, col[labels[i]]] = 0
    c[:, col["FILL"]] = 0
    c[:, col["FILL"]] = rng.integers(800, 3000, n) - c.sum(1)
    assert (c >= 0).all()
    obs = pd.DataFrame(
        {
            "gene": pd.Categorical(labels),
            "gem_group": rng.integers(1, 4, n).astype(np.int64),
            "UMI_count": (c.sum(1) + rng.integers(0, 40, n)).astype(np.float32),
            "sgID_AB": pd.Categorical([f"{g}_a|{g}_b" for g in labels]),
            "mitopercent": rng.random(n).astype(np.float32),
        },
        index=[f"{line.upper()}{i:04d}-{g}" for i, g in zip(range(n), rng.integers(1, 4, n), strict=True)],
    )
    obs.index.name = "cell_barcode"
    return c, obs, names


def write_screen(path: Path, c: np.ndarray, obs: pd.DataFrame, names: list[str], layout: str = "dense", md5=True):
    var = pd.DataFrame({"gene_name": pd.Categorical(names)}, index=[f"ENSG{i:05d}" for i in range(len(names))])
    X = c.astype(np.float32) if layout == "dense" else sparse.csr_matrix(c.astype(np.float32))
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)
    if md5:
        path.with_name(path.name + ".md5").write_text(f"{hashlib.md5(path.read_bytes()).hexdigest()}  {path.name}\n")
    return path


def write_metadata(d: Path, genes=MODEL_GENES, vocab=VOCAB) -> tuple[Path, Path]:
    tr, te = d / "train.json", d / "test.json"
    tr.write_text(json.dumps({"genes": genes, "labels": {"cell_line": LINES, "gene": vocab}}))
    te.write_text(json.dumps({"genes": genes, "labels": {"cell_line": ["hepg2"], "gene": TEST_TARGETS}}))
    return tr, te


def write_config(d: Path, screens: list[dict], **extra) -> Path:
    tr, te = write_metadata(d)
    cfg = {
        "raw_dir": str(d / "raw"),
        "out_dir": str(d / "out"),
        "train_json": str(tr),
        "test_json": str(te),
        "state_reference": str(d / "state_ref" / "state_reference.parquet"),
        "pert_key": "gene",
        "control": CTRL,
        "gene_col": "gene_name",
        "context_key": "cell_line",
        "target_context": "hepg2",
        "block": 37,
        "require_md5": True,
        "versions": ["filtered", "all"],
        "primary_version": "filtered",
        "knockdown": {"residual_expression": 0.30, "cell_residual_expression": 0.50, "min_cells": 30},
        "split": {"min_test_cells": 30},
        "audit": {"min_controls": 30},
        "screens": screens,
        **extra,
    }
    p = d / "replogle_raw.yaml"
    OmegaConf.save(OmegaConf.create(cfg), p)
    return p


def build_raw(d: Path, layouts=("dense", "csr"), designs=None, md5=True) -> tuple[Path, dict]:
    (d / "raw").mkdir(parents=True, exist_ok=True)
    screens, data = [], {}
    for k, (line, layout) in enumerate(zip(LINES, layouts, strict=True)):
        c, obs, names = make_counts(line, seed=k, design=(designs or {}).get(line))
        fname = f"{line}_raw.h5ad"
        write_screen(d / "raw" / fname, c, obs, names, layout, md5=md5)
        md5sum = (d / "raw" / (fname + ".md5")).read_text().split()[0] if md5 else None
        screens.append(
            {
                "name": f"{line}_screen",
                "cell_line": line,
                "file": fname,
                "url": "synthetic",
                "size": (d / "raw" / fname).stat().st_size,
                "md5": md5sum,
            }
        )
        data[line] = (c, obs, names)
    return write_config(d, screens), data


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    cfg_path, data = build_raw(d)
    manifest = prepare_replogle_raw.main(["--config", str(cfg_path)])
    assert manifest == 0
    return d, cfg_path, data


def read_out(d: Path, split: str, version: str) -> ad.AnnData:
    return ad.read_h5ad(d / "out" / f"replogle_{split}_raw_{version}.h5ad")


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_audit_passes_and_reports(tmp_path, layout):
    c, obs, names = make_counts("hepg2", 0)
    p = write_screen(tmp_path / "s.h5ad", c, obs, names, layout)
    rep = audit_raw_screen.audit(p, cell_line="hepg2", genes=MODEL_GENES, vocab=VOCAB, min_controls=30)
    assert rep["pass"], rep["checks"]
    assert rep["X"]["layout"] == layout and rep["cells"] == len(obs) and rep["genes"] == len(names)
    assert rep["controls"] == 50 and rep["n_targets"] == 5
    assert rep["outside_vocab"] == {"labels": 1, "cells": 35, "examples": ["TX"], "control_in_vocab": True}
    assert rep["duplicate_gene_names"] == {"DUP": 2} and rep["model_genes_affected_by_duplicates"] == []
    assert rep["mito"]["n_mt_genes"] == 2
    assert rep["cells_per_target"]["min"] == 20 and rep["label_categories"]["single_in_var"] == 5
    assert rep["guides"]["cells_disagreeing"] == 0
    lib = c.sum(1)
    assert rep["library_full"]["0.0"] == lib.min() and rep["library_full"]["1.0"] == lib.max()
    assert rep["zero_umi_cells"] == 0


def _audit_rc(tmp_path, c, obs, names, layout="dense", **kw) -> tuple[int, dict]:
    p = write_screen(tmp_path / "bad.h5ad", c, obs, names, layout)
    tr, _ = write_metadata(tmp_path)
    out = tmp_path / "audit.json"
    args = ["--h5ad", str(p), "--cell-line", "hepg2", "--metadata-json", str(tr), "--out", str(out)]
    args += [f"--{k.replace('_', '-')}={v}" for k, v in kw.items()]
    rc = audit_raw_screen.main(args)
    return rc, json.loads(out.read_text())


@pytest.mark.parametrize("layout", ["dense", "csr"])
@pytest.mark.parametrize("corruption", ["non_integer", "negative", "zero_library", "few_controls"])
def test_audit_fails_on_bad_counts_and_controls(tmp_path, layout, corruption):
    c, obs, names = make_counts("hepg2", 0)
    min_controls = 30
    if corruption == "non_integer":
        c[3, 5] += 0.5
    elif corruption == "negative":
        c[3, 5] = -1
    elif corruption == "zero_library":
        c[7] = 0
    else:
        min_controls = 300  # the default; the screen has 50
    rc, rep = _audit_rc(tmp_path, c, obs, names, layout, min_controls=min_controls)
    gate = {
        "non_integer": "integer_counts",
        "negative": "integer_counts",
        "zero_library": "no_zero_library",
        "few_controls": "min_controls",
    }[corruption]
    assert rc == 1 and not rep["pass"] and not rep["checks"][gate]["pass"]
    assert [k for k, v in rep["checks"].items() if not v["pass"]] == [gate]


@pytest.mark.parametrize("corruption", ["missing", "duplicated"])
def test_audit_fails_on_model_gene_problems(tmp_path, corruption):
    c, obs, names = make_counts("hepg2", 0)
    names = list(names)
    if corruption == "missing":
        names[names.index("M3")] = "M3-renamed"
    else:
        names[names.index("X1")] = "M3"
    rc, rep = _audit_rc(tmp_path, c, obs, names, min_controls=30)
    assert rc == 1 and not rep["checks"]["model_genes_once"]["pass"]
    assert rep["checks"]["model_genes_once"]["missing" if corruption == "missing" else "duplicated"] == ["M3"]
    if corruption == "duplicated":
        assert rep["model_genes_affected_by_duplicates"] == ["M3"]


def test_readers_handle_the_legacy_raw_layout(tmp_path):
    """The GEO/figshare files use anndata's 0.1.0 dataframe layout (codes + 'categories' object reference)."""
    p = tmp_path / "legacy.h5ad"
    cats = np.array([b"A", b"non-targeting"], dtype=object)
    with h5py.File(p, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["_index"] = "cell_barcode"
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.1.0"
        cg = obs.create_group("__categories")
        cds = cg.create_dataset("gene", data=cats, dtype=h5py.string_dtype())
        codes = obs.create_dataset("gene", data=np.array([1, 0, 0, 1], dtype=np.int16))
        codes.attrs["categories"] = cds.ref
        obs.create_dataset(
            "cell_barcode", data=np.array([b"c1", b"c2", b"c3", b"c4"], dtype=object), dtype=h5py.string_dtype()
        )
        obs.create_dataset("UMI_count", data=np.arange(4, dtype=np.float32))
    with h5py.File(p, "r") as f:
        assert audit_raw_screen.read_column(f["obs"], "gene").tolist() == ["non-targeting", "A", "A", "non-targeting"]
        assert audit_raw_screen.read_index(f["obs"]).tolist() == ["c1", "c2", "c3", "c4"]
        assert audit_raw_screen.read_column(f["obs"], "UMI_count").tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def _model_counts(c, names):
    pos, _, _ = audit_raw_screen.model_gene_positions(np.array(names), MODEL_GENES)
    return c[:, pos]


def test_prepare_versions_counts_order_and_obs(prepared):
    d, _, data = prepared
    for version in ("filtered", "all"):
        for split in ("train", "test"):
            a = read_out(d, split, version)
            assert list(a.var_names) == MODEL_GENES  # metadata order, not the raw file's
            assert sparse.isspmatrix_csr(a.X) and a.X.dtype == np.int32 and not a.layers
            assert list(a.obs.gene.cat.categories) == VOCAB and list(a.obs.cell_line.cat.categories) == LINES
            assert set(a.obs.columns) >= {
                "cell_id",
                "gene",
                "cell_line",
                "screen",
                "gem_group",
                "UMI_count",
                "library_full",
                "kd_keep",
            }
            assert a.obs_names.is_unique
            for line in a.obs.cell_line.unique():
                c, obs, names = data[line]
                o = a.obs[a.obs.cell_line == line]
                rows = obs.index.get_indexer(o.cell_id)
                assert (rows >= 0).all()
                np.testing.assert_array_equal(a[o.index].X.toarray(), _model_counts(c, names)[rows])
                np.testing.assert_array_equal(o.library_full.to_numpy(), c.sum(1)[rows])
                np.testing.assert_array_equal(o.gene.astype(str).to_numpy(), obs.gene.astype(str).to_numpy()[rows])
                np.testing.assert_array_equal(o.gem_group.to_numpy(), obs.gem_group.to_numpy()[rows])


def test_prepare_filtered_reproduces_the_knockdown_design(prepared):
    d, _, _ = prepared
    obs = pd.concat([read_out(d, s, "filtered").obs for s in ("train", "test")])
    assert obs.kd_keep.all()
    got = {line: g.gene.astype(str).value_counts().to_dict() for line, g in obs.groupby("cell_line", observed=True)}
    want = {line: {k: v for k, v in kept.items() if k in VOCAB} for line, kept in EXPECTED_KEPT.items()}
    assert got == want  # TX passes the filter but is outside the vocabulary
    te = read_out(d, "test", "filtered").obs
    assert set(te.cell_line) == {"hepg2"} and te.gene.astype(str).value_counts().to_dict() == {"T1": 36, "T2": 40}
    tr = read_out(d, "train", "filtered").obs
    assert tr[tr.cell_line == "hepg2"].gene.astype(str).value_counts().to_dict() == {CTRL: 50}


def test_prepare_all_version_keeps_every_vocab_cell_and_records_kd_keep(prepared):
    d, _, data = prepared
    tr, te = read_out(d, "train", "all").obs, read_out(d, "test", "all").obs
    assert te.gene.astype(str).value_counts().to_dict() == {"T1": 40, "T2": 40}
    assert tr[tr.cell_line == "hepg2"].gene.astype(str).value_counts().to_dict() == {CTRL: 50, "T3": 40, "T4": 20}
    assert len(tr[tr.cell_line == "k562"]) == 210
    allobs = pd.concat([tr, te])
    assert allobs.kd_keep.sum() == sum(v for line in EXPECTED_KEPT.values() for k, v in line.items() if k in VOCAB)
    assert allobs[allobs.gene == CTRL].kd_keep.all()  # controls are never filtered ...
    filt = pd.concat([read_out(d, s, "filtered").obs for s in ("train", "test")])
    assert (filt.gene == CTRL).sum() == (allobs.gene == CTRL).sum() == 100  # ... and appear in both versions


def test_prepare_obs_index_is_unique_when_barcodes_repeat_across_screens(tmp_path):
    cfg_path, data = build_raw(tmp_path)
    # give k562 the hepg2 barcodes (raw files from different screens can share them)
    c, obs, names = data["k562"]
    hep_ids = data["hepg2"][1].index
    obs = obs.copy()
    obs.index = list(hep_ids[: len(obs)]) + [f"EXTRA{i}" for i in range(max(0, len(obs) - len(hep_ids)))]
    p = write_screen(tmp_path / "raw" / "k562_raw.h5ad", c, obs, names, "csr")
    cfg = prepare_replogle_raw.load_config(
        str(cfg_path),
        [f"screens.1.size={p.stat().st_size}", f"screens.1.md5={hashlib.md5(p.read_bytes()).hexdigest()}"],
    )
    prepare_replogle_raw.prepare(cfg)
    a = read_out(tmp_path, "train", "all")
    assert a.obs_names.is_unique and a.obs.cell_id.duplicated().any()


def test_prepare_manifest(prepared):
    d, cfg_path, _ = prepared
    m = json.loads((d / "out" / "manifest.json").read_text())
    assert m["pass"]
    for rec in m["inputs"]:
        p = Path(rec["path"])
        assert rec["md5"] == hashlib.md5(p.read_bytes()).hexdigest() == rec["expected_md5"] and rec["md5_ok"]
        assert rec["size"] == p.stat().st_size == rec["expected_size"]
        assert "md5_computed" not in rec  # no checksums.json in this out_dir
    assert set(m["git"]) == {"commit", "dirty"}
    assert m["config"]["knockdown"]["min_cells"] == 30 and m["config"]["versions"] == ["filtered", "all"]
    kd = m["knockdown_filter"]
    assert kd["TRANSFORM"] and kd["SCOPE"] in ("per_screen", "merged") and len(kd["CELL_LOAD_COMMIT"]) == 40
    assert m["counts"]["filtered"]["test"]["hepg2"] == {"cells": 76, "targets": 2, "controls": 0}
    assert m["counts"]["filtered"]["train"]["k562"] == {"cells": 208, "targets": 4, "controls": 50}
    assert m["counts"]["all"]["test"]["total"]["cells"] == 80
    assert m["dropped_labels"]["filtered"]["hepg2_screen"] == {"cells": 35, "labels": 1, "top": {"TX": 35}}
    assert m["dropped_labels"]["all"]["k562_screen"]["cells"] == 0
    assert m["raw_counts"]["hepg2_screen"]["kd_kept_cells"] == 161
    assert all(r["equal_to_knockdown_filter_library"] for r in m["library_check"].values())
    assert m["controls_kept"]["pass"]
    assert m["split_checks"]["filtered"]["rules"]["test_target_min_cells"]["cells_per_test_target"]["min"] == 36
    assert (d / "out" / "knockdown_report.json").exists() and not (d / "out" / "_parts").exists()
    tt = pd.read_csv(d / "out" / "test_target_cells.csv", index_col=0)
    assert tt.to_dict() == {"filtered": {"T1": 36, "T2": 40}, "all": {"T1": 40, "T2": 40}}


def test_prepare_dense_and_csr_inputs_give_identical_outputs(tmp_path, prepared):
    d, _, _ = prepared
    cfg_path, _ = build_raw(tmp_path, layouts=("csr", "dense"))
    prepare_replogle_raw.main(["--config", str(cfg_path)])
    for v in ("filtered", "all"):
        for s in ("train", "test"):
            a, b = read_out(d, s, v), read_out(tmp_path, s, v)
            assert (a.X != b.X).nnz == 0
            pd.testing.assert_frame_equal(a.obs, b.obs)


def test_prepare_fails_when_a_test_target_has_no_donor_line(tmp_path):
    k562 = {k: v for k, v in DESIGN["k562"].items() if k != "T2"}
    cfg_path, _ = build_raw(tmp_path, designs={"k562": k562})
    with pytest.raises(RuntimeError, match="test_target_in_a_donor_line"):
        prepare_replogle_raw.main(["--config", str(cfg_path)])
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert not m["pass"] and m["split_checks"]["filtered"]["rules"]["test_target_in_a_donor_line"]["targets"] == ["T2"]


def test_prepare_fails_when_a_test_target_is_too_small_in_the_filtered_version(tmp_path):
    hep = dict(DESIGN["hepg2"], T1=(29, 1))  # passes stage 1, 29 cells left -> dropped by min_cells
    cfg_path, _ = build_raw(tmp_path, designs={"hepg2": hep})
    with pytest.raises(RuntimeError, match="filtered:test_target_min_cells"):
        prepare_replogle_raw.main(["--config", str(cfg_path)])
    rules = json.loads((tmp_path / "out" / "manifest.json").read_text())["split_checks"]
    assert rules["filtered"]["rules"]["test_target_min_cells"]["targets_below"] == {"T1": 0}
    assert rules["all"]["rules"]["test_target_min_cells"]["pass"]  # 30 cells unfiltered: reported, not gating
    assert not rules["all"]["rules"]["test_target_min_cells"]["required"]


@pytest.mark.parametrize("problem", ["no_md5", "wrong_md5", "wrong_size"])
def test_prepare_refuses_unverified_inputs(tmp_path, problem):
    cfg_path, _ = build_raw(tmp_path, md5=(problem != "no_md5"))
    over = {"no_md5": [], "wrong_md5": ["screens.0.md5=0123"], "wrong_size": ["screens.0.size=1"]}[problem]
    with pytest.raises(RuntimeError, match="raw inputs not ready"):
        prepare_replogle_raw.main(["--config", str(cfg_path), *over])


def test_prepare_honours_a_merged_knockdown_scope(tmp_path, monkeypatch):
    """With SCOPE='merged' the filter sees all lines at once: T3 averages to a 0.5 ratio and is dropped
    everywhere, while hepg2's 20 T4 cells count together with k562's 40 and survive min_cells."""
    kd = prepare_replogle_raw.kd
    calls = []
    real = kd.concat_target_expression
    monkeypatch.setattr(kd, "SCOPE", "merged")
    monkeypatch.setattr(kd, "concat_target_expression", lambda parts: calls.append(len(parts)) or real(parts))
    cfg_path, _ = build_raw(tmp_path)
    prepare_replogle_raw.main(["--config", str(cfg_path)])
    assert calls == [2]
    tr = read_out(tmp_path, "train", "filtered").obs
    assert "T3" not in set(tr.gene.astype(str))
    assert (tr[(tr.cell_line == "hepg2")].gene == "T4").sum() == 20


# ---------------------------------------------------------------------------
# STATE reference + identity gates
# ---------------------------------------------------------------------------
def build_state_from_raw(d: Path, data: dict, kept: pd.DataFrame) -> Path:
    """A STATE-like file: the kept cells, X = log1p(1e4 * c / L) with L over all raw genes, a gene superset."""
    xs, obs = [], []
    state_genes = [g for g in dict.fromkeys(raw_genes()) if g not in ("FILL", "DUP")]
    for line in LINES:
        c, o, names = data[line]
        first = audit_raw_screen.first_positions(np.array(names))
        k = kept[kept.cell_line == line]
        rows = o.index.get_indexer(k.cell_id)
        cc = c[rows]
        xs.append(np.log1p(1e4 * cc[:, [first[g] for g in state_genes]] / cc.sum(1, keepdims=True)).astype(np.float32))
        oo = o.iloc[rows][["gene", "gem_group", "UMI_count"]].copy()
        oo["gene"] = oo["gene"].astype(str)
        oo["cell_line"] = line
        obs.append(oo)
    obs = pd.concat(obs)
    obs["gene"] = pd.Categorical(obs.gene, categories=VOCAB)
    obs["cell_line"] = pd.Categorical(obs.cell_line, categories=LINES)
    p = d / "state.h5ad"
    ad.AnnData(X=np.vstack(xs), obs=obs, var=pd.DataFrame(index=state_genes)).write_h5ad(p)
    return p


@pytest.fixture(scope="module")
def state_ref(prepared):
    from prepare_replogle_state import main as prepare_state

    d, cfg_path, data = prepared
    kept = pd.concat([read_out(d, s, "filtered").obs for s in ("train", "test")])
    sd = d / "state"
    sd.mkdir(exist_ok=True)
    src = build_state_from_raw(sd, data, kept)
    tr, te = d / "train.json", d / "test.json"
    prepare_state(
        ["--src", str(src), "--train-json", str(tr), "--test-json", str(te), "--chunk", "50", "--out-dir", str(sd)]
    )
    out = d / "state_ref"
    make_state_reference.main(
        [
            "--state",
            str(src),
            "--train",
            str(sd / "replogle_train.h5ad"),
            "--test",
            str(sd / "replogle_test.h5ad"),
            "--metadata-json",
            str(tr),
            "--out-dir",
            str(out),
            "--block",
            "64",
        ]
    )
    return d, cfg_path, out / "state_reference.parquet"


def test_state_reference_columns_and_library(state_ref):
    d, _, ref = state_ref
    df = pd.read_parquet(ref)
    assert list(df.columns) == list(make_state_reference.COLUMNS)
    assert len(df) == 76 + 50 + 208 and (df.nz_mismatch == 0).all()
    assert df.lib_implied_rel_range.max() < 1e-5
    info = json.loads((ref.parent / "state_reference.json").read_text())
    assert info["rows"] == len(df) and info["output_md5"] == hashlib.md5(ref.read_bytes()).hexdigest()
    assert info["per_line_split"] == {"hepg2/test": 76, "hepg2/train": 50, "k562/train": 208}


def test_counts_hash_is_int32_little_endian_sha1():
    row = np.array([[0, 3, 1]], dtype=np.int64)
    assert make_state_reference.counts_hash(row) == [hashlib.sha1(np.array([0, 3, 1], "<i4").tobytes()).hexdigest()]
    with pytest.raises(ValueError):
        make_state_reference.counts_hash(np.array([[0.5, 1.0]]))


def test_identity_gates_pass_end_to_end(state_ref):
    d, cfg_path, ref = state_ref
    rc = verify_merge.main(["identity", "--config", str(cfg_path), f"state_reference={ref}"])
    assert rc == 0
    rep = json.loads((d / "out" / "verify" / "identity.json").read_text())
    assert rep["pass"] and {v["method"] for v in rep["mapping"].values()} == {"obs_index"}
    assert all(v["allowed"] for v in rep["mapping"].values()) and rep["allow_key_mapping"] is False
    a1 = rep["A1_cell_set_and_labels"]["per_line"]
    assert all(r["cell_id_mismatch"] == 0 and r["duplicated_ours"] == 0 for r in a1.values())
    assert rep["A3_library"]["max_rel_diff"] < 1e-6 and rep["A3_library"]["max_abs_diff"] < 0.01
    assert rep["A4_split"]["test"] == {"ours": 76, "state": 76, "only_ours": 0, "only_state": 0, "pass": True}


@pytest.fixture
def gate_inputs(state_ref):
    d, _, ref = state_ref
    paths = {s: d / "out" / f"replogle_{s}_raw_filtered.h5ad" for s in ("train", "test")}
    ours = verify_merge.load_ours(paths, block=17)
    universe = verify_merge.load_ours(
        {s: d / "out" / f"replogle_{s}_raw_all.h5ad" for s in ("train", "test")}, with_hash=False
    )
    uni = {line: set(g.cell_id) for line, g in universe.groupby("cell_line")}
    return ours, pd.read_parquet(ref), uni, paths


def _failed(report) -> set[str]:
    return {k for k in ("A1_cell_set_and_labels", "A2_counts_hash", "A3_library", "A4_split") if not report[k]["pass"]}


def test_gates_fail_on_one_changed_count(gate_inputs):
    ours, state, uni, paths = gate_inputs
    a = ad.read_h5ad(paths["train"])
    i = 5
    c = a.X[i].toarray().astype(np.int64)
    c[0, 3] += 1
    state = state.copy()
    state.loc[state.cell_id == a.obs.cell_id.iloc[i], "counts_hash"] = make_state_reference.counts_hash(c)[0]
    rep, table = verify_merge.identity(ours, state, universe=uni)
    assert _failed(rep) == {"A2_counts_hash"} and rep["A2_counts_hash"]["mismatch"] == 1
    assert table.gate.tolist() == ["A2_counts_hash"]


def test_gates_fail_on_a_library_off_by_1e3(gate_inputs):
    ours, state, uni, _ = gate_inputs
    state = state.copy()
    state.loc[state.index[10], "lib_full_state"] *= 1 + 1e-3
    rep, _ = verify_merge.identity(ours, state, universe=uni)
    assert _failed(rep) == {"A3_library"} and rep["A3_library"]["fail"] == 1


def test_gates_fail_on_a_dropped_cell(gate_inputs):
    ours, state, uni, _ = gate_inputs
    gone = state.iloc[3]
    rep, table = verify_merge.identity(ours, state.drop(index=state.index[3]), universe=uni)
    assert _failed(rep) == {"A1_cell_set_and_labels", "A4_split"}
    assert rep["A1_cell_set_and_labels"]["per_line"][gone.cell_line]["only_ours"] == 1
    assert rep["A4_split"]["differences"][0]["reason"] == "only in ours (not in STATE)"
    assert set(table.cell_id) == {gone.cell_id}


def test_gates_fail_on_a_swapped_label(gate_inputs):
    ours, state, uni, _ = gate_inputs
    state = state.copy()
    k = state[state.cell_line == "k562"]
    i = k.index[k.gene == "T3"][0]
    j = k.index[k.gene == "T4"][0]
    state.loc[[i, j], "gene"] = ["T4", "T3"]
    rep, _ = verify_merge.identity(ours, state, universe=uni)
    assert "A1_cell_set_and_labels" in _failed(rep)
    assert rep["A1_cell_set_and_labels"]["per_line"]["k562"]["label_mismatch"] == 2


def test_gates_fail_on_a_split_difference(gate_inputs):
    ours, state, uni, _ = gate_inputs
    state = state.copy()
    state.loc[state.index[state.split == "test"][0], "split"] = "train"
    rep, _ = verify_merge.identity(ours, state, universe=uni)
    assert _failed(rep) == {"A4_split"} and rep["A4_split"]["differences"][0]["reason"] == "split differs"


def test_key_mapping_is_refused_unless_allowed(gate_inputs):
    """The (gem_group, UMI_count, gene) fallback can pair the wrong cells; it needs --allow-key-mapping."""
    ours, state, uni, _ = gate_inputs
    state = state.copy()
    state["cell_id"] = state.cell_id + "-state"
    rep, table = verify_merge.identity(ours, state, universe=uni)
    assert {v["method"] for v in rep["mapping"].values()} == {"key"}
    assert not rep["pass"] and _failed(rep) == {"A1_cell_set_and_labels"}
    assert not any(v["allowed"] for v in rep["mapping"].values())
    assert table.gate.tolist() == ["A1_key_mapping"] * len(LINES)
    rep, table = verify_merge.identity(ours, state, universe=uni, allow_key_mapping=True)
    assert rep["pass"] and table.empty
    assert all(r["cell_id_mismatch"] is None for r in rep["A1_cell_set_and_labels"]["per_line"].values())


def test_identity_cli_needs_allow_key_mapping_for_the_key_fallback(state_ref, tmp_path):
    _, cfg_path, ref = state_ref
    st = pd.read_parquet(ref)
    st["cell_id"] = st.cell_id + "-state"
    mod = tmp_path / "state_reference.parquet"
    st.to_parquet(mod, index=False)
    args = ["identity", "--config", str(cfg_path), "--state-ref", str(mod), "--out-dir", str(tmp_path / "v")]
    assert verify_merge.main(args) == 1
    rep = json.loads((tmp_path / "v" / "identity.json").read_text())
    assert rep["allow_key_mapping"] is False and not rep["mapping"]["hepg2"]["allowed"]
    assert verify_merge.main([*args, "--allow-key-mapping"]) == 0


def test_cell_id_gate_catches_a_wrong_pairing(gate_inputs, monkeypatch):
    """Every barcode-mapped pair must carry the raw id its STATE id resolved to."""
    ours, state, uni, _ = gate_inputs
    real = verify_merge.map_line

    def swapped(o, s, u):
        mp, info = real(o, s, u)
        v = mp.to_numpy().copy()
        v[[0, 1]] = v[[1, 0]]
        return pd.Series(v, index=mp.index), info

    monkeypatch.setattr(verify_merge, "map_line", swapped)
    rep, table = verify_merge.identity(ours, state, universe=uni)
    assert not rep["A1_cell_set_and_labels"]["pass"]
    assert all(r["cell_id_mismatch"] == 2 for r in rep["A1_cell_set_and_labels"]["per_line"].values())
    assert (table.gate == "A1_cell_id").sum() == 2 * len(LINES)


@pytest.mark.parametrize("where", ["ours_across_splits", "ours_within_split", "state"])
def test_duplicated_cell_ids_fail_a1_instead_of_crashing(gate_inputs, where):
    ours, state, uni, _ = gate_inputs
    if where == "state":
        row = state[state.cell_line == "k562"].iloc[[0]]
        state = pd.concat([state, row], ignore_index=True)
        gate, line, key = "A1_duplicated_state", "k562", "duplicated_state"
    else:
        row = ours[(ours.cell_line == "hepg2") & (ours.split == "train")].iloc[[0]].copy()
        if where == "ours_across_splits":
            row["split"] = "test"
        ours = pd.concat([ours, row], ignore_index=True)
        gate, line, key = "A1_duplicated_ours", "hepg2", "duplicated_ours"
    rep, table = verify_merge.identity(ours, state, universe=uni)
    assert not rep["pass"] and "A1_cell_set_and_labels" in _failed(rep)
    assert rep["A1_cell_set_and_labels"]["per_line"][line][key] == 2
    assert rep["mapping"][line]["method"] == "obs_index"  # the first copy is still matched by barcode
    dup = table[table.gate == gate]
    assert len(dup) == 2 and set(dup.cell_id) == {row.cell_id.iloc[0]}
    if where == "ours_across_splits":
        assert "train" in dup.detail.iloc[0] and "test" in dup.detail.iloc[0]


def test_a3_requires_the_exact_integer_library(gate_inputs):
    ours, state, uni, _ = gate_inputs
    ours, state = ours.copy(), state.copy()

    def at(k):  # STATE row of our k-th cell
        return state.index[(state.cell_id == ours.cell_id[k]) & (state.cell_line == ours.cell_line[k])]

    ours.loc[0, "library_full"] = 100_000
    state.loc[at(0), "lib_full_state"] = 100_000.6  # relative 6e-6: passed the former 1e-5 rule
    rep, table = verify_merge.identity(ours, state, universe=uni)
    assert _failed(rep) == {"A3_library"} and rep["A3_library"]["fail"] == 1
    assert table.gate.tolist() == ["A3_library"]
    state.loc[at(0), "lib_full_state"] = 100_000.4
    state.loc[at(1), "lib_full_state"] = ours.library_full[1] + 0.3  # relative > 1e-4: failed the former rule
    rep, _ = verify_merge.identity(ours, state, universe=uni)
    assert rep["pass"] and rep["A3_library"]["max_abs_diff"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# A5 and Tier B on ground-truth dirs
# ---------------------------------------------------------------------------
def write_truth(d: Path, delta: pd.DataFrame, deg: pd.DataFrame, n_cells: dict, padj=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    delta.to_parquet(d / "delta_true.parquet")
    (padj if padj is not None else delta.abs() * 0 + 0.01).to_parquet(d / "padj.parquet")
    deg.to_parquet(d / "deg.parquet")
    (d / "info.json").write_text(json.dumps({"n_cells": n_cells, "control": CTRL}))
    return d


def _toy_truth(rng, perts=("P1", "P2", "P3"), genes=("P1", "P2", "P3", "g1", "g2", "g3", "g4", "g5", "g6")):
    delta = pd.DataFrame(rng.normal(0, 0.3, (len(perts), len(genes))), index=list(perts), columns=list(genes))
    return delta, delta.abs() > 0.2, dict.fromkeys(perts, 40)


@pytest.mark.parametrize("corruption", [None, "delta", "deg", "n_cells", "perturbation"])
def test_truth_identity_gate(tmp_path, corruption):
    rng = np.random.default_rng(0)
    delta, deg, n = _toy_truth(rng)
    ref = write_truth(tmp_path / "ref", delta, deg, n)
    d2, deg2, n2 = delta.copy(), deg.copy(), dict(n)
    if corruption == "delta":
        d2.iloc[1, 4] += 1e-6
    elif corruption == "deg":
        deg2.iloc[2, 3] = not deg2.iloc[2, 3]
    elif corruption == "n_cells":
        n2["P2"] = 39
    elif corruption == "perturbation":
        d2, deg2 = d2.drop(index="P3"), deg2.drop(index="P3")
    ours = write_truth(tmp_path / "ours", d2, deg2, n2)
    rc = verify_merge.main(["truth", "--ours", str(ours), "--reference", str(ref), "--out", str(tmp_path / "a5.json")])
    rep = json.loads((tmp_path / "a5.json").read_text())
    assert rc == (0 if corruption is None else 1) and rep["pass"] == (corruption is None)
    if corruption == "delta":
        assert rep["delta"]["max_abs_diff"] == pytest.approx(1e-6) and rep["padj"]["pass"]


def test_compare_filtered_vs_all(tmp_path):
    rng = np.random.default_rng(1)
    delta, deg, n = _toy_truth(rng, genes=("P1", "P2", "P3", *[f"g{i}" for i in range(40)]))
    deg.iloc[:, 3:10] = True  # >= 5 DEGs everywhere
    all_ = 1.1 * delta
    for p in delta.index:
        all_.loc[p, p] = delta.loc[p, p] + 5.0  # the target gene itself differs a lot; it must not count
    write_truth(tmp_path / "f", delta, deg, n)
    write_truth(tmp_path / "a", all_, deg, {p: 2 * v for p, v in n.items()})
    rc = verify_merge.main(
        ["compare", "--filtered", str(tmp_path / "f"), "--all", str(tmp_path / "a"), "--out-dir", str(tmp_path / "cmp")]
    )
    assert rc == 0
    t = pd.read_csv(tmp_path / "cmp" / "filtered_vs_all.csv", index_col=0)
    np.testing.assert_allclose(t.pearson, 1.0)
    np.testing.assert_allclose(t.slope_all_on_filtered, 1.1)
    np.testing.assert_allclose(t.deg_jaccard, 1.0)
    np.testing.assert_allclose(t.delta_target_all - t.delta_target_filtered, 5.0)
    s = json.loads((tmp_path / "cmp" / "filtered_vs_all.json").read_text())
    panel = {k: s["panel_min_degs"][k] for k in ("min_degs", "filtered", "all", "both", "either")}
    assert panel == {"min_degs": 5, "filtered": 3, "all": 3, "both": 3, "either": 3}
    assert s["n_cells_all"]["0.5"] == 80


def test_compare_panel_counts_degs_without_the_target(tmp_path):
    """The >= min_degs panel is chosen as evaluate.py --exclude-target chooses it: target gene not counted."""
    genes = ["P1", "P2", "P3", *[f"g{i}" for i in range(10)]]
    perts = ["P1", "P2", "P3"]
    delta = pd.DataFrame(np.random.default_rng(2).normal(0, 0.3, (3, len(genes))), index=perts, columns=genes)
    deg = pd.DataFrame(False, index=perts, columns=genes)
    deg.loc["P1", ["P1", "g0", "g1", "g2", "g3"]] = True  # 5 DEGs, 4 without the target
    deg.loc["P2", ["g0", "g1", "g2", "g3", "g4"]] = True  # 5, none of them the target
    deg.loc["P3", ["P3", "g0", "g1", "g2", "g3", "g4"]] = True  # 6, 5 without the target
    for name in ("f", "a"):
        write_truth(tmp_path / name, delta, deg, dict.fromkeys(perts, 40))
    table, s = verify_merge.compare_versions(
        verify_merge.read_truth(tmp_path / "f"), verify_merge.read_truth(tmp_path / "a"), min_degs=5
    )
    assert s["panel_min_degs"]["filtered"] == s["panel_min_degs"]["all"] == s["panel_min_degs"]["both"] == 2
    assert table.loc["P1", "n_deg_filtered"] == 5 and table.loc["P1", "n_deg_filtered_excl_target"] == 4


def test_ground_truth_runs_on_both_versions_and_compares(prepared, tmp_path):
    """ground_truth.py reads the prepared files as they are; filtered vs itself passes A5."""
    import ground_truth

    d, _, _ = prepared
    tr = str(d / "train.json")
    for v in ("filtered", "all"):
        ground_truth.main(
            [
                "--pert-h5ad",
                str(d / "out" / f"replogle_test_raw_{v}.h5ad"),
                "--ctrl-h5ad",
                str(d / "out" / f"replogle_train_raw_{v}.h5ad"),
                "--pert-key",
                "gene",
                "--control",
                CTRL,
                "--context",
                "cell_line=hepg2",
                "--metadata-json",
                tr,
                "--min-cells",
                "15",
                "--out-dir",
                str(tmp_path / v),
            ]
        )
    info = json.loads((tmp_path / "filtered" / "info.json").read_text())
    assert info["n_cells"] == {"T1": 36, "T2": 40} and info["n_control_cells"] == 50
    assert (
        verify_merge.main(
            [
                "truth",
                "--ours",
                str(tmp_path / "filtered"),
                "--reference",
                str(tmp_path / "filtered"),
                "--out",
                str(tmp_path / "a5.json"),
            ]
        )
        == 0
    )
    assert (
        verify_merge.main(
            [
                "truth",
                "--ours",
                str(tmp_path / "all"),
                "--reference",
                str(tmp_path / "filtered"),
                "--out",
                str(tmp_path / "a5_all.json"),
            ]
        )
        == 1
    )
    assert (
        verify_merge.main(
            [
                "compare",
                "--filtered",
                str(tmp_path / "filtered"),
                "--all",
                str(tmp_path / "all"),
                "--out-dir",
                str(tmp_path / "cmp"),
            ]
        )
        == 0
    )
    t = pd.read_csv(tmp_path / "cmp" / "filtered_vs_all.csv", index_col=0)
    assert t.loc["T1", "n_cells_all"] == 40 and t.loc["T1", "n_cells_filtered"] == 36


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _run_runner(cfg_path: Path, *overrides: str, job_root: Path | None = None, **env) -> subprocess.CompletedProcess:
    root = job_root or REPO
    runner = root / "experiments" / "perturbation_direction" / "run_replogle_raw_data.sh"
    env = dict(os.environ, PYTHON=sys.executable, CONFIG=str(cfg_path), JOB_ROOT=str(root), **env)
    return subprocess.run(["bash", str(runner), *overrides], env=env, capture_output=True, text=True, timeout=600)


def _started(r: subprocess.CompletedProcess) -> set[str]:
    return set(re.findall(r"\] start (\S+)", r.stdout))


def test_runner_end_to_end_resumes_and_stops_on_failure(tmp_path, state_ref):
    _, _, ref = state_ref  # same seeds -> the same synthetic cells as this fresh build
    cfg_path, data = build_raw(tmp_path)
    r = _run_runner(cfg_path, f"state_reference={ref}")
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "out"
    assert (out / "DONE_raw_data").exists() and "RAW_DATA_DONE" in r.stdout
    assert {p.name for p in (out / "logs").glob("*.done")} == {
        "checksums.done",
        "audit_hepg2_screen.done",
        "audit_k562_screen.done",
        "prepare.done",
        "verify_identity.done",
    }
    assert json.loads((out / "checksums.json").read_text())["pass"]
    assert all(rec["md5_computed"] == rec["md5"] for rec in json.loads((out / "manifest.json").read_text())["inputs"])
    assert json.loads((out / "verify" / "identity.json").read_text())["pass"]
    assert "skip prepare" in _run_runner(cfg_path, f"state_reference={ref}").stdout

    # a corrupted screen stops the run at its audit, before anything is merged
    c, obs, names = data["k562"]
    c = c.copy()
    c[0, 0] += 0.5
    p = write_screen(tmp_path / "raw" / "k562_raw.h5ad", c, obs, names, "csr")
    out2 = tmp_path / "out2"
    r = _run_runner(
        cfg_path,
        f"state_reference={ref}",
        f"out_dir={out2}",
        f"screens.1.size={p.stat().st_size}",
        f"screens.1.md5={hashlib.md5(p.read_bytes()).hexdigest()}",
    )
    assert r.returncode == 1 and "FAIL  audit_k562_screen" in r.stderr
    assert not (out2 / "DONE_raw_data").exists() and not (out2 / "logs" / "prepare.log").exists()


def _job_copy(tmp_path: Path) -> Path:
    """A throw-away JOB_ROOT holding a copy of the pipeline scripts, so a test can edit them."""
    job = tmp_path / "job"
    dst = job / "experiments" / "perturbation_direction"
    dst.mkdir(parents=True)
    for p in E.iterdir():
        if p.suffix in (".py", ".sh"):
            shutil.copy2(p, dst / p.name)
    return job


def test_runner_reruns_exactly_what_changed(tmp_path, state_ref):
    _, _, ref = state_ref
    cfg_path, _ = build_raw(tmp_path)
    job = _job_copy(tmp_path)
    code = job / "experiments" / "perturbation_direction"
    out, logs = tmp_path / "out", tmp_path / "out" / "logs"
    ov = f"state_reference={ref}"
    gates = {"checksums", "audit_hepg2_screen", "audit_k562_screen"}
    every = gates | {"prepare", "verify_identity"}

    def run(*extra, **env):
        r = _run_runner(cfg_path, ov, *extra, job_root=job, **env)
        return r, _started(r)

    r, started = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert started == every
    assert all((logs / f"{s}.done").read_text().startswith("hash=") for s in every)
    assert all(json.loads((logs / f"{s}.hash.json").read_text())["code"] for s in every)
    # nothing changed but a provenance note in the config: nothing runs
    assert run("screens.0.md5_source=edited")[1] == set()

    # a module that only prepare imports (checksums too: verify_merge imports prepare's load_config)
    with open(code / "knockdown_filter.py", "a") as fh:
        fh.write("\n# edited\n")
    r, started = run()
    assert r.returncode == 0 and started == {"checksums", "prepare", "verify_identity"}, r.stdout
    assert "code" in r.stdout  # the log says which hash component changed

    # a step that runs first drops the markers downstream of it, so a failure cannot leave them valid
    src = (code / "prepare_replogle_raw.py").read_text()
    head = "def main(argv=None) -> int:\n"
    (code / "prepare_replogle_raw.py").write_text(src.replace(head, head + "    raise SystemExit('boom')\n", 1))
    r, started = run()
    assert r.returncode != 0 and "prepare" in started and "verify_identity" not in started
    assert not (logs / "prepare.done").exists() and not (logs / "verify_identity.done").exists()
    assert not (out / "DONE_raw_data").exists()
    (code / "prepare_replogle_raw.py").write_text(src)
    r, started = run()
    assert r.returncode == 0 and started == {"checksums", "prepare", "verify_identity"}, r.stdout

    # markers of the earlier runner (a timestamp, no hash): gates and verify rerun, prepare is adopted
    for s in every:
        (logs / f"{s}.done").write_text("2026-11-10T19:00:16\n")
    h5 = out / "replogle_train_raw_filtered.h5ad"
    before = h5.stat().st_mtime_ns
    r, started = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert started == gates | {"verify_identity"} and "adopt prepare" in r.stdout
    assert h5.stat().st_mtime_ns == before
    marker = (logs / "prepare.done").read_text()
    assert marker.startswith("hash=") and "adopted=" in marker and "2026-11-10T19:00:16" in marker
    assert json.loads((logs / "prepare.adopt.json").read_text())["adoptable"]
    assert run()[1] == set()

    # ... unless its manifest disagrees with the current code: then prepare is rebuilt
    (logs / "prepare.done").write_text("2026-11-10T19:00:16\n")
    m = json.loads((out / "manifest.json").read_text())
    m["knockdown_filter"]["module_sha256"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(m))
    r, started = run()
    assert r.returncode == 0 and started == {"prepare", "verify_identity"} and "not adoptable" in r.stdout
    assert h5.stat().st_mtime_ns != before

    assert run(FORCE="1")[1] == every


def test_checksums_recompute_md5_against_config_and_sidecar(tmp_path):
    cfg_path, _ = build_raw(tmp_path)
    assert verify_merge.main(["checksums", "--config", str(cfg_path), "--jobs", "2"]) == 0
    rep = json.loads((tmp_path / "out" / "checksums.json").read_text())
    assert rep["pass"] and rep["jobs"] == 2 and [r["name"] for r in rep["files"]] == ["hepg2_screen", "k562_screen"]
    for r in rep["files"]:
        p = Path(r["path"])
        assert r["md5_computed"] == hashlib.md5(p.read_bytes()).hexdigest() == r["md5_expected"] == r["md5_sidecar"]
        assert r["size"] == p.stat().st_size and r["mtime_ns"] == p.stat().st_mtime_ns and r["pass"]


@pytest.mark.parametrize("problem", ["config_md5", "sidecar", "bytes", "no_sidecar"])
def test_checksums_fail_on_any_disagreement(tmp_path, problem):
    cfg_path, _ = build_raw(tmp_path)
    raw = tmp_path / "raw" / "k562_raw.h5ad"
    sidecar = raw.with_name(raw.name + ".md5")
    over = []
    if problem == "config_md5":
        over = ["screens.1.md5=" + "0" * 32]
    elif problem == "sidecar":
        sidecar.write_text("0" * 32 + "  k562_raw.h5ad\n")
    elif problem == "bytes":  # same size, one byte flipped: config md5 and sidecar are both stale
        b = bytearray(raw.read_bytes())
        b[-1] ^= 1
        raw.write_bytes(bytes(b))
    else:
        sidecar.unlink()
    assert verify_merge.main(["checksums", "--config", str(cfg_path), *over]) == 1
    rep = json.loads((tmp_path / "out" / "checksums.json").read_text())
    assert not rep["pass"] and [r["name"] for r in rep["files"] if not r["pass"]] == ["k562_screen"]
    bad = rep["files"][1]
    want = {
        "config_md5": (False, True),
        "sidecar": (True, False),
        "bytes": (False, False),
        "no_sidecar": (True, None),
    }[problem]
    assert (bad["match_expected"], bad["match_sidecar"]) == want


def test_prepare_manifest_records_the_recomputed_md5(tmp_path):
    cfg_path, _ = build_raw(tmp_path)
    assert verify_merge.main(["checksums", "--config", str(cfg_path)]) == 0
    prepare_replogle_raw.main(["--config", str(cfg_path)])
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    for rec in m["inputs"]:
        assert rec["md5_computed"] == hashlib.md5(Path(rec["path"]).read_bytes()).hexdigest() == rec["md5"]
    # an entry for a file touched since is not used ...
    raw = tmp_path / "raw" / "hepg2_raw.h5ad"
    st = raw.stat()
    os.utime(raw, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    prepare_replogle_raw.main(["--config", str(cfg_path)])
    rec = json.loads((tmp_path / "out" / "manifest.json").read_text())["inputs"][0]
    assert rec["md5_computed"] is None and "md5_computed_note" in rec
    # ... and a current entry that disagrees stops prepare
    ck_path = tmp_path / "out" / "checksums.json"
    ck = json.loads(ck_path.read_text())
    ck["files"][0].update(mtime_ns=raw.stat().st_mtime_ns, md5_computed="0" * 32)
    ck_path.write_text(json.dumps(ck))
    with pytest.raises(RuntimeError, match="raw inputs not ready"):
        prepare_replogle_raw.main(["--config", str(cfg_path)])


def test_repo_config_records_where_each_md5_comes_from():
    import step_hash

    path = str(REPO / "experiments" / "configs" / "perturbation_direction" / "replogle_raw.yaml")
    cfg = prepare_replogle_raw.load_config(path)
    src = {s.name: s.md5_source for s in cfg.screens}
    assert set(src) == {"hepg2", "jurkat", "k562_essential", "rpe1"}
    assert "figshare" in src["k562_essential"] and "figshare" in src["rpe1"]
    assert "no published md5" in src["hepg2"] and "dfc676e4186b8c8c173ff14e1194cc8a" in src["hepg2"]
    assert "no published md5" in src["jurkat"]
    # provenance notes never make a step rerun; anything else in the config does
    edited = prepare_replogle_raw.load_config(path, ["screens.0.md5_source=edited"])
    assert step_hash.config_digest(cfg) == step_hash.config_digest(edited)
    assert step_hash.config_digest(cfg) != step_hash.config_digest(prepare_replogle_raw.load_config(path, ["block=1"]))


def test_code_closure_follows_every_pipeline_import(tmp_path):
    import step_hash

    (tmp_path / "a.py").write_text("import os\nimport b as bee\n\n\ndef f():\n    from c import x\n    return x\n")
    (tmp_path / "b.py").write_text("import numpy\n")
    (tmp_path / "c.py").write_text("from b import y\nx = 1\n")
    (tmp_path / "d.py").write_text("x = 2\n")
    code = step_hash.code_closure([tmp_path / "a.py"])
    assert set(code) == {"a.py", "b.py", "c.py"}
    assert code["c.py"] == hashlib.sha256((tmp_path / "c.py").read_bytes()).hexdigest()
    real = step_hash.code_closure([E / "verify_merge.py"])
    assert {"audit_raw_screen.py", "make_state_reference.py", "prepare_replogle_raw.py", "knockdown_filter.py"} < set(
        real
    )


def test_step_hash_changes_with_each_component(tmp_path):
    import step_hash

    f, up = tmp_path / "in.json", tmp_path / "up.done"
    f.write_text("{}")
    up.write_text("hash=1\n")
    script = tmp_path / "s.py"
    script.write_text("x = 1\n")

    def h(**kw):
        args = {"config_digest": "c", "argv": ["python", "s.py"], "code": [script], "files": [f], "upstream": [up]}
        return step_hash.step_hash(**{**args, **kw})[0]

    base = h()
    assert h() == base
    assert len({base, h(config_digest="d"), h(argv=["python", "s.py", "--x"]), h(files=[]), h(upstream=[])}) == 5
    script.write_text("x = 2\n")
    assert h() != base
    script.write_text("x = 1\n")
    up.write_text("hash=2\n")
    assert h() != base


def test_adopt_prepare_accepts_only_a_consistent_legacy_build(tmp_path):
    import step_hash

    cfg_path, _ = build_raw(tmp_path)
    assert verify_merge.main(["checksums", "--config", str(cfg_path)]) == 0
    prepare_replogle_raw.main(["--config", str(cfg_path)])

    def adopt(*over):
        return step_hash.adopt_prepare(prepare_replogle_raw.load_config(str(cfg_path), list(over)))

    rep = adopt()
    assert rep["adoptable"], rep["checks"]
    assert adopt("screens.0.md5_source=a note")["adoptable"]
    rep = adopt("knockdown.min_cells=31")
    assert not rep["adoptable"] and [k for k, v in rep["checks"].items() if not v["pass"]] == ["config"]
    out = tmp_path / "out" / "replogle_test_raw_all.h5ad"  # an output rewritten after the manifest
    st = out.stat()
    os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))
    rep = adopt()
    assert [k for k, v in rep["checks"].items() if not v["pass"]] == ["output_all_test"]
    os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns))
    (tmp_path / "out" / "checksums.json").unlink()  # no fresh evidence on the raw bytes
    rep = adopt()
    assert not rep["adoptable"] and not rep["checks"]["input_hepg2_screen"]["pass"]


def _legit_suffixes(ours, state, uni, n=7):
    """Make n cells of the later line carry a barcode that is a STATE cell of the earlier line, as in STATE:
    raw id = that barcode (ours + universe), STATE id = barcode + '-1' (anndata obs_names_make_unique)."""
    first, later = list(dict.fromkeys(state.cell_line))[:2]
    ours, state, uni = ours.copy(), state.copy(), {k: set(v) for k, v in uni.items()}
    base = state.cell_id[state.cell_line == first].iloc[:n].to_numpy()
    k = state.index[state.cell_line == later][:n]
    old = state.loc[k, "cell_id"].to_numpy()
    for o, b in zip(old, base, strict=True):
        ours.loc[(ours.cell_line == later) & (ours.cell_id == o), "cell_id"] = b
        uni[later].discard(o)
        uni[later].add(b)
    state.loc[k, "cell_id"] = [f"{b}-1" for b in base]
    return ours, state, uni, first, later, k


def test_make_unique_suffixed_state_ids_map_to_raw_ids(gate_inputs):
    """STATE renamed barcodes that repeat across screens to '<barcode>-1' (anndata obs_names_make_unique)."""
    ours, state, uni, _ = gate_inputs
    ours, state, uni, first, later, k = _legit_suffixes(ours, state, uni)
    rep, _ = verify_merge.identity(ours, state, universe=uni)
    assert rep["pass"]
    assert rep["mapping"][later]["method"] == "obs_index_make_unique"
    assert rep["mapping"][later]["make_unique_suffix_stripped"] == 7
    assert rep["mapping"][later]["suffix_base_is_state_id_of_earlier_line"] == 7
    assert rep["mapping"][first]["method"] == "obs_index"


def test_suffix_without_an_earlier_state_cell_fails(gate_inputs):
    """A '-1' on an id whose bare barcode is a raw id of the line but no STATE cell of an earlier line is
    not how STATE's make-unique arises: A1 must fail (A2/A3 alone would not catch the id provenance)."""
    ours, state, uni, _ = gate_inputs
    state = state.copy()
    later = list(dict.fromkeys(state.cell_line))[1]
    k = state.index[state.cell_line == later][:7]
    state.loc[k, "cell_id"] = state.loc[k, "cell_id"] + "-1"
    rep, table = verify_merge.identity(ours, state, universe=uni)
    assert rep["mapping"][later]["method"] == "obs_index_make_unique"
    assert rep["mapping"][later]["suffix_base_is_state_id_of_earlier_line"] == 0
    assert not rep["A1_cell_set_and_labels"]["per_line"][later]["pass"] and not rep["pass"]
    assert (table.gate == "A1_suffix_base").sum() == 7
    # a suffix whose base is not a raw id of the line is not stripped: that falls back to the key mapping,
    # which fails the gate unless it is explicitly allowed
    state.loc[k[0], "cell_id"] = "NOT-A-RAW-ID-1"
    rep, _ = verify_merge.identity(ours, state, universe=uni)
    assert rep["mapping"][later]["method"] == "key" and not rep["pass"]
    assert verify_merge.identity(ours, state, universe=uni, allow_key_mapping=True)[0]["pass"]
