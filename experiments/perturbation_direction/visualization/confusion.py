"""Three-class calibration diagnostics on the report's changed-versus-unchanged shared panel.

Rows are observed down/unchanged/up; columns are predicted down/unchanged/up.
The primary plot averages row-normalized matrices over perturbations (equal
weight per perturbation with that observed class). A pooled companion plot
weights gene–perturbation pairs instead. Neither is the report's AUROC or its
median three-class balanced accuracy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.perturbation_direction.dashboard.dashboard_data import load_effects, load_truth, resolve
from experiments.perturbation_direction.evaluation.freeze_results import sha256
from experiments.perturbation_direction.evaluation.metrics import unchanged_mask

CLASSES = ("Down", "Unchanged", "Up")


def confusion_counts(pred, truth, deg, unchanged, tau=0.1):
    """Integer 3×3 matrix; grey-zone genes are omitted and nonfinite scored predictions rejected."""
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    deg, unchanged = np.asarray(deg, dtype=bool), np.asarray(unchanged, dtype=bool)
    if not (pred.shape == truth.shape == deg.shape == unchanged.shape):
        raise ValueError("prediction, truth and masks must have identical shapes")
    if np.any(deg & unchanged):
        raise ValueError("changed and unchanged masks must not overlap")
    scored = deg | unchanged
    if not np.isfinite(truth[scored]).all() or np.any(deg & (truth == 0)):
        raise ValueError("scored truth must be finite, with nonzero effects for DEGs")
    if not np.isfinite(pred[scored]).all():
        raise ValueError("missing/nonfinite predictions on scored genes")
    observed = np.where(deg, np.where(truth < 0, 0, 2), 1)
    predicted = np.where(pred < -tau, 0, np.where(pred > tau, 2, 1))
    return np.bincount(3 * observed[scored] + predicted[scored], minlength=9).reshape(3, 3)


def normalize_rows(counts):
    """Normalize along predicted classes; an absent observed class yields NaN, not zero."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)


def summarize_counts(counts):
    """Return macro row probabilities, pooled row probabilities, counts and row-support counts."""
    counts = np.asarray(counts)
    probabilities = normalize_rows(counts)
    support = (counts.sum(axis=2) > 0).sum(axis=0)
    macro = np.divide(
        np.nansum(probabilities, axis=0),
        support[:, None],
        out=np.full((3, 3), np.nan),
        where=support[:, None] > 0,
    )
    pooled_counts = counts.sum(axis=0)
    return macro, normalize_rows(pooled_counts), pooled_counts, support


def plot_matrices(matrices, supports, path, title):
    """Shared color scale and true-class row normalization; export both vector PDF and PNG."""
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(
            1, len(matrices), figsize=(3.7 * len(matrices) + 0.5, 3.9), squeeze=False, layout="constrained"
        )
        for ax, (name, matrix) in zip(axes[0], matrices.items(), strict=True):
            image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    value = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        f"{100 * value:.1f}%" if np.isfinite(value) else "—",
                        ha="center",
                        va="center",
                        color="white" if value > 0.55 else "black",
                    )
            ax.set_xticks(range(3), CLASSES)
            ax.set_yticks(range(3), [f"{label}\n(n={n})" for label, n in zip(CLASSES, supports[name], strict=True)])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Observed class")
            ax.set_title({"scLDM_s0": "scLDM (seed 0)", "knn": "kNN", "generic": "Generic"}.get(name, name))
        fig.suptitle(title)
        fig.colorbar(image, ax=list(axes[0]), label="Row proportion", shrink=0.8)
        for suffix in (".pdf", ".png"):
            fig.savefig(path.with_suffix(suffix), dpi=220)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    eval_dir, out = pathlib.Path(cfg.eval_dir), pathlib.Path(cfg.out_dir)
    manifest_path = eval_dir / "methods.json"
    manifest = json.loads(manifest_path.read_text())
    settings = manifest.get("unchanged")
    if settings is None:
        raise ValueError("evaluation manifest lacks changed-versus-unchanged settings")
    names = list(cfg.methods)
    if not names or len(set(names)) != len(names):
        raise ValueError("methods must be a nonempty list of unique names")
    if set(names) & set(manifest.get("direction_only", [])):
        raise ValueError("direction-only predictions cannot be thresholded as effect sizes")
    truth_path = resolve(manifest["truth_dir"])
    truth = load_truth(truth_path)
    if truth.padj is None:
        raise ValueError("padj.parquet is required to define unchanged genes")
    panel = pd.read_csv(cfg.panel_csv)
    selected = panel.loc[(panel["report"] == "unchanged") & panel["shared"], "perturbation"].tolist()
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("shared unchanged panel must be nonempty and unique")
    missing = set(selected) - set(truth.delta.index)
    if missing:
        raise ValueError(f"shared panel missing from truth: {sorted(missing)}")
    delta = truth.delta.loc[selected]
    deg = truth.deg.reindex(index=selected, columns=delta.columns).astype(bool)
    unchanged = unchanged_mask(delta, truth.padj, deg, settings["max_effect"], settings["min_padj"])
    if manifest.get("exclude_target", True):
        for perturbation in selected:
            if perturbation in deg.columns:
                deg.loc[perturbation, perturbation] = False
                unchanged.loc[perturbation, perturbation] = False
    if (deg.sum(axis=1) < manifest["min_degs"]).any() or (unchanged.sum(axis=1) < settings["min_unchanged"]).any():
        raise ValueError("panel is incompatible with this evaluation's truth or thresholds")

    macro, pooled, macro_support, pooled_support, rows = {}, {}, {}, {}, []
    inputs = {str(manifest_path), str(pathlib.Path(cfg.panel_csv))}
    inputs.update(str(truth_path / name) for name in ("delta_true.parquet", "deg.parquet", "padj.parquet", "info.json"))
    for name in names:
        entry = next(m for m in manifest["methods"] if m["name"] == name)
        if entry["kind"] == "shuffled":
            raise ValueError("select unshuffled effect-size methods for this calibration figure")
        inputs.add(str(resolve(entry)))
        effects = load_effects(manifest, name, truth).reindex(index=selected, columns=delta.columns)
        counts = np.stack(
            [
                confusion_counts(effects.loc[p], delta.loc[p], deg.loc[p], unchanged.loc[p], settings["tau"])
                for p in selected
            ]
        )
        macro[name], pooled[name], total, support = summarize_counts(counts)
        macro_support[name], pooled_support[name] = support, total.sum(axis=1)
        for k, perturbation in enumerate(selected):
            for i, observed in enumerate(CLASSES):
                for j, predicted in enumerate(CLASSES):
                    rows.append(
                        {
                            "method": name,
                            "perturbation": perturbation,
                            "observed": observed,
                            "predicted": predicted,
                            "count": int(counts[k, i, j]),
                        }
                    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_perturbation_counts.csv", index=False)
    summary = [
        {
            "method": name,
            "observed": CLASSES[i],
            "predicted": CLASSES[j],
            "macro_proportion": macro[name][i, j],
            "pooled_proportion": pooled[name][i, j],
            "n_perturbations_with_class": int(macro_support[name][i]),
            "n_gene_pairs_with_class": int(pooled_support[name][i]),
        }
        for name in names
        for i in range(3)
        for j in range(3)
    ]
    pd.DataFrame(summary).to_csv(out / "confusion_summary.csv", index=False)
    provenance = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "settings": settings,
        "exclude_target": manifest.get("exclude_target", True),
        "panel": selected,
        "inputs_sha256": {p: sha256(pathlib.Path(p)) for p in sorted(inputs)},
        "note": "Rows observed, columns predicted; grey zone excluded. Macro averages row probabilities, not medians.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_matrices(macro, macro_support, out / "confusion_macro", "Three-class calibration · equal perturbation weights")
    plot_matrices(
        pooled, pooled_support, out / "confusion_pooled", "Three-class calibration · pooled gene–perturbation pairs"
    )
    print(f"Wrote confusion matrices for {len(selected)} perturbations to {out}")


if __name__ == "__main__":
    main()
